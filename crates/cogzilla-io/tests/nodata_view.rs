use std::path::PathBuf;
use std::sync::Arc;

use cogzilla_io::CogStore;

/// A `?nodata=` view of an opened raster: the file itself when nothing changes, else one view
/// per value with an id of its own, its parent's source id, and the parent's IFD tables shared.
#[tokio::test]
async fn nodata_view_shares_the_parent_and_keeps_its_source() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../benches/fixtures/landsat_c2_B4.tif");
    if !path.exists() {
        return;
    }
    let uri = path.canonicalize().unwrap().to_string_lossy().to_string();
    let store = CogStore::new(16 * 1024 * 1024);
    let cog = store.open_uri(&uri).await.unwrap();
    assert_eq!(cog.index.nodata, Some(0.0), "the fixture's own nodata");
    assert_eq!(cog.source_id, cog.id);

    assert!(Arc::ptr_eq(&store.with_nodata(&cog, None), &cog), "no override");
    assert!(Arc::ptr_eq(&store.with_nodata(&cog, Some(0.0)), &cog), "the file's own value");

    let view = store.with_nodata(&cog, Some(1000.0));
    assert_ne!(view.id, cog.id, "decoded apart from the file's blocks");
    assert_eq!(view.source_id, cog.id, "a change found through the view evicts the parent");
    assert_eq!(view.index.nodata, Some(1000.0));
    assert_eq!(view.index.nodata_f64, Some(1000.0));
    assert!(Arc::ptr_eq(&view.index.ifds, &cog.index.ifds), "the tile tables are not copied");
    assert!(Arc::ptr_eq(&store.with_nodata(&cog, Some(1000.0)), &view), "one view per value");
    assert!(!Arc::ptr_eq(&store.with_nodata(&cog, Some(7.0)), &view));

    store.clear_open_uris();
    let after = store.with_nodata(&cog, Some(1000.0));
    assert!(!Arc::ptr_eq(&after, &view), "cleared with the open rasters");
    assert_eq!(after.source_id, cog.id);
}

/// A tiled uint16 TIFF of two 16x16 tiles with `GDAL_NODATA=0`: the first left unwritten
/// (`TileByteCounts` 0), the second all 7s.
fn write_sparse_u16(path: &std::path::Path) {
    let entries: [(u16, u16, u32, u32); 12] = [
        (256, 3, 1, 32),  // ImageWidth
        (257, 3, 1, 16),  // ImageLength
        (258, 3, 1, 16),  // BitsPerSample
        (259, 3, 1, 1),   // Compression: none
        (262, 3, 1, 1),   // MinIsBlack
        (277, 3, 1, 1),   // SamplesPerPixel
        (284, 3, 1, 1),   // PlanarConfiguration: chunky
        (322, 3, 1, 16),  // TileWidth
        (323, 3, 1, 16),  // TileLength
        (324, 4, 2, 0),   // TileOffsets (filled below)
        (325, 4, 2, 0),   // TileByteCounts (filled below)
        (42113, 2, 2, u32::from_le_bytes([b'0', 0, 0, 0])), // GDAL_NODATA "0"
    ];
    let ifd_len = 2 + entries.len() * 12 + 4;
    let offsets_at = 8 + ifd_len as u32;
    let counts_at = offsets_at + 8;
    let data_at = counts_at + 8;
    let mut buf = Vec::new();
    buf.extend_from_slice(b"II");
    buf.extend_from_slice(&42u16.to_le_bytes());
    buf.extend_from_slice(&8u32.to_le_bytes());
    buf.extend_from_slice(&(entries.len() as u16).to_le_bytes());
    for (tag, ty, count, val) in entries {
        let val = match tag {
            324 => offsets_at,
            325 => counts_at,
            _ => val,
        };
        buf.extend_from_slice(&tag.to_le_bytes());
        buf.extend_from_slice(&ty.to_le_bytes());
        buf.extend_from_slice(&count.to_le_bytes());
        buf.extend_from_slice(&val.to_le_bytes());
    }
    buf.extend_from_slice(&0u32.to_le_bytes());
    for v in [0u32, data_at, 0, 16 * 16 * 2] {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    for _ in 0..16 * 16 {
        buf.extend_from_slice(&7u16.to_le_bytes());
    }
    std::fs::write(path, buf).unwrap();
}

/// GDAL reads an unwritten tile as the file's nodata; once `?nodata=` names another value that
/// is data, as the stored pixels equal to the file's nodata are.
#[tokio::test]
async fn nodata_view_reads_unwritten_tiles_as_the_files_nodata() {
    let dir = std::env::temp_dir().join(format!("cogzilla_sparse_view_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("sparse_u16.tif");
    write_sparse_u16(&path);
    let store = CogStore::new(16 * 1024 * 1024);
    let cog = store.register_local("sparse", &path).unwrap();
    let tiles = [Some(0), Some(1)];

    let own = store.get_blocks(&cog, 0, &tiles).await.unwrap();
    assert!(own[0].is_none(), "the file's read: the unwritten tile is no data");
    assert_eq!(own[1].as_ref().unwrap().bands[0].data.get_f32(0), 7.0);

    let view = store.with_nodata(&cog, Some(7.0));
    assert_eq!(view.sparse_fill, Some(0.0));
    let blocks = store.get_blocks(&view, 0, &tiles).await.unwrap();
    let fill = &blocks[0].as_ref().expect("the unwritten tile is the file's nodata, now data").bands[0];
    assert!(fill.stats.no_nodata && !fill.stats.all_nodata);
    assert_eq!((fill.width, fill.height, fill.data.get_f32(0), fill.data.get_f32(255)), (16, 16, 0.0, 0.0));
    let sevens = &blocks[1].as_ref().unwrap().bands[0];
    assert!(sevens.data.get_f32(0).is_nan(), "7 is the view's nodata");

    // A NaN nodata on integers masks nothing: the file's 0s, written or not, are data.
    assert_eq!(store.with_nodata(&cog, Some(f32::NAN)).sparse_fill, Some(0.0));
    let _ = std::fs::remove_dir_all(&dir);
}

/// The fill tile of an unwritten tile is one per raster and IFD, shared by every view of it.
#[tokio::test]
async fn nodata_views_share_one_fill_tile() {
    let dir = std::env::temp_dir().join(format!("cogzilla_sparse_share_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("sparse_u16.tif");
    write_sparse_u16(&path);
    let store = CogStore::new(16 * 1024 * 1024);
    let cog = store.register_local("sparse", &path).unwrap();
    let a = store.get_blocks(&store.with_nodata(&cog, Some(7.0)), 0, &[Some(0)]).await.unwrap();
    let b = store.get_blocks(&store.with_nodata(&cog, Some(5.0)), 0, &[Some(0)]).await.unwrap();
    assert!(Arc::ptr_eq(a[0].as_ref().unwrap(), b[0].as_ref().unwrap()));
    assert!(store.cache.current_bytes() >= 16 * 16 * 4, "held in the block cache, under its budget");
    let _ = std::fs::remove_dir_all(&dir);
}

/// Per GDAL `SPARSE_OK`, an unwritten sparse tile in a TIFF with NO `GDAL_NODATA` and no mask/alpha
/// must decode as valid `0.0` pixels (not masked), while `?nodata=0` masks it.
#[tokio::test]
async fn sparse_tiff_without_nodata_reads_unwritten_tiles_as_valid_zeros() {
    let dir = std::env::temp_dir().join(format!("cogzilla_sparse_no_nd_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("sparse_u16_no_nd.tif");
    let entries: [(u16, u16, u32, u32); 11] = [
        (256, 3, 1, 32), // ImageWidth
        (257, 3, 1, 16), // ImageLength
        (258, 3, 1, 16), // BitsPerSample
        (259, 3, 1, 1),  // Compression: none
        (262, 3, 1, 1),  // MinIsBlack
        (277, 3, 1, 1),  // SamplesPerPixel
        (284, 3, 1, 1),  // PlanarConfiguration: chunky
        (322, 3, 1, 16), // TileWidth
        (323, 3, 1, 16), // TileLength
        (324, 4, 2, 0),  // TileOffsets
        (325, 4, 2, 0),  // TileByteCounts
    ];
    let ifd_len = 2 + entries.len() * 12 + 4;
    let offsets_at = 8 + ifd_len as u32;
    let counts_at = offsets_at + 8;
    let data_at = counts_at + 8;
    let mut buf = Vec::new();
    buf.extend_from_slice(b"II");
    buf.extend_from_slice(&42u16.to_le_bytes());
    buf.extend_from_slice(&8u32.to_le_bytes());
    buf.extend_from_slice(&(entries.len() as u16).to_le_bytes());
    for (tag, ty, count, val) in entries {
        let val = match tag {
            324 => offsets_at,
            325 => counts_at,
            _ => val,
        };
        buf.extend_from_slice(&tag.to_le_bytes());
        buf.extend_from_slice(&ty.to_le_bytes());
        buf.extend_from_slice(&count.to_le_bytes());
        buf.extend_from_slice(&val.to_le_bytes());
    }
    buf.extend_from_slice(&0u32.to_le_bytes());
    for v in [0u32, data_at, 0, 16 * 16 * 2] {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    for _ in 0..16 * 16 {
        buf.extend_from_slice(&7u16.to_le_bytes());
    }
    std::fs::write(&path, buf).unwrap();

    let store = CogStore::new(16 * 1024 * 1024);
    let cog = store.register_local("sparse_no_nd", &path).unwrap();
    assert_eq!(cog.sparse_fill, Some(0.0));

    let blocks = store.get_blocks(&cog, 0, &[Some(0), Some(1)]).await.unwrap();
    let fill = &blocks[0].as_ref().expect("unwritten tile without nodata must be valid 0s").bands[0];
    assert!(fill.stats.no_nodata && !fill.stats.all_nodata);
    assert_eq!((fill.data.get_f32(0), fill.data.get_f32(255)), (0.0, 0.0));
    assert_eq!(blocks[1].as_ref().unwrap().bands[0].data.get_f32(0), 7.0);

    let nd0 = store.with_nodata(&cog, Some(0.0));
    assert_eq!(nd0.sparse_fill, None);
    let masked = store.get_blocks(&nd0, 0, &[Some(0)]).await.unwrap();
    assert!(masked[0].is_none(), "?nodata=0 masks the unwritten 0-filled tile");

    let _ = std::fs::remove_dir_all(&dir);
}
