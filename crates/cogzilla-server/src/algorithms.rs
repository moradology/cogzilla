//! Single authoritative registry of supported TiTiler / cogzilla raster algorithms.
//!
//! Every downstream site — `/algorithms` and `/algorithms/{id}` metadata, query validation,
//! pydantic-compatible `algorithm_params` validation, output dtype inference (`array_type`),
//! DEM margin / output band count queries, `http.rs` `BandRequest` dispatch, and `pgstac.rs`
//! `BandAlgorithm` dispatch — derives its behaviour from [`ALGORITHMS`] and [`AlgorithmId`].

use cogzilla_raster::SampleType;
use serde_json::{Map, Value, json};

use crate::engine::EngineError;

/// Canonical identifier for every supported raster algorithm.
///
/// Exhaustive matches on `AlgorithmId` ensure adding a new entry to [`ALGORITHMS`]
/// cannot silently be dropped by either the COG/STAC or pgSTAC route stacks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AlgorithmId {
    Hillshade,
    Slope,
    Aspect,
    Terrarium,
    TerrainRgb,
    NormalizedIndex,
    Cast,
    Ceil,
    Floor,
    Min,
    Max,
    Median,
    Mean,
    Std,
    Var,
    Sum,
    Grayscale,
    Bitonal,
    Contours,
}

/// Dispatch family for an algorithm: neighbourhood/terrain DEM kernel vs per-pixel band math.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AlgorithmKind {
    Dem,
    BandMath,
}

/// How an algorithm parameter is typed for pydantic-lax validation and `/algorithms/{id}` JSON schema.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParamKind {
    /// Strict integer in both validation and JSON schema (`"type": "integer"`).
    Int,
    /// Floating-point number in both validation and JSON schema (`"type": "number"`).
    Float,
    /// Accepts fractional numbers in validation (wider than TiTiler), advertised as `"integer"` in JSON schema.
    FloatSchemaInt,
    /// Optional floating-point number (`number | null`).
    OptionalFloat,
}

/// Default value advertised in `/algorithms/{id}` JSON schema.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ParamDefault {
    Int(i64),
    Float(f64),
    Null,
}

/// Schema + validation metadata for a single `algorithm_params` field.
#[derive(Clone, Copy, Debug)]
pub struct ParamDef {
    pub name: &'static str,
    pub title: &'static str,
    pub description: Option<&'static str>,
    pub kind: ParamKind,
    pub default: ParamDefault,
    pub min: f64,
    pub max: f64,
}

/// Output sample-type rule for an algorithm, shared between `/algorithms/{id}` and `array_type`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputDtype {
    /// Always outputs this fixed sample type (`uint8`, `float32`, `float64`).
    Fixed(SampleType),
    /// Preserves the input bands' sample type (`min`, `max`, `grayscale`).
    SameAsInput,
    /// Widens integer types to 64-bit (`sum`), preserving floats.
    SumWiden,
    /// Keeps `float32`, promotes everything else to `float64` (`median`).
    MedianWiden,
    /// Determined by `algorithm_params.dtype` (`cast`, default `uint8`).
    CastParam,
    /// Cogzilla-native `aspect`: advertises `"float32"` in schema, leaves array retyping untouched.
    AspectNative,
}

impl OutputDtype {
    /// JSON schema `outputs.dtype` string, or `None` for JSON `null`.
    pub const fn schema_name(self) -> Option<&'static str> {
        match self {
            Self::Fixed(SampleType::Uint8) | Self::CastParam => Some("uint8"),
            Self::Fixed(SampleType::Float32) | Self::AspectNative => Some("float32"),
            Self::Fixed(SampleType::Float64) | Self::MedianWiden => Some("float64"),
            Self::Fixed(_) | Self::SameAsInput | Self::SumWiden => None,
        }
    }

    /// Resolve the concrete output [`SampleType`] for raw array responses (`npy`/`tif`).
    pub const fn resolve(self, cast: Option<SampleType>, read: SampleType) -> Option<SampleType> {
        use SampleType as T;
        match self {
            Self::CastParam => cast,
            Self::Fixed(t) => Some(t),
            Self::SameAsInput => Some(read),
            Self::SumWiden => Some(match read {
                T::Bool | T::Uint1 | T::Uint4 | T::Uint8 | T::Uint12 | T::Uint16 | T::Uint32 | T::Uint64 => T::Uint64,
                T::Int8 | T::Int16 | T::Int32 | T::Int64 => T::Int64,
                T::Float16 => T::Float32,
                T::Float32 | T::Float64 => read,
            }),
            Self::MedianWiden => Some(if matches!(read, T::Float32) { T::Float32 } else { T::Float64 }),
            Self::AspectNative => None,
        }
    }
}

/// Complete definition of one supported algorithm.
#[derive(Clone, Copy, Debug)]
pub struct AlgorithmDef {
    pub id: AlgorithmId,
    /// Canonical id advertised in `/algorithms` and `/algorithms/{id}`.
    pub name: &'static str,
    /// Additional accepted query-string aliases (matched case-insensitively).
    pub aliases: &'static [&'static str],
    /// Human-readable title in `/algorithms/{id}`.
    pub title: &'static str,
    /// Pydantic model name used in validation errors (when this algorithm has parameters).
    pub pydantic_name: Option<&'static str>,
    pub description: &'static str,
    pub input_bands: Option<u8>,
    /// Fixed output band count (`None` when it preserves the input band count).
    pub out_bands: Option<u8>,
    pub out_dtype: OutputDtype,
    pub out_min: Option<f64>,
    pub out_max: Option<f64>,
    /// Default edge-buffer crop subtracted from `?buffer=` on DEM stencil algorithms (`Some(3.0)`
    /// for `hillshade`/`slope`/`aspect`, `None` for non-stencil algorithms).
    pub dem_margin: Option<f32>,
    pub kind: AlgorithmKind,
    pub params: &'static [ParamDef],
}

impl AlgorithmDef {
    /// Whether `query` matches this algorithm's canonical name or any of its aliases (ASCII case-insensitive).
    pub fn matches_name(&self, query: &str) -> bool {
        self.name.eq_ignore_ascii_case(query)
            || self.aliases.iter().any(|alias| alias.eq_ignore_ascii_case(query))
    }

    /// Build the TiTiler 2.4 `GET /algorithms/{algorithmId}` JSON payload from this definition.
    pub fn to_metadata_json(&self) -> Value {
        let mut parameters = Map::new();
        for p in self.params {
            let mut obj = Map::new();
            match p.kind {
                ParamKind::OptionalFloat => {
                    obj.insert(
                        "anyOf".into(),
                        json!([
                            { "maximum": p.max, "minimum": p.min, "type": "number" },
                            { "type": "null" }
                        ]),
                    );
                    obj.insert("default".into(), default_to_json(p.default));
                    obj.insert("title".into(), Value::String(p.title.into()));
                }
                ParamKind::Int | ParamKind::FloatSchemaInt => {
                    obj.insert("default".into(), default_to_json(p.default));
                    if let Some(desc) = p.description {
                        obj.insert("description".into(), Value::String(desc.into()));
                    }
                    obj.insert("maximum".into(), json!(p.max as i64));
                    obj.insert("minimum".into(), json!(p.min as i64));
                    obj.insert("title".into(), Value::String(p.title.into()));
                    obj.insert("type".into(), Value::String("integer".into()));
                }
                ParamKind::Float => {
                    obj.insert("default".into(), default_to_json(p.default));
                    if let Some(desc) = p.description {
                        obj.insert("description".into(), Value::String(desc.into()));
                    }
                    obj.insert("maximum".into(), json!(p.max));
                    obj.insert("minimum".into(), json!(p.min));
                    obj.insert("title".into(), Value::String(p.title.into()));
                    obj.insert("type".into(), Value::String("number".into()));
                }
            }
            parameters.insert(p.name.into(), Value::Object(obj));
        }

        let min_json = self.out_min.map(|v| json!([v])).unwrap_or(Value::Null);
        let max_json = self.out_max.map(|v| json!([v])).unwrap_or(Value::Null);

        json!({
            "title": self.title,
            "description": self.description,
            "inputs": { "nbands": self.input_bands },
            "outputs": {
                "nbands": self.out_bands,
                "dtype": self.out_dtype.schema_name(),
                "min": min_json,
                "max": max_json,
            },
            "parameters": Value::Object(parameters),
        })
    }
}

fn default_to_json(d: ParamDefault) -> Value {
    match d {
        ParamDefault::Int(i) => json!(i),
        ParamDefault::Float(f) => json!(f),
        ParamDefault::Null => Value::Null,
    }
}

const HILLSHADE_PARAMS: &[ParamDef] = &[
    ParamDef {
        name: "azimuth",
        title: "Azimuth",
        description: None,
        kind: ParamKind::Int,
        default: ParamDefault::Int(45),
        min: 0.0,
        max: 360.0,
    },
    ParamDef {
        name: "angle_altitude",
        title: "Angle Altitude",
        description: None,
        kind: ParamKind::Float,
        default: ParamDefault::Float(45.0),
        min: -90.0,
        max: 90.0,
    },
    ParamDef {
        name: "buffer",
        title: "Buffer",
        description: None,
        kind: ParamKind::Int,
        default: ParamDefault::Int(3),
        min: 0.0,
        max: 99.0,
    },
    ParamDef {
        name: "z_exaggeration",
        title: "Z Exaggeration",
        description: None,
        kind: ParamKind::Float,
        default: ParamDefault::Float(1.0),
        min: 1e-6,
        max: 1_000_000.0,
    },
];

const SLOPE_PARAMS: &[ParamDef] = &[
    ParamDef {
        name: "buffer",
        title: "Buffer",
        description: Some("Buffer size for edge effects"),
        kind: ParamKind::Int,
        default: ParamDefault::Int(3),
        min: 0.0,
        max: 99.0,
    },
    ParamDef {
        name: "z_exaggeration",
        title: "Z Exaggeration",
        description: None,
        kind: ParamKind::Float,
        default: ParamDefault::Float(1.0),
        min: 1e-6,
        max: 1_000_000.0,
    },
];

const ASPECT_PARAMS: &[ParamDef] = &[
    ParamDef {
        name: "buffer",
        title: "Buffer",
        description: Some("Buffer size for edge effects"),
        kind: ParamKind::Int,
        default: ParamDefault::Int(3),
        min: 0.0,
        max: 99.0,
    },
    ParamDef {
        name: "z_exaggeration",
        title: "Z Exaggeration",
        description: None,
        kind: ParamKind::Float,
        default: ParamDefault::Float(1.0),
        min: 1e-6,
        max: 1_000_000.0,
    },
];

const TERRARIUM_PARAMS: &[ParamDef] = &[ParamDef {
    name: "nodata_height",
    title: "Nodata Height",
    description: None,
    kind: ParamKind::OptionalFloat,
    default: ParamDefault::Null,
    min: -99_999.0,
    max: 99_999.0,
}];

const TERRAINRGB_PARAMS: &[ParamDef] = &[
    ParamDef {
        name: "interval",
        title: "Interval",
        description: None,
        kind: ParamKind::Float,
        default: ParamDefault::Float(0.1),
        min: 0.0,
        max: 1.0,
    },
    ParamDef {
        name: "baseval",
        title: "Baseval",
        description: None,
        kind: ParamKind::Float,
        default: ParamDefault::Float(-10_000.0),
        min: -99_999.0,
        max: 99_999.0,
    },
    ParamDef {
        name: "nodata_height",
        title: "Nodata Height",
        description: None,
        kind: ParamKind::OptionalFloat,
        default: ParamDefault::Null,
        min: -99_999.0,
        max: 99_999.0,
    },
];

const CONTOURS_PARAMS: &[ParamDef] = &[
    ParamDef {
        name: "increment",
        title: "Increment",
        description: None,
        // TiTiler's is a whole number; a fraction is accepted here for data whose values lie
        // between whole numbers (reflectance, say).
        kind: ParamKind::FloatSchemaInt,
        default: ParamDefault::Int(35),
        min: 0.0,
        max: 999.0,
    },
    ParamDef {
        name: "thickness",
        title: "Thickness",
        description: None,
        kind: ParamKind::Int,
        default: ParamDefault::Int(1),
        min: 0.0,
        max: 10.0,
    },
    ParamDef {
        name: "minz",
        title: "Minz",
        description: None,
        kind: ParamKind::Int,
        default: ParamDefault::Int(-12_000),
        min: -99_999.0,
        max: 99_999.0,
    },
    ParamDef {
        name: "maxz",
        title: "Maxz",
        description: None,
        kind: ParamKind::Int,
        default: ParamDefault::Int(8_000),
        min: -99_999.0,
        max: 99_999.0,
    },
];

/// The single authoritative table of all supported algorithms.
pub const ALGORITHMS: &[AlgorithmDef] = &[
    AlgorithmDef {
        id: AlgorithmId::Hillshade,
        name: "hillshade",
        aliases: &[],
        title: "Hillshade",
        pydantic_name: Some("HillShade"),
        description: "Create hillshade from DEM dataset.",
        input_bands: Some(1),
        out_bands: Some(1),
        out_dtype: OutputDtype::Fixed(SampleType::Uint8),
        out_min: None,
        out_max: None,
        dem_margin: Some(3.0),
        kind: AlgorithmKind::Dem,
        params: HILLSHADE_PARAMS,
    },
    AlgorithmDef {
        id: AlgorithmId::Slope,
        name: "slope",
        aliases: &[],
        title: "Slope",
        pydantic_name: Some("Slope"),
        description: "Calculate degrees of slope from DEM dataset.",
        input_bands: Some(1),
        out_bands: Some(1),
        out_dtype: OutputDtype::Fixed(SampleType::Float32),
        out_min: Some(0.0),
        out_max: Some(90.0),
        dem_margin: Some(3.0),
        kind: AlgorithmKind::Dem,
        params: SLOPE_PARAMS,
    },
    AlgorithmDef {
        id: AlgorithmId::Aspect,
        name: "aspect",
        aliases: &[],
        title: "Aspect",
        pydantic_name: Some("Aspect"),
        description: "Calculate compass aspect in degrees (0-360) from DEM dataset.",
        input_bands: Some(1),
        out_bands: Some(1),
        out_dtype: OutputDtype::AspectNative,
        out_min: Some(0.0),
        out_max: Some(360.0),
        dem_margin: Some(3.0),
        kind: AlgorithmKind::Dem,
        params: ASPECT_PARAMS,
    },
    AlgorithmDef {
        id: AlgorithmId::Terrarium,
        name: "terrarium",
        aliases: &[],
        title: "Terrarium",
        pydantic_name: Some("Terrarium"),
        description: "Encode DEM into RGB (Mapzen Terrarium).",
        input_bands: Some(1),
        out_bands: Some(3),
        out_dtype: OutputDtype::Fixed(SampleType::Uint8),
        out_min: None,
        out_max: None,
        dem_margin: None,
        kind: AlgorithmKind::Dem,
        params: TERRARIUM_PARAMS,
    },
    AlgorithmDef {
        id: AlgorithmId::TerrainRgb,
        name: "terrainrgb",
        aliases: &["terrain-rgb", "mapbox"],
        title: "TerrainRGB",
        pydantic_name: Some("TerrainRGB"),
        description: "Encode DEM into RGB (Mapbox Terrain RGB).",
        input_bands: Some(1),
        out_bands: Some(3),
        out_dtype: OutputDtype::Fixed(SampleType::Uint8),
        out_min: None,
        out_max: None,
        dem_margin: None,
        kind: AlgorithmKind::Dem,
        params: TERRAINRGB_PARAMS,
    },
    AlgorithmDef {
        id: AlgorithmId::NormalizedIndex,
        name: "normalizedIndex",
        aliases: &[],
        title: "Normalized Difference Index",
        pydantic_name: None,
        description: "Compute normalized difference index from two bands.",
        input_bands: Some(2),
        out_bands: Some(1),
        out_dtype: OutputDtype::Fixed(SampleType::Float32),
        out_min: Some(-1.0),
        out_max: Some(1.0),
        dem_margin: None,
        kind: AlgorithmKind::BandMath,
        params: &[],
    },
    AlgorithmDef {
        id: AlgorithmId::Cast,
        name: "cast",
        aliases: &[],
        title: "Cast data to Integer",
        pydantic_name: None,
        description: "Cast data to Integer.",
        input_bands: None,
        out_bands: None,
        out_dtype: OutputDtype::CastParam,
        out_min: Some(0.0),
        out_max: Some(255.0),
        dem_margin: None,
        kind: AlgorithmKind::BandMath,
        params: &[],
    },
    AlgorithmDef {
        id: AlgorithmId::Ceil,
        name: "ceil",
        aliases: &[],
        title: "Round data to the smallest integer",
        pydantic_name: None,
        description: "Round data to the smallest integer.",
        input_bands: None,
        out_bands: None,
        out_dtype: OutputDtype::Fixed(SampleType::Uint8),
        out_min: Some(0.0),
        out_max: Some(255.0),
        dem_margin: None,
        kind: AlgorithmKind::BandMath,
        params: &[],
    },
    AlgorithmDef {
        id: AlgorithmId::Floor,
        name: "floor",
        aliases: &[],
        title: "Round data to the largest integer",
        pydantic_name: None,
        description: "Round data to the largest integer.",
        input_bands: None,
        out_bands: None,
        out_dtype: OutputDtype::Fixed(SampleType::Uint8),
        out_min: Some(0.0),
        out_max: Some(255.0),
        dem_margin: None,
        kind: AlgorithmKind::BandMath,
        params: &[],
    },
    AlgorithmDef {
        id: AlgorithmId::Min,
        name: "min",
        aliases: &[],
        title: "Minimum",
        pydantic_name: None,
        description: "Return the Minimum along the `bands` axis.",
        input_bands: None,
        out_bands: Some(1),
        out_dtype: OutputDtype::SameAsInput,
        out_min: None,
        out_max: None,
        dem_margin: None,
        kind: AlgorithmKind::BandMath,
        params: &[],
    },
    AlgorithmDef {
        id: AlgorithmId::Max,
        name: "max",
        aliases: &[],
        title: "Maximum",
        pydantic_name: None,
        description: "Return the Maximum along the `bands` axis.",
        input_bands: None,
        out_bands: Some(1),
        out_dtype: OutputDtype::SameAsInput,
        out_min: None,
        out_max: None,
        dem_margin: None,
        kind: AlgorithmKind::BandMath,
        params: &[],
    },
    AlgorithmDef {
        id: AlgorithmId::Median,
        name: "median",
        aliases: &[],
        title: "Median",
        pydantic_name: None,
        description: "Return the Median along the `bands` axis.",
        input_bands: None,
        out_bands: Some(1),
        out_dtype: OutputDtype::MedianWiden,
        out_min: None,
        out_max: None,
        dem_margin: None,
        kind: AlgorithmKind::BandMath,
        params: &[],
    },
    AlgorithmDef {
        id: AlgorithmId::Mean,
        name: "mean",
        aliases: &[],
        title: "Mean",
        pydantic_name: None,
        description: "Return the Mean along the `bands` axis.",
        input_bands: None,
        out_bands: Some(1),
        out_dtype: OutputDtype::Fixed(SampleType::Float64),
        out_min: None,
        out_max: None,
        dem_margin: None,
        kind: AlgorithmKind::BandMath,
        params: &[],
    },
    AlgorithmDef {
        id: AlgorithmId::Std,
        name: "std",
        aliases: &[],
        title: "Standard deviation",
        pydantic_name: None,
        description: "Return the Standard Deviation along the `bands` axis.",
        input_bands: None,
        out_bands: Some(1),
        out_dtype: OutputDtype::Fixed(SampleType::Float64),
        out_min: None,
        out_max: None,
        dem_margin: None,
        kind: AlgorithmKind::BandMath,
        params: &[],
    },
    AlgorithmDef {
        id: AlgorithmId::Var,
        name: "var",
        aliases: &[],
        title: "Variance",
        pydantic_name: None,
        description: "Return Variance along the `bands` axis.",
        input_bands: None,
        out_bands: Some(1),
        out_dtype: OutputDtype::Fixed(SampleType::Float64),
        out_min: None,
        out_max: None,
        dem_margin: None,
        kind: AlgorithmKind::BandMath,
        params: &[],
    },
    AlgorithmDef {
        id: AlgorithmId::Sum,
        name: "sum",
        aliases: &[],
        title: "Sum",
        pydantic_name: None,
        description: "Return the Sum along the `bands` axis.",
        input_bands: None,
        out_bands: Some(1),
        out_dtype: OutputDtype::SumWiden,
        out_min: None,
        out_max: None,
        dem_margin: None,
        kind: AlgorithmKind::BandMath,
        params: &[],
    },
    AlgorithmDef {
        id: AlgorithmId::Grayscale,
        name: "grayscale",
        aliases: &[],
        title: "Transform a RGB Image to Grayscale",
        pydantic_name: None,
        description: "Transform a RGB Image to Grayscale using the ITU-R 601-2 luma.",
        input_bands: None,
        out_bands: Some(1),
        out_dtype: OutputDtype::SameAsInput,
        out_min: None,
        out_max: None,
        dem_margin: None,
        kind: AlgorithmKind::BandMath,
        params: &[],
    },
    AlgorithmDef {
        id: AlgorithmId::Bitonal,
        name: "bitonal",
        aliases: &[],
        title: "Transform an Image to Bitonal",
        pydantic_name: None,
        description: "All values larger than 127 are set to 255 (white), all other values to 0 (black).",
        input_bands: None,
        out_bands: Some(1),
        out_dtype: OutputDtype::Fixed(SampleType::Uint8),
        out_min: None,
        out_max: None,
        dem_margin: None,
        kind: AlgorithmKind::BandMath,
        params: &[],
    },
    AlgorithmDef {
        id: AlgorithmId::Contours,
        name: "contours",
        aliases: &[],
        title: "Contours",
        pydantic_name: Some("Contours"),
        description: "Create contours from DEM dataset.",
        input_bands: Some(1),
        out_bands: Some(3),
        out_dtype: OutputDtype::Fixed(SampleType::Uint8),
        out_min: None,
        out_max: None,
        dem_margin: None,
        kind: AlgorithmKind::Dem,
        params: CONTOURS_PARAMS,
    },
];

/// Canonical list of advertised algorithm names, derived from [`ALGORITHMS`].
pub const SUPPORTED_ALGORITHMS: &[&str] = &[
    ALGORITHMS[0].name,
    ALGORITHMS[1].name,
    ALGORITHMS[2].name,
    ALGORITHMS[3].name,
    ALGORITHMS[4].name,
    ALGORITHMS[5].name,
    ALGORITHMS[6].name,
    ALGORITHMS[7].name,
    ALGORITHMS[8].name,
    ALGORITHMS[9].name,
    ALGORITHMS[10].name,
    ALGORITHMS[11].name,
    ALGORITHMS[12].name,
    ALGORITHMS[13].name,
    ALGORITHMS[14].name,
    ALGORITHMS[15].name,
    ALGORITHMS[16].name,
    ALGORITHMS[17].name,
    ALGORITHMS[18].name,
];

/// Look up an algorithm definition by canonical name or alias (ASCII case-insensitive).
pub fn lookup_algorithm(name: &str) -> Option<&'static AlgorithmDef> {
    ALGORITHMS.iter().find(|def| def.matches_name(name))
}

/// Returns true if `name` (case-insensitive) matches any supported TiTiler algorithm or alias.
pub fn is_supported_algorithm(name: &str) -> bool {
    lookup_algorithm(name).is_some()
}

/// Comma-separated canonical algorithm names for error messages.
pub fn supported_algorithms_display() -> String {
    ALGORITHMS.iter().map(|a| a.name).collect::<Vec<_>>().join(", ")
}

/// TiTiler 2.4 `AlgorithmFactory` `GET /algorithms` response.
pub fn list_algorithms_json(host: &str) -> Value {
    let origin = if host.starts_with("http://") || host.starts_with("https://") {
        host.trim_end_matches('/').to_string()
    } else {
        format!("http://{}", host.trim_end_matches('/'))
    };
    let algorithms: Vec<Value> = ALGORITHMS
        .iter()
        .map(|def| {
            let id = def.name;
            json!({
                "id": id,
                "links": [
                    {
                        "href": format!("{origin}/algorithms/{id}"),
                        "rel": "item",
                        "type": "application/json",
                        "title": format!("Definition of {id} Algorithm"),
                    }
                ]
            })
        })
        .collect();
    json!({ "algorithms": algorithms })
}

/// TiTiler 2.4 `AlgorithmFactory` `GET /algorithms/{algorithmId}` response.
pub fn get_algorithm_json(id: &str) -> Result<Value, EngineError> {
    match lookup_algorithm(id) {
        Some(def) => Ok(def.to_metadata_json()),
        None => Err(EngineError::NotFound(format!(
            "unknown algorithm '{id}' (supported: {})",
            supported_algorithms_display()
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_and_supported_names_are_in_lockstep() {
        assert_eq!(SUPPORTED_ALGORITHMS.len(), ALGORITHMS.len());
        for (name, def) in SUPPORTED_ALGORITHMS.iter().zip(ALGORITHMS.iter()) {
            assert_eq!(*name, def.name);
            assert!(is_supported_algorithm(def.name));
            assert_eq!(lookup_algorithm(def.name).map(|d| d.id), Some(def.id));
            assert!(get_algorithm_json(def.name).is_ok());
            for alias in def.aliases {
                assert!(is_supported_algorithm(alias));
                assert_eq!(lookup_algorithm(alias).map(|d| d.id), Some(def.id));
            }
        }
        // Check metadata shape for representative algorithms.
        let hs = get_algorithm_json("hillshade").unwrap();
        assert_eq!(hs["title"], "Hillshade");
        assert_eq!(hs["outputs"]["dtype"], "uint8");
        assert_eq!(hs["parameters"]["azimuth"]["maximum"], 360);
        let aspect = get_algorithm_json("aspect").unwrap();
        assert_eq!(aspect["outputs"]["dtype"], "float32");
        assert_eq!(aspect["outputs"]["max"], json!([360.0]));
        assert_eq!(aspect["parameters"]["buffer"]["default"], 3);
        let contours = get_algorithm_json("contours").unwrap();
        assert_eq!(contours["outputs"]["nbands"], 3);
        assert_eq!(contours["parameters"]["increment"]["type"], "integer");
        let ni = get_algorithm_json("normalizedIndex").unwrap();
        assert_eq!(ni["title"], "Normalized Difference Index");
        assert_eq!(get_algorithm_json("normalizedindex").unwrap(), ni);
    }
}
