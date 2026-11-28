//! The type an array of data is written in, as TiTiler writes it: the raster's own type,
//! the type that holds every band when several are stacked, and what numexpr makes of band
//! math. The renderers work in single precision; an array they write is rewritten in this
//! type before it is answered.

use cogzilla_kernel::ResampleMode;
use cogzilla_raster::SampleType;

use super::ParsedQuery;

/// The type that holds the numbers of both, as numpy promotes them when it stacks bands.
pub(crate) fn wider_type(a: SampleType, b: SampleType) -> SampleType {
    use SampleType as T;
    if a == b {
        return a;
    }
    let float = |t| matches!(t, T::Float16 | T::Float32 | T::Float64);
    let signed = |t| matches!(t, T::Int8 | T::Int16 | T::Int32 | T::Int64);
    let bits = |t| match t {
        T::Bool | T::Uint1 | T::Uint4 | T::Uint8 | T::Int8 => 8u32,
        T::Uint12 | T::Uint16 | T::Int16 | T::Float16 => 16,
        T::Uint32 | T::Int32 | T::Float32 => 32,
        T::Uint64 | T::Int64 | T::Float64 => 64,
    };
    if float(a) || float(b) {
        // float32 holds whole numbers of 16 bits and no more (and Float16 promotes to Float32).
        let narrow = |t| if float(t) { bits(t) <= 32 } else { bits(t) <= 16 };
        return if narrow(a) && narrow(b) { T::Float32 } else { T::Float64 };
    }
    let sized = |kinds: [SampleType; 4], width: u32| kinds[match width { 8 => 0, 16 => 1, 32 => 2, _ => 3 }];
    match (signed(a), signed(b)) {
        (false, false) => sized([T::Uint8, T::Uint16, T::Uint32, T::Uint64], bits(a).max(bits(b))),
        (true, true) => sized([T::Int8, T::Int16, T::Int32, T::Int64], bits(a).max(bits(b))),
        // A signed type one size up from the unsigned one, or the signed one's own if larger.
        _ => {
            let (own, unsigned) = if signed(a) { (bits(a), bits(b)) } else { (bits(b), bits(a)) };
            if own == 64 && unsigned < 64 {
                T::Int64
            } else {
                match (unsigned * 2).max(own) {
                    16 => T::Int16,
                    32 => T::Int32,
                    64 => T::Int64,
                    _ => T::Float64,
                }
            }
        }
    }
}

/// Split a top-level `where(cond, a, b)` call into its `(then_branch, else_branch)` substrings.
fn where_branches(expr: &str) -> Option<(&str, &str)> {
    let trimmed = expr.trim();
    let inner = trimmed
        .strip_prefix("where")
        .or_else(|| trimmed.strip_prefix("WHERE"))?
        .trim_start()
        .strip_prefix('(')?
        .strip_suffix(')')?;
    let mut depth = 0usize;
    let mut commas = [0usize; 2];
    let mut found = 0usize;
    for (idx, b) in inner.bytes().enumerate() {
        match b {
            b'(' => depth += 1,
            b')' => {
                // Closing paren before the final char means `where(...)` wasn't the outer expression.
                if depth == 0 {
                    return None;
                }
                depth -= 1;
            }
            b',' if depth == 0 => {
                if found < 2 {
                    commas[found] = idx;
                }
                found += 1;
            }
            _ => {}
        }
    }
    if depth == 0 && found == 2 {
        Some((inner[commas[0] + 1..commas[1]].trim(), inner[commas[1] + 1..].trim()))
    } else {
        None
    }
}

fn is_int_literal(expr: &str) -> bool {
    let s = expr.trim();
    let digits = s.strip_prefix(['+', '-']).unwrap_or(s);
    !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit()) && s.parse::<i32>().is_ok()
}

fn single_expr_type(expr: &str, read: SampleType) -> SampleType {
    let trimmed = expr.trim();
    if let Some((a, b)) = where_branches(trimmed) {
        let a_lit = is_int_literal(a);
        let b_lit = is_int_literal(b);
        if a_lit && b_lit {
            return SampleType::Int32;
        }
        let ta = single_expr_type(a, read);
        let tb = single_expr_type(b, read);
        // In numexpr, combining an integer literal with float32 keeps float32 (unlike numpy wider_type).
        return match (ta, tb) {
            (SampleType::Float32, SampleType::Int32) if b_lit => SampleType::Float32,
            (SampleType::Int32, SampleType::Float32) if a_lit => SampleType::Float32,
            _ => wider_type(ta, tb),
        };
    }
    if is_int_literal(trimmed) {
        return SampleType::Int32;
    }
    let whole = |e: &str| {
        let mut rest = e;
        while let Some(at) = rest.find('b') {
            let digits = rest[at + 1..].bytes().take_while(u8::is_ascii_digit).count();
            if digits == 0 {
                return false;
            }
            rest = &rest[at + 1 + digits..];
        }
        !e.contains("**")
            && !e
                .bytes()
                .any(|c| c == b'/' || c == b'.' || c == b'^' || (c.is_ascii_alphabetic() && c != b'b'))
    };
    // float32 stays float32 through operations and most functions; a literal with a
    // fraction/exponent promotes to float64 EXCEPT when it is the constant exponent of `**`
    // (numexpr compiles `pow(f32, float_const)` in single precision), and double-only numexpr
    // functions (`minimum`, `maximum`) promote to float64.
    let has_non_pow_float_literal = |e: &str| {
        let bytes = e.as_bytes();
        let mut i = 0usize;
        while i < bytes.len() {
            // Skip numeric literal immediately following `**` (e.g. `b1**0.5`, `b2**-1.5`).
            if bytes[i] == b'*' && i + 1 < bytes.len() && bytes[i + 1] == b'*' {
                i += 2;
                while i < bytes.len() && bytes[i].is_ascii_whitespace() {
                    i += 1;
                }
                if i < bytes.len() && matches!(bytes[i], b'+' | b'-') {
                    i += 1;
                }
                while i < bytes.len() && (bytes[i].is_ascii_digit() || bytes[i] == b'.') {
                    i += 1;
                }
                if i < bytes.len() && matches!(bytes[i], b'e' | b'E') {
                    i += 1;
                    if i < bytes.len() && matches!(bytes[i], b'+' | b'-') {
                        i += 1;
                    }
                    while i < bytes.len() && bytes[i].is_ascii_digit() {
                        i += 1;
                    }
                }
                continue;
            }
            if bytes[i] == b'.' {
                return true;
            }
            if bytes[i].is_ascii_digit()
                && i + 1 < bytes.len()
                && matches!(bytes[i + 1], b'e' | b'E')
            {
                return true;
            }
            i += 1;
        }
        false
    };
    let plain = |e: &str| {
        !has_non_pow_float_literal(e) && !e.contains("minimum(") && !e.contains("maximum(")
    };
    if read == SampleType::Float32 && plain(trimmed) {
        SampleType::Float32
    } else if !whole(trimmed) {
        SampleType::Float64
    } else {
        match read {
            SampleType::Uint1
            | SampleType::Uint4
            | SampleType::Uint8
            | SampleType::Int8
            | SampleType::Uint12
            | SampleType::Uint16
            | SampleType::Int16 => SampleType::Int32,
            _ => SampleType::Float64,
        }
    }
}

/// The type band math comes out in, as numexpr has it in TiTiler, from the type `read`
/// of the bands it is worked on: whole numbers of up to 16 bits stay whole, as 32-bit
/// ones, through sums, differences and products, and 32-bit floats stay so; a division, a
/// function or a fraction makes doubles; `where(cond, a, b)` takes its type from `(a, b)`.
pub(crate) fn band_math_type<'a>(expressions: impl Iterator<Item = &'a String>, read: SampleType) -> SampleType {
    let mut out: Option<SampleType> = None;
    for e in expressions {
        let t = single_expr_type(e, read);
        out = Some(match out {
            Some(prev) => wider_type(prev, t),
            None => t,
        });
    }
    out.unwrap_or(SampleType::Float64)
}

/// Whether a request turns values into a picture's bytes: a rescale, a colour map, a colour
/// formula. TiTiler's array of such a request is those bytes.
pub(super) fn shown(pq: &ParsedQuery) -> bool {
    !pq.rescale.is_empty()
        || pq.colormap_explicit
        || pq.colormap_name_set
        || pq.custom_colormap_json.is_some()
        || pq.color_formula.is_some()
}

/// The type `algorithm_params={"dtype": ...}` names for `cast`, `uint8` when none is named;
/// `None` for a name that is not a type supported by `cast`.
pub(crate) fn cast_type(dtype: Option<&str>) -> Option<SampleType> {
    let st = SampleType::from_numpy_name(dtype.unwrap_or("uint8"))?;
    match st {
        SampleType::Bool | SampleType::Uint1 | SampleType::Uint4 | SampleType::Uint12 | SampleType::Uint64 | SampleType::Int64 | SampleType::Float16 => None,
        _ => Some(st),
    }
}

/// A number put in an integer `kind` as numpy's `astype` puts it: truncated toward zero
/// and, in types of up to 16 bits, wrapped around the type's range (300 is 44 and -1 is 255
/// in uint8), by way of a 32-bit integer as numpy converts; other types take it as it is.
pub(crate) fn astype(kind: SampleType, v: f64) -> f64 {
    use SampleType as T;
    // `as i32` truncates and saturates, and makes nothing of NaN, as numpy's conversion does.
    let whole = v as i32;
    match kind {
        T::Uint8 => f64::from(whole as u8),
        T::Int8 => f64::from(whole as i8),
        T::Uint16 => f64::from(whole as u16),
        T::Int16 => f64::from(whole as i16),
        _ => v,
    }
}

/// Round to the nearest and hold to `kind`'s range, as GDAL puts a resampled value into an
/// integer type (where [`astype`] would wrap a cubic kernel's undershoot around).
fn round_into(kind: SampleType) -> fn(f64) -> f64 {
    use SampleType as T;
    match kind {
        T::Bool | T::Uint1 => |v: f64| v.round().clamp(0.0, 1.0),
        T::Uint4 => |v: f64| v.round().clamp(0.0, 15.0),
        T::Uint8 => |v: f64| v.round().clamp(0.0, 255.0),
        T::Int8 => |v: f64| v.round().clamp(-128.0, 127.0),
        T::Uint12 => |v: f64| v.round().clamp(0.0, 4095.0),
        T::Uint16 => |v: f64| v.round().clamp(0.0, 65_535.0),
        T::Int16 => |v: f64| v.round().clamp(-32_768.0, 32_767.0),
        T::Uint32 => |v: f64| v.round().clamp(0.0, 4_294_967_295.0),
        T::Int32 => |v: f64| v.round().clamp(-2_147_483_648.0, 2_147_483_647.0),
        _ => f64::round,
    }
}

/// What decides the type of an array across `/cog`, `/stac`, `/mosaicjson`, and `/searches`,
/// read from the request.
#[derive(Debug, Clone, PartialEq)]
pub(super) enum ArrayRule {
    /// The bands as they were read, in the type that holds them all; float32 once unscaled.
    /// `resampled`: read with a resampling other than nearest, so a value may fall between
    /// whole numbers.
    Data { unscale: bool, resampled: bool },
    /// Band math over them.
    Math { unscale: bool },
    /// An algorithm's output, typed as TiTiler's algorithm types it: its name, lowercase,
    /// and for `cast` the type asked (`None` when what was asked is not a type).
    Algorithm { name: String, cast: Option<SampleType>, unscale: bool, resampled: bool },
    /// A `/searches` pixel-selection mosaic fold (`stdev` → float64, `count` → uint8,
    /// named spectral index → float32, else band math or widened read type).
    Selection {
        unscale: bool,
        selection: super::pixel_selection::Selection,
        index: bool,
    },
}

impl ArrayRule {
    /// The rule for a request, or none when what was asked turns the values into a
    /// picture's bytes (a rescale, a colour map, a colour formula), or does band math on
    /// single assets: those arrays are written as they are.
    pub(super) fn of(pq: &ParsedQuery) -> Option<Self> {
        if shown(pq) || !pq.asset_exprs.is_empty() {
            return None;
        }
        let resampled = pq.resample != ResampleMode::Nearest || pq.reproject != ResampleMode::Nearest;
        if let Some(name) = &pq.algorithm {
            let cast = cast_type(pq.cast_dtype.as_deref());
            return Some(Self::Algorithm { name: name.clone(), cast, unscale: pq.unscale, resampled });
        }
        Some(match pq.expr_raw {
            Some(_) => Self::Math { unscale: pq.unscale },
            None => Self::Data { unscale: pq.unscale, resampled },
        })
    }

    /// Whether the rule reads the bands with their scale and offset applied.
    pub(super) fn unscales(&self) -> bool {
        match self {
            Self::Data { unscale, .. }
            | Self::Math { unscale }
            | Self::Algorithm { unscale, .. }
            | Self::Selection { unscale, .. } => *unscale,
        }
    }

    /// The type for bands read as `reads`, with `expression` the band math as worked on
    /// (`b1`, `b2`, ... after any asset names are rewritten), several separated by `;`.
    pub(super) fn kind(&self, reads: &[SampleType], expression: Option<&str>) -> Option<SampleType> {
        let parts: Option<Vec<String>> =
            expression.map(|e| e.split(';').map(str::to_string).collect());
        self.kind_with_exprs(reads, parts.as_ref().map(|p| p.iter()))
    }

    /// Like [`Self::kind`], taking pre-split expression strings directly (zero extra allocation
    /// for callers that already hold expression slices or iterators).
    pub(super) fn kind_with_exprs<'a>(
        &self,
        reads: &[SampleType],
        expressions: Option<impl Iterator<Item = &'a String>>,
    ) -> Option<SampleType> {
        let first = *reads.first()?;
        let read = if self.unscales() {
            SampleType::Float32
        } else {
            reads[1..].iter().fold(first, |a, b| wider_type(a, *b))
        };
        match self {
            Self::Data { .. } => Some(read),
            Self::Math { .. } => Some(band_math_type(expressions?, read)),
            Self::Algorithm { name, cast, .. } => {
                // Over band math, the algorithm works on the expression's outputs, in their type.
                let read = match expressions {
                    Some(exprs) => band_math_type(exprs, read),
                    None => read,
                };
                algorithm_type(name, *cast, read)
            }
            Self::Selection { selection, index, .. } => {
                use super::pixel_selection::Selection;
                if matches!(selection, Selection::Stdev) {
                    Some(SampleType::Float64)
                } else if matches!(selection, Selection::Count) {
                    Some(SampleType::Uint8)
                } else if *index {
                    Some(SampleType::Float32)
                } else {
                    match expressions {
                        Some(exprs) => Some(band_math_type(exprs, read)),
                        None => Some(read),
                    }
                }
            }
        }
    }

    /// Staged typing for per-asset band math (`own` per asset in `reads`), stacked via numpy
    /// promotion, optionally followed by a request-level `expression`, and finally any `Algorithm`.
    pub(super) fn staged_kind(
        &self,
        reads: &[SampleType],
        own: &[Option<String>],
        request: &str,
    ) -> Option<SampleType> {
        if own.len() != reads.len() {
            return None;
        }
        let unscale = self.unscales();
        let stacked = own
            .iter()
            .zip(reads)
            .map(|(own, read)| match own {
                Some(expr) => Self::Math { unscale }.kind(&[*read], Some(expr)),
                None => Self::Data { unscale, resampled: false }.kind(&[*read], None),
            })
            .reduce(|a, b| Some(wider_type(a?, b?)))
            .flatten()?;
        let after_math = if request.is_empty() {
            stacked
        } else {
            let parts: Vec<String> = request.split(';').map(str::to_string).collect();
            let whole = band_math_type(parts.iter(), SampleType::Uint16) == SampleType::Int32;
            if stacked == SampleType::Int32 && whole {
                SampleType::Int32
            } else {
                band_math_type(parts.iter(), stacked)
            }
        };
        if let Self::Algorithm { name, cast, .. } = self {
            algorithm_type(name, *cast, after_math)
        } else {
            Some(after_math)
        }
    }

    /// How each number is rounded before it is put in `kind`: TiTiler's `cast` truncates
    /// (`astype`), `ceil` and `floor` round up and down first, and all three wrap around the
    /// range of a small integer type ([`astype`]). The bands as read with a resampling, and
    /// their `min` and `max`, round to the nearest: GDAL puts a resampled value into an
    /// integer band's type that way (a value read with nearest is whole already, and is put
    /// as it is). `None` for anything else.
    pub(super) fn rounding(&self, kind: SampleType) -> Option<fn(f64) -> f64> {
        kind.storable_range()?;
        match self {
            Self::Data { unscale: false, resampled: true } => Some(round_into(kind)),
            Self::Algorithm { name, unscale: false, resampled: true, .. } if matches!(name.as_str(), "min" | "max") => {
                Some(round_into(kind))
            }
            Self::Algorithm { name, .. } => match name.as_str() {
                "cast" => Some(f64::trunc),
                "ceil" => Some(f64::ceil),
                "floor" => Some(f64::floor),
                _ => None,
            },
            _ => None,
        }
    }
}

/// The type TiTiler's algorithm `name` writes its output in, from the type `read` of the
/// bands it works on (derived from [`crate::algorithms::ALGORITHMS`]).
fn algorithm_type(name: &str, cast: Option<SampleType>, read: SampleType) -> Option<SampleType> {
    crate::algorithms::lookup_algorithm(name)?.out_dtype.resolve(cast, read)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stacked_bands_take_the_type_that_holds_them_all() {
        use SampleType as T;
        for (a, b, both) in [
            (T::Uint8, T::Uint16, T::Uint16), (T::Uint16, T::Int16, T::Int32), (T::Uint8, T::Int8, T::Int16),
            (T::Uint8, T::Int16, T::Int16), (T::Uint8, T::Int32, T::Int32), (T::Int8, T::Int16, T::Int16),
            (T::Uint16, T::Float32, T::Float32), (T::Uint32, T::Float32, T::Float64), (T::Float32, T::Float64, T::Float64),
            (T::Uint32, T::Int32, T::Int64), (T::Uint64, T::Int32, T::Float64), (T::Uint8, T::Uint8, T::Uint8), (T::Uint16, T::Uint32, T::Uint32),
        ] {
            assert_eq!(wider_type(a, b), both, "{a:?} with {b:?}");
            assert_eq!(wider_type(b, a), both, "{b:?} with {a:?}");
        }
    }

    #[test]
    fn band_math_keeps_whole_numbers_whole_until_it_divides() {
        use SampleType as T;
        let kind = |e: &str, read| band_math_type(std::iter::once(&e.to_string()), read);
        assert_eq!(kind("b1*2", T::Uint16), T::Int32);
        assert_eq!(kind("b1+b2", T::Int16), T::Int32);
        assert_eq!(kind("b1/2", T::Uint16), T::Float64);
        assert_eq!(kind("b1*0.5", T::Uint8), T::Float64);
        assert_eq!(kind("sqrt(b1)", T::Uint8), T::Float64);
        assert_eq!(kind("b1/2", T::Float32), T::Float32);
        assert_eq!(kind("b1*0.5", T::Float32), T::Float64);
        assert_eq!(kind("minimum(b1,b2)+maximum(b2,b3)", T::Float32), T::Float64);
        assert_eq!(kind("b1*2", T::Int32), T::Float64);
        assert_eq!(kind("b1**-1", T::Uint16), T::Float64);
        assert_eq!(kind("b1^-1", T::Uint8), T::Float64);
        assert_eq!(kind("b1**3", T::Uint16), T::Float64);
        assert_eq!(kind("b1**-1", T::Float32), T::Float32);
        assert_eq!(kind("(b1**0.5)+(b2**1.5)", T::Float32), T::Float32);
        assert_eq!(kind("(b1**0.5)+0.1", T::Float32), T::Float64);
        assert_eq!(kind("b1**0.5", T::Uint8), T::Float64);
        // where(cond, a, b) takes its type from (a, b), not float literals in cond
        for read in [T::Uint8, T::Uint16, T::Int16, T::Float32, T::Float64] {
            assert_eq!(kind("where((b1>0.1)&(b2<0.3),1,0)", read), T::Int32, "{read:?}");
        }
        assert_eq!(kind("where((b1>0.25)|(b3>0.25),b1,b2)", T::Float32), T::Float32);
        assert_eq!(kind("where((b1>0.25)|(b3>0.25),b1,b2)", T::Uint8), T::Int32);
        assert_eq!(kind("where(b1>0.5,b1,0)", T::Float32), T::Float32);
        assert_eq!(kind("where(b1>0.5,b1,0.5)", T::Float32), T::Float64);
        assert_eq!(kind("where(b1>0.5,1,0)+0.5", T::Float32), T::Float64);
    }

    #[test]
    fn a_request_says_which_rule_its_array_follows() {
        let rule = |q: &str| ArrayRule::of(&super::super::parse_query(Some(q)));
        assert_eq!(rule("url=a.tif"), Some(ArrayRule::Data { unscale: false, resampled: false }));
        assert_eq!(rule("url=a.tif&unscale=true"), Some(ArrayRule::Data { unscale: true, resampled: false }));
        assert_eq!(rule("url=a.tif&expression=b1*2"), Some(ArrayRule::Math { unscale: false }));
        for shown in ["rescale=0,100", "colormap_name=viridis", "color_formula=gamma%20r%202", "algorithm=hillshade&rescale=0,1", "assets=a%7Cexpression=b1*2"] {
            assert_eq!(rule(&format!("url=a.tif&{shown}")), None, "{shown}");
        }
        use SampleType as T;
        assert_eq!(ArrayRule::Data { unscale: false, resampled: false }.kind(&[T::Uint16], None), Some(T::Uint16));
        assert_eq!(ArrayRule::Data { unscale: false, resampled: false }.kind(&[T::Uint8, T::Uint16, T::Int16], None), Some(T::Int32));
        assert_eq!(ArrayRule::Data { unscale: true, resampled: false }.kind(&[T::Int16], None), Some(T::Float32));
        assert_eq!(ArrayRule::Math { unscale: false }.kind(&[T::Uint16, T::Uint16], Some("b1+b2;b1*2")), Some(T::Int32));
        assert_eq!(ArrayRule::Math { unscale: false }.kind(&[T::Uint16], Some("(b2-b1)/(b2+b1)")), Some(T::Float64));
        assert_eq!(ArrayRule::Math { unscale: false }.kind(&[T::Uint16], None), None);
        assert_eq!(ArrayRule::Data { unscale: false, resampled: false }.kind(&[], None), None);
    }

    #[test]
    fn an_algorithm_writes_its_array_in_the_type_titiler_gives_it() {
        use SampleType as T;
        let rule = |q: &str| ArrayRule::of(&super::super::parse_query(Some(q))).unwrap();
        let kind = |alg: &str, read| rule(&format!("url=a.tif&algorithm={alg}")).kind(&[read], None);
        for (alg, read, out) in [
            ("cast", T::Float32, T::Uint8), ("ceil", T::Int16, T::Uint8), ("floor", T::Float64, T::Uint8),
            ("bitonal", T::Uint16, T::Uint8), ("hillshade", T::Float32, T::Uint8), ("contours", T::Int16, T::Uint8),
            ("terrarium", T::Float32, T::Uint8), ("terrainrgb", T::Int16, T::Uint8),
            ("min", T::Int16, T::Int16), ("max", T::Uint16, T::Uint16), ("grayscale", T::Uint8, T::Uint8),
            ("sum", T::Uint16, T::Uint64), ("sum", T::Int16, T::Int64), ("sum", T::Float32, T::Float32),
            ("mean", T::Float32, T::Float64), ("std", T::Uint8, T::Float64), ("var", T::Float32, T::Float64),
            ("median", T::Float32, T::Float32), ("median", T::Uint16, T::Float64),
            ("normalizedIndex", T::Uint16, T::Float32), ("slope", T::Int16, T::Float32),
        ] {
            assert_eq!(kind(alg, read), Some(out), "{alg} over {read:?}");
        }
        assert_eq!(kind("aspect", T::Float32), None);
        let stacked = rule("url=a.tif&algorithm=max");
        assert_eq!(stacked.kind(&[T::Uint8, T::Int16], None), Some(T::Int16));
        assert_eq!(rule("url=a.tif&algorithm=max&unscale=true").kind(&[T::Int16], None), Some(T::Float32));
        let cast_to = |dtype: &str| rule(&format!("url=a.tif&algorithm=cast&algorithm_params=%7B%22dtype%22:%22{dtype}%22%7D")).kind(&[T::Float32], None);
        assert_eq!(cast_to("int16"), Some(T::Int16));
        assert_eq!(cast_to("float32"), Some(T::Float32));
        assert_eq!(cast_to("complex"), None);
    }

    #[test]
    fn cast_ceil_and_floor_round_then_wrap_as_numpy_astype_does() {
        use SampleType as T;
        let rule = |alg: &str| ArrayRule::of(&super::super::parse_query(Some(&format!("url=a.tif&algorithm={alg}")))).unwrap();
        let put = |alg: &str, v: f64| astype(T::Uint8, rule(alg).rounding(T::Uint8).unwrap()(v));
        assert_eq!(put("cast", 300.0), 44.0);
        assert_eq!(put("cast", -1.0), 255.0);
        assert_eq!(put("cast", -300.7), 212.0);
        assert_eq!(put("cast", 255.9), 255.0);
        assert_eq!(put("cast", 70_000.5), 112.0);
        assert_eq!(put("cast", 1e10), 255.0);
        assert_eq!(put("ceil", 255.2), 0.0);
        assert_eq!(put("floor", -0.5), 255.0);
        assert_eq!(astype(T::Int16, 40_000.0), -25_536.0);
        assert_eq!(astype(T::Int8, 200.0), -56.0);
        // A read, and its min and max, round as GDAL rounds a resampled value into the type.
        assert!(rule("min").rounding(T::Uint8).is_none());
        let resampled = ArrayRule::of(&super::super::parse_query(Some("url=a.tif&algorithm=min&resampling=bilinear"))).unwrap();
        assert_eq!(resampled.rounding(T::Uint8).unwrap()(2.5), 3.0);
        assert_eq!(ArrayRule::Data { unscale: false, resampled: true }.rounding(T::Uint16).unwrap()(17836.6), 17837.0);
        assert!(ArrayRule::Data { unscale: false, resampled: true }.rounding(T::Float32).is_none());
        assert!(ArrayRule::Data { unscale: false, resampled: false }.rounding(T::Uint16).is_none());
        // Held to the type's range, not wrapped: a cubic undershoot beside nodata 0 is 0.
        assert_eq!(ArrayRule::Data { unscale: false, resampled: true }.rounding(T::Uint16).unwrap()(-4.6), 0.0);
        assert_eq!(ArrayRule::Data { unscale: false, resampled: true }.rounding(T::Uint8).unwrap()(255.7), 255.0);
        assert!(rule("mean").rounding(T::Uint8).is_none());
        assert!(rule("cast").rounding(T::Float32).is_none());
    }

    #[test]
    fn staged_kind_handles_mixed_types_unscale_and_algorithms() {
        use SampleType as T;
        let math = ArrayRule::Math { unscale: false };
        // Uint16 whole math (Int32) stacked with plain Float32 -> Float64.
        assert_eq!(
            math.staged_kind(&[T::Uint16, T::Float32], &[Some("b1*2".into()), None], ""),
            Some(T::Float64)
        );
        // Uint16 whole math (Int32) + plain Uint16 -> Int32, preserved under whole request math.
        assert_eq!(
            math.staged_kind(&[T::Uint16, T::Uint16], &[Some("b1*2".into()), None], "b1+b2"),
            Some(T::Int32)
        );
        // Fixed-dtype algorithm applies AFTER per-asset & request math.
        let floor = ArrayRule::Algorithm { name: "floor".into(), cast: None, unscale: false, resampled: false };
        assert_eq!(
            floor.staged_kind(&[T::Uint16, T::Uint16], &[Some("b1*2".into()), None], ""),
            Some(T::Uint8)
        );
        assert_eq!(
            floor.staged_kind(&[T::Uint16], &[Some("b1*2".into())], "b1+1"),
            Some(T::Uint8)
        );
        // Input-dependent algorithm (`max`) sees the Float64 from `b1/2`.
        let max_alg = ArrayRule::Algorithm { name: "max".into(), cast: None, unscale: false, resampled: false };
        assert_eq!(
            max_alg.staged_kind(&[T::Uint16], &[Some("b1/2".into())], ""),
            Some(T::Float64)
        );
        // Unscaled math reads bands as Float32.
        let math_unscale = ArrayRule::Math { unscale: true };
        assert_eq!(
            math_unscale.staged_kind(&[T::Uint16, T::Uint16], &[Some("b1*2".into()), None], ""),
            Some(T::Float32)
        );
    }

    #[test]
    fn round_into_matches_storable_range_across_all_variants() {
        use SampleType as T;
        let all = [
            T::Bool,
            T::Uint1,
            T::Uint4,
            T::Uint8,
            T::Int8,
            T::Uint12,
            T::Uint16,
            T::Int16,
            T::Uint32,
            T::Int32,
            T::Uint64,
            T::Int64,
            T::Float16,
            T::Float32,
            T::Float64,
        ];
        for kind in all {
            let f = round_into(kind);
            match (kind, kind.storable_range()) {
                (T::Uint64 | T::Int64, Some(_)) | (_, None) => {
                    assert_eq!(f(12.6), 13.0, "{kind:?}: round_into should round");
                }
                (_, Some((lo, hi))) => {
                    assert_eq!(f(lo - 100.0), lo, "{kind:?}: lower clamp must match storable_range");
                    assert_eq!(f(hi + 100.0), hi, "{kind:?}: upper clamp must match storable_range");
                    assert_eq!(f(lo + 0.6), lo + 1.0, "{kind:?}: midpoint rounds to nearest");
                }
            }
        }
    }

    #[test]
    fn selection_and_math_dtype_matrix_is_uniform_across_families() {
        use super::super::pixel_selection::Selection;
        use SampleType as T;

        // Every family agrees on int32 / float32 / float64 band-math expressions.
        for (expr, unscale, expected) in [
            ("b1+b2", false, T::Int32),
            ("b1+b2", true, T::Float32),
            ("b1/b2", false, T::Float64),
            ("b1/b2", true, T::Float32),
            ("b1*0.5", false, T::Float64),
            ("b1*0.5", true, T::Float64),
        ] {
            assert_eq!(
                ArrayRule::Math { unscale }.kind(&[T::Uint16], Some(expr)),
                Some(expected),
                "Math ({expr}, unscale={unscale})"
            );
            assert_eq!(
                ArrayRule::Selection {
                    unscale,
                    selection: Selection::First,
                    index: false,
                }
                .kind(&[T::Uint16], Some(expr)),
                Some(expected),
                "Selection::First ({expr}, unscale={unscale})"
            );
        }

        // Selection precedence: Stdev -> Float64, Count -> Uint8, named index -> Float32.
        assert_eq!(
            ArrayRule::Selection { unscale: false, selection: Selection::Stdev, index: false }
                .kind(&[T::Uint16], Some("b1+b2")),
            Some(T::Float64)
        );
        assert_eq!(
            ArrayRule::Selection { unscale: true, selection: Selection::Count, index: true }
                .kind(&[T::Float32], Some("b1/b2")),
            Some(T::Uint8)
        );
        assert_eq!(
            ArrayRule::Selection { unscale: false, selection: Selection::First, index: true }
                .kind(&[T::Uint16], None),
            Some(T::Float32)
        );
        assert_eq!(
            ArrayRule::Selection { unscale: false, selection: Selection::First, index: false }
                .kind(&[T::Uint16], None),
            Some(T::Uint16)
        );
        assert_eq!(
            ArrayRule::Selection { unscale: true, selection: Selection::First, index: false }
                .kind(&[T::Uint16], None),
            Some(T::Float32)
        );
    }
}
