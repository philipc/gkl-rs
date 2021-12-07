//! AVX and AVX-512 versions of the PairHMM forward algorithm.

use std::convert::TryInto;
use std::os::raw::{c_char, c_int};

lazy_static::lazy_static! {
    static ref FORWARD32: Option<ForwardF32> = forward_f32();
    static ref FORWARD64: Option<Forward> = forward_f64();
    static ref LOG10_INITIAL_CONSTANT_32: f32 = (120.0f32).exp2().log10();
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
struct Testcase {
    rslen: c_int,
    haplen: c_int,
    q: *const c_char,
    i: *const c_char,
    d: *const c_char,
    c: *const c_char,
    hap: *const c_char,
    rs: *const c_char,
}

extern "C" {
    fn compute_avxs(arg1: *mut Testcase) -> f32;
    fn compute_avxd(arg1: *mut Testcase) -> f64;
    fn compute_avx512s(arg1: *mut Testcase) -> f32;
    fn compute_avx512d(arg1: *mut Testcase) -> f64;
    #[link_name = "\u{1}_ZN11ConvertChar15conversionTableE"]
    static mut ConvertChar_conversionTable: [u8; 255usize];
}

fn convert_char_init() {
    unsafe {
        ConvertChar_conversionTable[b'A' as usize] = 0;
        ConvertChar_conversionTable[b'C' as usize] = 1;
        ConvertChar_conversionTable[b'T' as usize] = 2;
        ConvertChar_conversionTable[b'G' as usize] = 3;
        ConvertChar_conversionTable[b'N' as usize] = 4;
    }
}

fn testcase(hap: &[u8], rs: &[u8], q: &[u8], i: &[u8], d: &[u8], c: &[u8]) -> Testcase {
    let haplen = hap.len();
    let rslen = rs.len();
    assert_eq!(rslen, q.len());
    assert_eq!(rslen, i.len());
    assert_eq!(rslen, d.len());
    assert_eq!(rslen, c.len());
    Testcase {
        rslen: rslen.try_into().unwrap(),
        haplen: haplen.try_into().unwrap(),
        q: q.as_ptr() as *const c_char,
        i: i.as_ptr() as *const c_char,
        d: d.as_ptr() as *const c_char,
        c: c.as_ptr() as *const c_char,
        hap: hap.as_ptr() as *const c_char,
        rs: rs.as_ptr() as *const c_char,
    }
}

/// The type of a PairHMM forward function that returns an `f32`.
///
/// This is returned by the CPU feature detection functions.
pub type ForwardF32 = fn(hap: &[u8], rs: &[u8], q: &[u8], i: &[u8], d: &[u8], c: &[u8]) -> f32;

/// The type of a PairHMM forward function that returns an `f64`.
///
/// This is returned by the CPU feature detection functions.
pub type Forward = fn(hap: &[u8], rs: &[u8], q: &[u8], i: &[u8], d: &[u8], c: &[u8]) -> f64;

/// Return the `f32x8` implementation of the PairHMM forward function if supported by the CPU features.
pub fn forward_f32x8() -> Option<ForwardF32> {
    if !is_x86_feature_detected!("avx") {
        return None;
    }
    convert_char_init();
    fn f(hap: &[u8], rs: &[u8], q: &[u8], i: &[u8], d: &[u8], c: &[u8]) -> f32 {
        let mut tc = testcase(hap, rs, q, i, d, c);
        unsafe { compute_avxs(&mut tc) }
    }
    Some(f)
}

/// Return the `f64x4` implementation of the PairHMM forward function if supported by the CPU features.
pub fn forward_f64x4() -> Option<Forward> {
    if !is_x86_feature_detected!("avx") {
        return None;
    }
    convert_char_init();
    fn f(hap: &[u8], rs: &[u8], q: &[u8], i: &[u8], d: &[u8], c: &[u8]) -> f64 {
        let mut tc = testcase(hap, rs, q, i, d, c);
        unsafe { compute_avxd(&mut tc) }
    }
    Some(f)
}

/// Return the `f32x16` implementation of the PairHMM forward function if supported by the CPU features.
pub fn forward_f32x16() -> Option<ForwardF32> {
    if !is_x86_feature_detected!("avx512f")
        || !is_x86_feature_detected!("avx512dq")
        || !is_x86_feature_detected!("avx512vl")
    {
        return None;
    }
    convert_char_init();
    fn f(hap: &[u8], rs: &[u8], q: &[u8], i: &[u8], d: &[u8], c: &[u8]) -> f32 {
        let mut tc = testcase(hap, rs, q, i, d, c);
        unsafe { compute_avx512s(&mut tc) }
    }
    Some(f)
}

/// Return the `f64x8` implementation of the PairHMM forward function if supported by the CPU features.
pub fn forward_f64x8() -> Option<Forward> {
    if !is_x86_feature_detected!("avx512f")
        || !is_x86_feature_detected!("avx512dq")
        || !is_x86_feature_detected!("avx512vl")
    {
        return None;
    }
    convert_char_init();
    fn f(hap: &[u8], rs: &[u8], q: &[u8], i: &[u8], d: &[u8], c: &[u8]) -> f64 {
        let mut tc = testcase(hap, rs, q, i, d, c);
        unsafe { compute_avx512d(&mut tc) }
    }
    Some(f)
}

/// Return the fastest `f32` PairHMM forward function that is supported by the CPU features.
pub fn forward_f32() -> Option<ForwardF32> {
    forward_f32x16().or_else(forward_f32x8)
}

/// Return the fastest `f64` PairHMM forward function that is supported by the CPU features.
pub fn forward_f64() -> Option<Forward> {
    forward_f64x8().or_else(forward_f64x4)
}

/// Use the fastest PairHMM forward function this is supported by the CPU features.
///
/// This function will first compute using `f32` (if a fast implementation exists),
/// and if a precision threshold is not met then it will repeat the computation using `f64`.
pub fn forward(hap: &[u8], rs: &[u8], q: &[u8], i: &[u8], d: &[u8], c: &[u8]) -> f64 {
    if let Some(f) = *FORWARD32 {
        let result = f(hap, rs, q, i, d, c);
        if result > -28. - *LOG10_INITIAL_CONSTANT_32 {
            return result as f64;
        }
    }
    // TODO: provide fallback implemenation
    FORWARD64.unwrap()(hap, rs, q, i, d, c)
}
