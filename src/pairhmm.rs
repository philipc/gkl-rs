use std::convert::TryInto;
use std::os::raw::{c_char, c_int};

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
    static mut compute_fp_avxs: Option<unsafe extern "C" fn(arg1: *mut Testcase) -> f32>;
    static mut compute_fp_avxd: Option<unsafe extern "C" fn(arg1: *mut Testcase) -> f64>;
    static mut compute_fp_avx512s: Option<unsafe extern "C" fn(arg1: *mut Testcase) -> f32>;
    static mut compute_fp_avx512d: Option<unsafe extern "C" fn(arg1: *mut Testcase) -> f64>;
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

pub fn forward_avxs(hap: &[u8], rs: &[u8], q: &[u8], i: &[u8], d: &[u8], c: &[u8]) -> Option<f32> {
    if !is_x86_feature_detected!("avx") {
        return None;
    }
    convert_char_init();
    let mut tc = testcase(hap, rs, q, i, d, c);
    Some(unsafe { compute_fp_avxs.unwrap()(&mut tc) })
}

pub fn forward_avxd(hap: &[u8], rs: &[u8], q: &[u8], i: &[u8], d: &[u8], c: &[u8]) -> Option<f64> {
    if !is_x86_feature_detected!("avx") {
        return None;
    }
    convert_char_init();
    let mut tc = testcase(hap, rs, q, i, d, c);
    Some(unsafe { compute_fp_avxd.unwrap()(&mut tc) })
}

pub fn forward_avx512s(
    hap: &[u8],
    rs: &[u8],
    q: &[u8],
    i: &[u8],
    d: &[u8],
    c: &[u8],
) -> Option<f32> {
    if !is_x86_feature_detected!("avx512f")
        || !is_x86_feature_detected!("avx512dq")
        || !is_x86_feature_detected!("avx512vl")
    {
        return None;
    }
    convert_char_init();
    let mut tc = testcase(hap, rs, q, i, d, c);
    Some(unsafe { compute_fp_avx512s.unwrap()(&mut tc) })
}

pub fn forward_avx512d(
    hap: &[u8],
    rs: &[u8],
    q: &[u8],
    i: &[u8],
    d: &[u8],
    c: &[u8],
) -> Option<f64> {
    if !is_x86_feature_detected!("avx512f")
        || !is_x86_feature_detected!("avx512dq")
        || !is_x86_feature_detected!("avx512vl")
    {
        return None;
    }
    convert_char_init();
    let mut tc = testcase(hap, rs, q, i, d, c);
    Some(unsafe { compute_fp_avx512d.unwrap()(&mut tc) })
}
