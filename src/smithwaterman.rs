use std::os::raw::c_char;

extern "C" {
    pub static mut runSWOnePairBT_fp_avx2: ::std::option::Option<
        unsafe extern "C" fn(
            match_: i32,
            mismatch: i32,
            open: i32,
            extend: i32,
            seq1: *const u8,
            seq2: *const u8,
            len1: i16,
            len2: i16,
            overhangStrategy: i8,
            cigarArray: *mut ::std::os::raw::c_char,
            cigarLen: i32,
            cigarCount: *mut u32,
            offset: *mut i32,
        ) -> i32,
    >;
    pub static mut runSWOnePairBT_fp_avx512: ::std::option::Option<
        unsafe extern "C" fn(
            match_: i32,
            mismatch: i32,
            open: i32,
            extend: i32,
            seq1: *const u8,
            seq2: *const u8,
            len1: i16,
            len2: i16,
            overhangStrategy: i8,
            cigarArray: *mut ::std::os::raw::c_char,
            cigarLen: i32,
            cigarCount: *mut u32,
            offset: *mut i32,
        ) -> i32,
    >;

}

#[derive(Debug, Clone, Copy)]
pub struct Parameters {
    /// Match value.
    ///
    /// Must be >= 0.
    pub match_value: i32,

    /// Mismatch penalty.
    ///
    /// Must be <= 0.
    pub mismatch_penalty: i32,

    /// Gap open penalty.
    ///
    /// Must be <= 0.
    pub gap_open_penalty: i32,

    /// Gap extension penalty.
    ///
    /// Must be <= 0.
    pub gap_extend_penalty: i32,
}

impl Parameters {
    pub fn new(
        match_value: i32,
        mismatch_penalty: i32,
        gap_open_penalty: i32,
        gap_extend_penalty: i32,
    ) -> Self {
        Parameters {
            match_value,
            mismatch_penalty,
            gap_open_penalty,
            gap_extend_penalty,
        }
    }

    fn validate(self) {
        assert!(self.match_value >= 0);
        assert!(self.mismatch_penalty <= 0);
        assert!(self.gap_open_penalty <= 0);
        assert!(self.gap_extend_penalty <= 0);
    }
}

/// How overhangs should be treated during Smith-Waterman alignment.
#[derive(Debug, Clone, Copy)]
#[repr(i8)]
pub enum OverhangStrategy {
    /// Add softclips for the overhangs.
    SoftClip = 9,

    /// Treat the overhangs as proper insertions/deletions.
    InDel = 10,

    /// Treat the overhangs as proper insertions/deletions for leading (but not trailing) overhangs.
    ///
    /// This is useful e.g. when we want to merge dangling tails in an assembly graph: because we don't
    /// expect the dangling tail to reach the end of the reference path we are okay ignoring trailing
    /// deletions - but leading indels are still very much relevant.
    LeadingInDel = 11,

    /// Just ignore the overhangs.
    Ignore = 12,
}

pub fn align_avx2(
    ref_array: &[u8],
    alt_array: &[u8],
    parameters: Parameters,
    overhang_strategy: OverhangStrategy,
) -> Option<(Vec<u8>, usize)> {
    if !is_x86_feature_detected!("avx2") {
        return None;
    }
    parameters.validate();
    // TODO: array length validation
    let ref_len = ref_array.len();
    let alt_len = alt_array.len();
    let cigar_len = 2 * std::cmp::max(ref_len, alt_len);
    let mut cigar_array = Vec::with_capacity(cigar_len);
    let mut count = 0u32;
    let mut offset = 0i32;
    let result = unsafe {
        runSWOnePairBT_fp_avx2.unwrap()(
            parameters.match_value,
            parameters.mismatch_penalty,
            parameters.gap_open_penalty,
            parameters.gap_extend_penalty,
            ref_array.as_ptr() as _,
            alt_array.as_ptr() as _,
            ref_len as i16,
            alt_len as i16,
            overhang_strategy as i8,
            cigar_array.as_mut_ptr() as *mut c_char,
            cigar_len as i32,
            &mut count,
            &mut offset,
        )
    };
    if result == 0 {
        unsafe { cigar_array.set_len(count as usize) };
        Some((cigar_array, offset as usize))
    } else {
        None
    }
}

pub fn align_avx512(
    ref_array: &[u8],
    alt_array: &[u8],
    parameters: Parameters,
    overhang_strategy: OverhangStrategy,
) -> Option<(Vec<u8>, usize)> {
    if !is_x86_feature_detected!("avx2") {
        return None;
    }
    parameters.validate();
    // TODO: array length validation
    let ref_len = ref_array.len();
    let alt_len = alt_array.len();
    let cigar_len = 2 * std::cmp::max(ref_len, alt_len);
    let mut cigar_array = Vec::with_capacity(cigar_len);
    let mut count = 0u32;
    let mut offset = 0i32;
    let result = unsafe {
        runSWOnePairBT_fp_avx512.unwrap()(
            parameters.match_value,
            parameters.mismatch_penalty,
            parameters.gap_open_penalty,
            parameters.gap_extend_penalty,
            ref_array.as_ptr() as _,
            alt_array.as_ptr() as _,
            ref_len as i16,
            alt_len as i16,
            overhang_strategy as i8,
            cigar_array.as_mut_ptr() as *mut c_char,
            cigar_len as i32,
            &mut count,
            &mut offset,
        )
    };
    if result == 0 {
        unsafe { cigar_array.set_len(count as usize) };
        Some((cigar_array, offset as usize))
    } else {
        None
    }
}
