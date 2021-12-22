//! AVX2 and AVX-512 versions of the Smith-Waterman sequence alignment algorithm.

use std::fmt;

/// The error type returned by the alignment function.
#[derive(Debug, Clone, Copy)]
pub struct Error(&'static str);

impl fmt::Display for Error {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(self.0)
    }
}

impl std::error::Error for Error {}

/// The result type returned by the alignment function.
pub type Result<T> = std::result::Result<T, Error>;

/// limited due to the internal implementation of the native code in C
const MAX_SW_SEQUENCE_LENGTH: usize = 32 * 1024 - 1; // 2^15 - 1
/// prevents integer overflow on the diagonal of the scoring matrix
const MAXIMUM_SW_MATCH_VALUE: i32 = 64 * 1024; // 2^16

/// The parameters used by Smith-Waterman alignment.
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
    /// Create a new `Parameters`.
    ///
    /// The values are not validated by this call.  Instead, the alignment
    /// algorithms will validate them before use.
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

    /// Validate the parameter values.
    pub fn validate(self) -> Result<()> {
        if self.match_value < 0 {
            return Err(Error("match value must be >= 0"));
        }
        if self.match_value > MAXIMUM_SW_MATCH_VALUE {
            return Err(Error("match value exceeds maximum"));
        }
        if self.mismatch_penalty > 0 {
            return Err(Error("mismatch penalty must be <= 0"));
        }
        if self.gap_open_penalty > 0 {
            return Err(Error("gap open penalty must be <= 0"));
        }
        if self.gap_extend_penalty > 0 {
            return Err(Error("gap extend penalty must be <= 0"));
        }
        Ok(())
    }
}

/// How overhangs should be treated during Smith-Waterman alignment.
#[derive(Debug, Clone, Copy)]
#[repr(i8)]
pub enum OverhangStrategy {
    /// Add softclips for the overhangs.
    SoftClip = 9,

    /// Treat the overhangs as proper insertions/deletions.
    Indel = 10,

    /// Treat the overhangs as proper insertions/deletions for leading (but not trailing) overhangs.
    ///
    /// This is useful e.g. when we want to merge dangling tails in an assembly graph: because we don't
    /// expect the dangling tail to reach the end of the reference path we are okay ignoring trailing
    /// deletions - but leading indels are still very much relevant.
    LeadingIndel = 11,

    /// Just ignore the overhangs.
    Ignore = 12,
}

/// The type of an alignment function.
///
/// This is returned by the CPU feature detection functions.
pub type Align = fn(
    ref_array: &[u8],
    alt_array: &[u8],
    parameters: Parameters,
    overhang_strategy: OverhangStrategy,
) -> Result<(Vec<u8>, isize)>;

/// Return the `32x8` implementation of the alignment function if supported by the CPU features.
pub fn align_i32x8() -> Option<Align> {
    cfg_if::cfg_if! {
        if #[cfg(all(target_arch = "x86_64", feature = "c-avx"))] {
            c::align_i32x8()
        } else {
            None
        }
    }
}

/// Return the `32x16` implementation of the alignment function if supported by the CPU features.
pub fn align_i32x16() -> Option<Align> {
    cfg_if::cfg_if! {
        if #[cfg(all(target_arch = "x86_64", feature = "c-avx512"))] {
            c::align_i32x16()
        } else {
            None
        }
    }
}

/// Return the fastest alignment function that is supported by the CPU features.
pub fn align() -> Option<Align> {
    align_i32x16().or_else(align_i32x8)
}

#[cfg(all(target_arch = "x86_64", feature = "c-core"))]
mod c {
    use super::{Align, Error, OverhangStrategy, Parameters, Result};
    use std::os::raw::c_char;

    const SW_SUCCESS: i32 = 0;
    const SW_MEMORY_ALLOCATION_FAILED: i32 = 1;

    extern "C" {
        fn runSWOnePairBT_avx2(
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
        ) -> i32;
        fn runSWOnePairBT_avx512(
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
        ) -> i32;
    }

    /// Return the AVX2 alignment function if supported by the CPU features.
    pub fn align_i32x8() -> Option<Align> {
        if !is_x86_feature_detected!("avx2") {
            return None;
        }
        fn f(
            ref_array: &[u8],
            alt_array: &[u8],
            parameters: Parameters,
            overhang_strategy: OverhangStrategy,
        ) -> Result<(Vec<u8>, isize)> {
            if ref_array.len() > super::MAX_SW_SEQUENCE_LENGTH
                || alt_array.len() > super::MAX_SW_SEQUENCE_LENGTH
            {
                return Err(Error("sequences exceed maximum length"));
            }
            parameters.validate()?;
            let ref_len = ref_array.len();
            let alt_len = alt_array.len();
            let cigar_len = 2 * std::cmp::max(ref_len, alt_len);
            let mut cigar_array = Vec::with_capacity(cigar_len);
            let mut count = 0u32;
            let mut offset = 0i32;
            let result = unsafe {
                runSWOnePairBT_avx2(
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
            if result != 0 {
                return Err(Error("compute failed"));
            }
            unsafe { cigar_array.set_len(count as usize) };
            Ok((cigar_array, offset as isize))
        }
        Some(f)
    }

    /// Return the AVX-512 alignment function if supported by the CPU features.
    pub fn align_i32x16() -> Option<Align> {
        if !is_x86_feature_detected!("avx512f")
            || !is_x86_feature_detected!("avx512dq")
            || !is_x86_feature_detected!("avx512vl")
            || !is_x86_feature_detected!("avx512bw")
        {
            return None;
        }
        fn f(
            ref_array: &[u8],
            alt_array: &[u8],
            parameters: Parameters,
            overhang_strategy: OverhangStrategy,
        ) -> Result<(Vec<u8>, isize)> {
            if ref_array.len() > super::MAX_SW_SEQUENCE_LENGTH
                || alt_array.len() > super::MAX_SW_SEQUENCE_LENGTH
            {
                return Err(Error("sequences exceed maximum length"));
            }
            parameters.validate()?;
            let ref_len = ref_array.len();
            let alt_len = alt_array.len();
            let cigar_len = 2 * std::cmp::max(ref_len, alt_len);
            let mut cigar_array = Vec::with_capacity(cigar_len);
            let mut count = 0u32;
            let mut offset = 0i32;
            let result = unsafe {
                runSWOnePairBT_avx512(
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
            match result {
                SW_SUCCESS => {}
                SW_MEMORY_ALLOCATION_FAILED => return Err(Error("SW memory allocation failed")),
                _ => return Err(Error("unknown SW error")),
            }
            unsafe { cigar_array.set_len(count as usize) };
            Ok((cigar_array, offset as isize))
        }
        Some(f)
    }
}
