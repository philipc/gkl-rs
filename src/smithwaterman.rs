//! AVX2 and AVX-512 versions of the Smith-Waterman sequence alignment algorithm.

use crate::vector::Int32Vector;
use std::io::Write;
use std::{cmp, fmt};

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
const MATCH: i16 = 0;
const INSERT: i16 = 1;
const DELETE: i16 = 2;
const INSERT_EXT: i16 = 4;
const DELETE_EXT: i16 = 8;
const SOFTCLIP: i16 = 9;
/*
const INDEL: i16 = 10;
const LEADING_INDEL: i16 = 11;
const IGNORE: i16 = 12;
*/

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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

/// Return the scalar `i32` implementation of the alignment function.
///
/// The compiler may autovectorize this function.
pub fn align_i32x1() -> Align {
    fn f(
        ref_array: &[u8],
        alt_array: &[u8],
        parameters: Parameters,
        overhang_strategy: OverhangStrategy,
    ) -> Result<(Vec<u8>, isize)> {
        let v = crate::vector::I32x1;
        compute_and_backtrack(v, ref_array, alt_array, parameters, overhang_strategy)
    }
    f
}

/// Return the `32x8` implementation of the alignment function if supported by the CPU features.
pub fn align_i32x8() -> Option<Align> {
    cfg_if::cfg_if! {
        if #[cfg(all(target_arch = "x86_64", feature = "c-avx"))] {
            c::align_i32x8()
        } else if #[cfg(target_arch = "x86_64")] {
            x86_64_avx::align_i32x8()
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
        } else if #[cfg(all(target_arch = "x86_64", feature = "nightly"))] {
            x86_64_avx512::align_i32x16()
        } else {
            None
        }
    }
}

/// Return the fastest alignment function that is supported by the CPU features.
pub fn align() -> Option<Align> {
    align_i32x16().or_else(align_i32x8)
}

#[inline]
fn compute_and_backtrack<V: Int32Vector>(
    v: V,
    ref_array: &[u8],
    alt_array: &[u8],
    parameters: Parameters,
    overhang_strategy: OverhangStrategy,
) -> Result<(Vec<u8>, isize)> {
    if ref_array.len() > MAX_SW_SEQUENCE_LENGTH || alt_array.len() > MAX_SW_SEQUENCE_LENGTH {
        return Err(Error("sequences exceed maximum length"));
    }

    // Number of antidiagonals.
    // TODO: avoid 1-based index?
    let antidiag_num = ref_array.len() + alt_array.len() + 1;
    // Maximum length of a backtrack antidiagonal.
    let max_antidiag_len = cmp::min(ref_array.len(), alt_array.len());
    // Number of i16 in a backtrack antidiagonal, rounded to vector size.
    let backtrack_stride = (max_antidiag_len + 2 * V::LANES - 1) & !(2 * V::LANES - 1);
    // Number of i16 in the backtrack array.
    let backtrack_len = backtrack_stride * antidiag_num;
    // The backtrack array, which contains `i16` but must have the alignment of `V::Vec`.
    // This array will never be completely initialized, so all access is via raw pointers.
    let mut backtrack: Vec<V::Vec> = Vec::with_capacity(backtrack_len / 2 / V::LANES);

    let (max_i, max_j) = compute(
        v,
        ref_array,
        alt_array,
        parameters,
        overhang_strategy,
        backtrack.as_mut_ptr() as *mut i16,
        backtrack_stride,
        backtrack_len,
    );
    let cigar = get_cigar(
        ref_array,
        alt_array,
        overhang_strategy,
        backtrack.as_ptr() as *const i16,
        backtrack_stride,
        backtrack_len,
        max_i,
        max_j,
    );
    Ok(cigar)
}

#[inline]
fn compute<V: Int32Vector>(
    v: V,
    ref_array: &[u8],
    alt_array: &[u8],
    parameters: Parameters,
    overhang_strategy: OverhangStrategy,
    backtrack: *mut i16,
    backtrack_stride: usize,
    backtrack_len: usize,
) -> (usize, usize) {
    let nrow = ref_array.len();
    let ncol = alt_array.len();

    // This length is used for all arrays in order to reduce the number of variables
    // required for the inner loop.
    let max_len = cmp::max(nrow, ncol) + 1;

    // Reverse one sequence so that comparing subsequences will be along an antidiagonal.
    // Change to one based index.
    // Add padding to allow vector reads starting at the end.
    let mut seq1_rev = vec![0i32; max_len + V::LANES];
    for (i, x) in ref_array.iter().copied().enumerate() {
        seq1_rev[max_len - i - 1] = x.into();
    }

    // Change to one based index.
    // Add padding to allow vector reads starting at the end.
    let mut seq2 = vec![0i32; max_len + V::LANES];
    for (i, x) in alt_array.iter().copied().enumerate() {
        seq2[i + 1] = x.into();
    }

    // E/F calculation needs one previous antidiagonal.
    // This is overwritten in place as we proceed along the antidiagonal.
    let low_init_value = i32::MIN / 2;
    let mut e = vec![low_init_value; max_len + V::LANES];
    let mut f = vec![low_init_value; max_len + V::LANES];

    // H calculation needs two previous antidiagonals.
    // The oldest is overwritten in place as we proceed along the antidiagonal.
    let hwidth = max_len + V::LANES;
    let mut h = vec![0i32; 2 * hwidth];
    //h[max_len / 2] = 0;

    let mut max_score = i32::MIN;
    let mut max_i = 0;
    let mut max_j = 0;

    // TODO: extra antidiagonal?
    for antidiag in 1..=(nrow + ncol) {
        // Calculate endpoints of the antidiagonal: (ilo, jlo) -> (ihi, jhi).
        // Invariant: i + j == antidiag
        let ilo = cmp::min(antidiag - 1, nrow);
        let jhi = cmp::min(antidiag - 1, ncol);
        let ihi = antidiag - jhi;
        let jlo = antidiag - ilo;

        // Swap between copies of H on alternating antidiagonals.
        let prev = (!antidiag & 1) * hwidth;
        let cur = (antidiag & 1) * hwidth;

        macro_rules! compute_inner {
            ($j:expr) => {{
                let i = antidiag - $j;

                let diag_ind = max_len + $j - i;
                let h_left_ind = prev + ((diag_ind - 1) >> 1);
                let h_top_ind = h_left_ind + 1;
                let h_cur_ind = cur + (diag_ind >> 1);

                let inde = max_len - i;
                debug_assert!(inde + V::LANES <= e.len());
                let e10 = unsafe { v.loadu(e.get_unchecked(inde)) };
                let ext_score_h = v.add(e10, v.splat(parameters.gap_extend_penalty));
                debug_assert!(h_left_ind + V::LANES <= h.len());
                let h10 = unsafe { v.loadu(h.get_unchecked(h_left_ind)) };
                let open_score_h = v.add(h10, v.splat(parameters.gap_open_penalty));
                let e11 = v.max(open_score_h, ext_score_h);
                let open_gt_ext_h = v.from_mask(v.cmpgt(open_score_h, ext_score_h));
                let ext_vec = v.andnot(open_gt_ext_h, v.splat(INSERT_EXT.into()));
                debug_assert!(inde + V::LANES <= e.len());
                unsafe { v.storeu(e.get_unchecked_mut(inde), e11) };

                let indf = $j;
                debug_assert!(indf + V::LANES <= f.len());
                let f01 = unsafe { v.loadu(f.get_unchecked(indf)) };
                let ext_score_v = v.add(f01, v.splat(parameters.gap_extend_penalty));
                debug_assert!(h_top_ind + V::LANES <= h.len());
                let h01 = unsafe { v.loadu(h.get_unchecked(h_top_ind)) };
                let open_score_v = v.add(h01, v.splat(parameters.gap_open_penalty));
                let f11 = v.max(ext_score_v, open_score_v);
                let open_gt_ext_v = v.from_mask(v.cmpgt(open_score_v, ext_score_v));
                let ext_vec = v.or(ext_vec, v.andnot(open_gt_ext_v, v.splat(DELETE_EXT.into())));
                debug_assert!(indf + V::LANES <= f.len());
                unsafe { v.storeu(f.get_unchecked_mut(indf), f11) };

                let seq1_ind = max_len - i;
                debug_assert!(seq1_ind + V::LANES <= seq1_rev.len());
                let s1 = unsafe { v.loadu(seq1_rev.get_unchecked(seq1_ind)) };
                let seq2_ind = $j;
                debug_assert!(seq2_ind + V::LANES <= seq2.len());
                let s2 = unsafe { v.loadu(seq2.get_unchecked(seq2_ind)) };
                let cmp11 = v.cmpeq(s1, s2);
                let sbt11 = v.blend(
                    v.splat(parameters.mismatch_penalty),
                    v.splat(parameters.match_value),
                    cmp11,
                );

                debug_assert!(h_cur_ind + V::LANES <= h.len());
                let h00 = unsafe { v.loadu(h.get_unchecked(h_cur_ind)) };
                let m11 = v.add(h00, sbt11);
                let h11 = v.max(v.splat(-100_000_000), m11);

                let bt_vec_0 = v.and(v.splat(INSERT.into()), v.from_mask(v.cmpgt(e11, h11)));
                let h11 = v.max(h11, e11);

                let bt_vec_0 = v.blend(bt_vec_0, v.splat(DELETE.into()), v.cmpgt(f11, h11));
                let h11 = v.max(h11, f11);

                let bt_vec_0 = v.or(bt_vec_0, ext_vec);
                debug_assert!(h_cur_ind + V::LANES <= h.len());
                unsafe {
                    v.storeu(h.get_unchecked_mut(h_cur_ind), h11);
                }
                bt_vec_0
            }};
        }

        let mut j = jlo;
        while j + V::LANES <= jhi {
            let backtrack_ind = j - jlo;
            let bt_vec_0 = compute_inner!(j);
            j += V::LANES;
            let bt_vec_1 = compute_inner!(j);
            j += V::LANES;
            let bt_vec = v.pack(bt_vec_0, bt_vec_1);
            debug_assert!(backtrack_ind + V::LANES * 2 <= backtrack_stride);
            debug_assert!(
                (antidiag * backtrack_stride + backtrack_ind + V::LANES * 2) <= backtrack_len
            );
            unsafe {
                v.stream(
                    backtrack.offset((antidiag * backtrack_stride + backtrack_ind) as isize),
                    bt_vec,
                );
            }
        }
        if j <= jhi {
            let bt_vec_0 = compute_inner!(j);
            let bt_vec_1 = v.zero();
            let bt_vec = v.pack(bt_vec_0, bt_vec_1);
            let backtrack_ind = j - jlo;
            debug_assert!(backtrack_ind + V::LANES * 2 <= backtrack_stride);
            debug_assert!(
                (antidiag * backtrack_stride + backtrack_ind + V::LANES * 2) <= backtrack_len
            );
            unsafe {
                v.stream(
                    backtrack.offset((antidiag * backtrack_stride + backtrack_ind) as isize),
                    bt_vec,
                );
            }
        }

        let curhi = cur + ((max_len + jhi - ihi) >> 1);
        let curlo = cur + ((max_len + jlo - ilo) >> 1);

        // Update edge conditions of antidiagonal.
        match overhang_strategy {
            OverhangStrategy::Indel | OverhangStrategy::LeadingIndel => {
                h[curhi + 1] =
                    parameters.gap_open_penalty + jhi as i32 * parameters.gap_extend_penalty;
                h[curlo - 1] =
                    parameters.gap_open_penalty + ilo as i32 * parameters.gap_extend_penalty;
            }
            OverhangStrategy::SoftClip | OverhangStrategy::Ignore => {
                h[curhi + 1] = 0;
                h[curlo - 1] = 0;
            }
        }
        f[jhi + 1] = low_init_value;
        e[max_len - ilo - 1] = low_init_value;

        // Check for new max score on edges.
        if ilo == nrow {
            if overhang_strategy == OverhangStrategy::SoftClip
                || overhang_strategy == OverhangStrategy::Ignore
            {
                let score = h[curlo];
                if max_score < score
                    || ((max_score == score)
                        && ((ilo as isize - jlo as isize).abs()
                            < (max_i as isize - max_j as isize).abs()))
                {
                    max_score = score;
                    max_i = ilo;
                    max_j = jlo;
                }
            }
        }
        if jhi == ncol {
            let score = h[curhi];
            if (max_score < score)
                || ((max_score == score)
                    && ((max_j == ncol)
                        || ((ihi as isize - jhi as isize).abs()
                            <= (max_i as isize - max_j as isize).abs())))
            {
                max_score = score;
                max_i = ihi;
                max_j = jhi;
            }
        }
    }
    /*
    if overhang_strategy == OverHangStrategy::Indel {
        p->score = h[cur + ((max_len  + ncol - nrow) >> 1)];
    } else {
        p->score = max_score;
    }
    */
    (max_i, max_j)
}

fn get_cigar(
    ref_array: &[u8],
    alt_array: &[u8],
    overhang_strategy: OverhangStrategy,
    backtrack: *const i16,
    backtrack_stride: usize,
    backtrack_len: usize,
    max_i: usize,
    max_j: usize,
) -> (Vec<u8>, isize) {
    let nrow = ref_array.len();
    let ncol = alt_array.len();
    let (mut i, mut j) = match overhang_strategy {
        OverhangStrategy::Indel => (nrow, ncol),
        OverhangStrategy::LeadingIndel => (max_i, ncol),
        OverhangStrategy::SoftClip | OverhangStrategy::Ignore => (max_i, max_j),
    };

    // TODO: is reservation worth it?
    let mut cigar_array: Vec<(i16, u16)> = Vec::with_capacity(nrow + ncol);
    if j < ncol {
        cigar_array.push((SOFTCLIP, (ncol - j) as u16));
    }

    let mut state = 0;
    while i > 0 && j > 0 {
        let antidiag = i + j;
        let jlo = if antidiag <= nrow { 1 } else { antidiag - nrow };
        let backtrack_ind = j - jlo;
        debug_assert!(backtrack_ind < backtrack_stride);
        debug_assert!((antidiag * backtrack_stride + backtrack_ind) < backtrack_len);
        let btrack =
            unsafe { *backtrack.offset((antidiag * backtrack_stride + backtrack_ind) as isize) };
        if state == INSERT_EXT {
            j -= 1;
            cigar_array.last_mut().unwrap().1 += 1;
            state = btrack & INSERT_EXT;
        } else if state == DELETE_EXT {
            i -= 1;
            cigar_array.last_mut().unwrap().1 += 1;
            state = btrack & DELETE_EXT;
        } else {
            match btrack & 3 {
                MATCH => {
                    i -= 1;
                    j -= 1;
                    cigar_array.push((MATCH, 1));
                    state = 0;
                }
                INSERT => {
                    j -= 1;
                    cigar_array.push((INSERT, 1));
                    state = btrack & INSERT_EXT;
                }
                DELETE => {
                    i -= 1;
                    cigar_array.push((DELETE, 1));
                    state = btrack & DELETE_EXT;
                }
                _ => unreachable!(),
            }
        }
    }

    let alignment_offset = if overhang_strategy == OverhangStrategy::SoftClip {
        if j > 0 {
            cigar_array.push((SOFTCLIP, j as u16));
        }
        i as isize
    } else if overhang_strategy == OverhangStrategy::Ignore {
        if j > 0 {
            cigar_array.push((cigar_array.last().unwrap().0, j as u16));
        }
        i as isize - j as isize
    } else {
        if i > 0 {
            cigar_array.push((DELETE, i as u16));
        } else if j > 0 {
            cigar_array.push((INSERT, j as u16));
        }
        0
    };

    // TODO: this can be avoided
    let mut cigar_array2: Vec<(i16, u16)> = Vec::with_capacity(cigar_array.len());
    let mut prev = cigar_array[0];
    for cur in &cigar_array[1..] {
        if prev.0 == cur.0 {
            prev.1 += cur.1;
        } else {
            cigar_array2.push(prev);
            prev = *cur;
        }
    }
    cigar_array2.push(prev);

    let mut cigar = Vec::new();
    for (state, count) in cigar_array2.iter().rev().copied() {
        let state = match state {
            MATCH => b'M',
            INSERT => b'I',
            DELETE => b'D',
            SOFTCLIP => b'S',
            _ => b'R',
        };
        write!(&mut cigar, "{}", count).unwrap();
        cigar.push(state);
    }
    (cigar, alignment_offset.into())
}

#[cfg(all(target_arch = "x86_64", not(feature = "c-avx")))]
mod x86_64_avx {
    use super::{compute_and_backtrack, Align, OverhangStrategy, Parameters, Result};
    use crate::vector::AvxI32x8;

    #[target_feature(enable = "avx2")]
    unsafe fn target_align_i32x8(
        ref_array: &[u8],
        alt_array: &[u8],
        parameters: Parameters,
        overhang_strategy: OverhangStrategy,
    ) -> Result<(Vec<u8>, isize)> {
        let v = AvxI32x8::new_unchecked();
        compute_and_backtrack(v, ref_array, alt_array, parameters, overhang_strategy)
    }

    pub fn align_i32x8() -> Option<Align> {
        if AvxI32x8::new().is_some() {
            fn f(
                ref_array: &[u8],
                alt_array: &[u8],
                parameters: Parameters,
                overhang_strategy: OverhangStrategy,
            ) -> Result<(Vec<u8>, isize)> {
                unsafe { target_align_i32x8(ref_array, alt_array, parameters, overhang_strategy) }
            }
            Some(f)
        } else {
            None
        }
    }
}

#[cfg(all(target_arch = "x86_64", not(feature = "c-avx512"), feature = "nightly"))]
mod x86_64_avx512 {
    use super::{compute_and_backtrack, Align, OverhangStrategy, Parameters, Result};
    use crate::vector::AvxI32x16;

    #[cfg(feature = "nightly")]
    #[target_feature(enable = "avx512f,avx512dq")]
    unsafe fn target_align_i32x16(
        ref_array: &[u8],
        alt_array: &[u8],
        parameters: Parameters,
        overhang_strategy: OverhangStrategy,
    ) -> Result<(Vec<u8>, isize)> {
        let v = AvxI32x16::new_unchecked();
        compute_and_backtrack(v, ref_array, alt_array, parameters, overhang_strategy)
    }

    #[cfg(feature = "nightly")]
    pub fn align_i32x16() -> Option<Align> {
        if AvxI32x16::new().is_some() {
            fn f(
                ref_array: &[u8],
                alt_array: &[u8],
                parameters: Parameters,
                overhang_strategy: OverhangStrategy,
            ) -> Result<(Vec<u8>, isize)> {
                unsafe { target_align_i32x16(ref_array, alt_array, parameters, overhang_strategy) }
            }
            Some(f)
        } else {
            None
        }
    }
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
