//! PairHMM forward algorithm.
use crate::vector::Vector;
use num_traits::{Float, One, Zero};
use std::{cmp, mem};

lazy_static::lazy_static! {
    static ref FORWARD32: Option<ForwardF32> = forward_f32();
    static ref FORWARD64: Forward = forward_f64();
    static ref LOG10_INITIAL_CONSTANT_32: f32 = (120.0f32).exp2().log10();
    static ref CONTEXT32: Context<f32> = Context::new();
    static ref CONTEXT64: Context<f64> = Context::new();
}

/// The type of a PairHMM forward function that returns an `f32`.
///
/// This is returned by the CPU feature detection functions.
pub type ForwardF32 = fn(hap: &[u8], rs: &[u8], q: &[u8], i: &[u8], d: &[u8], c: &[u8]) -> f32;

/// The type of a PairHMM forward function that returns an `f64`.
///
/// This is returned by the CPU feature detection functions.
pub type Forward = fn(hap: &[u8], rs: &[u8], q: &[u8], i: &[u8], d: &[u8], c: &[u8]) -> f64;

/// Return the scalar `f32` implementation of the PairHMM forward function.
///
/// The compiler may autovectorize this function.
///
/// This may be slower than the scalar `f64` implementation due to autovectorization differences.
pub fn forward_f32x1() -> ForwardF32 {
    fn f(hap: &[u8], rs: &[u8], q: &[u8], i: &[u8], d: &[u8], c: &[u8]) -> f32 {
        let ctx = &CONTEXT32;
        unsafe { compute::<f32>(ctx, hap, rs, q, i, d, c) }
    }
    f
}

/// Return the scalar `f64` implementation of the PairHMM forward function.
///
/// The compiler may autovectorize this function.
pub fn forward_f64x1() -> Forward {
    fn f(hap: &[u8], rs: &[u8], q: &[u8], i: &[u8], d: &[u8], c: &[u8]) -> f64 {
        let ctx = &CONTEXT64;
        unsafe { compute::<f64>(ctx, hap, rs, q, i, d, c) }
    }
    f
}

/// Return the `f32x8` implementation of the PairHMM forward function if supported by the CPU features.
pub fn forward_f32x8() -> Option<ForwardF32> {
    cfg_if::cfg_if! {
        if #[cfg(all(target_arch = "x86_64", feature = "c-avx"))] {
            c::forward_f32x8()
        } else if #[cfg(target_arch = "x86_64")] {
            x86_64_avx::forward_f32x8()
        } else {
            None
        }
    }
}

/// Return the `f64x4` implementation of the PairHMM forward function if supported by the CPU features.
pub fn forward_f64x4() -> Option<Forward> {
    cfg_if::cfg_if! {
        if #[cfg(all(target_arch = "x86_64", feature = "c-avx"))] {
            c::forward_f64x4()
        } else if #[cfg(target_arch = "x86_64")] {
            x86_64_avx::forward_f64x4()
        } else {
            None
        }
    }
}

/// Return the `f32x16` implementation of the PairHMM forward function if supported by the CPU features.
pub fn forward_f32x16() -> Option<ForwardF32> {
    cfg_if::cfg_if! {
        if #[cfg(all(target_arch = "x86_64", feature = "c-avx512"))] {
            c::forward_f32x16()
        } else if #[cfg(all(target_arch = "x86_64", feature = "nightly"))] {
            x86_64_avx512::forward_f32x16()
        } else {
            None
        }
    }
}

/// Return the `f64x8` implementation of the PairHMM forward function if supported by the CPU features.
pub fn forward_f64x8() -> Option<Forward> {
    cfg_if::cfg_if! {
        if #[cfg(all(target_arch = "x86_64", feature = "c-avx512"))] {
            c::forward_f64x8()
        } else if #[cfg(all(target_arch = "x86_64", feature = "nightly"))] {
            x86_64_avx512::forward_f64x8()
        } else {
            None
        }
    }
}

/// Return the fastest `f32` PairHMM forward function that is supported by the CPU features.
///
/// Does not return `forward_f32x1` since `forward_f64x1` may be faster.
pub fn forward_f32() -> Option<ForwardF32> {
    forward_f32x16().or_else(forward_f32x8)
}

/// Return the fastest `f64` PairHMM forward function that is supported by the CPU features.
pub fn forward_f64() -> Forward {
    forward_f64x8()
        .or_else(forward_f64x4)
        .unwrap_or_else(forward_f64x1)
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
    FORWARD64(hap, rs, q, i, d, c)
}

trait ContextFloat: Float + From<u8> {
    const INITIAL_CONSTANT_EXP: Self;
    const TEN: Self;
}

impl ContextFloat for f32 {
    const INITIAL_CONSTANT_EXP: Self = 120.0;
    const TEN: Self = 10.0;
}

impl ContextFloat for f64 {
    const INITIAL_CONSTANT_EXP: Self = 1020.0;
    const TEN: Self = 10.0;
}

/// Context containing calculations that can be reused for different data.
struct Context<T: ContextFloat> {
    initial_constant: T,
    log10_initial_constant: T,
    convert: [u8; 256],
    ph2pr: [T; 128],
    match_to_match_prob: Vec<T>,
}

impl<T: ContextFloat> Context<T> {
    const MAX_QUAL: u8 = 127;

    fn new() -> Self {
        let initial_constant = T::INITIAL_CONSTANT_EXP.exp2();
        let log10_initial_constant = initial_constant.log10();

        let mut convert = [0; 256];
        convert[b'A' as usize] = 0;
        convert[b'C' as usize] = 1;
        convert[b'T' as usize] = 2;
        convert[b'G' as usize] = 3;
        convert[b'N' as usize] = 4;

        let mut ph2pr = [T::zero(); 128];
        for x in 0..128u8 {
            ph2pr[x as usize] = T::TEN.powf(-(<T as From<u8>>::from(x) / T::TEN));
        }

        let inv_ln10 = T::one() / T::TEN.ln();
        let mut match_to_match_prob =
            Vec::with_capacity(((Self::MAX_QUAL as usize + 1) * (Self::MAX_QUAL as usize + 2)) / 2);
        for i in 0..=Self::MAX_QUAL {
            for j in 0..=i {
                let a = <T as num_traits::NumCast>::from(-0.1f32).unwrap() * From::from(i);
                let b = <T as num_traits::NumCast>::from(-0.1f32).unwrap() * From::from(j);
                let sum = T::TEN.powf(a) + T::TEN.powf(b);
                let match_to_match_log10 = (-T::min(T::one(), sum)).ln_1p() * inv_ln10;
                match_to_match_prob.push(T::TEN.powf(match_to_match_log10));
            }
        }

        Context {
            convert,
            initial_constant,
            log10_initial_constant,
            ph2pr,
            match_to_match_prob,
        }
    }

    fn match_to_match_prob(&self, mut a: u8, mut b: u8) -> T {
        if a > b {
            mem::swap(&mut a, &mut b);
        }
        debug_assert!(b <= Self::MAX_QUAL);
        self.match_to_match_prob[((b as usize * (b as usize + 1)) / 2) + a as usize]
    }
}

struct BitMaskVec<V: Vector> {
    rows: V::IndexArray,
    shift: V::MaskArray,
    col_masks: Vec<[V::Mask; 5]>,
}

impl<V: Vector> BitMaskVec<V> {
    fn new(hap: &[u8], convert: &[u8; 256]) -> Self {
        // Number of antidiagonals in a stripe.
        let max_d = hap.len() + V::LANES - 1;
        let mut col_masks = vec![
            [
                V::Mask::default(),
                V::Mask::default(),
                V::Mask::default(),
                V::Mask::default(),
                !V::Mask::default()
            ];
            (max_d + V::MASK_BITS - 1) / V::MASK_BITS
        ];
        for (col, hap) in hap.iter().copied().enumerate() {
            let masks = &mut col_masks[col / V::MASK_BITS];
            let bit = V::Mask::from(1) << (V::MASK_BITS - 1 - col % V::MASK_BITS);
            let hap = convert[hap as usize];
            if hap == 4 {
                for j in 0..5 {
                    masks[j] |= bit;
                }
            } else {
                masks[hap as usize] |= bit;
            }
        }
        BitMaskVec {
            rows: V::IndexArray::default(),
            shift: V::MaskArray::default(),
            col_masks,
        }
    }

    fn init_row(&mut self, rs: &[u8], convert: &[u8; 256]) {
        for (i, rs) in rs.iter().take(V::LANES).copied().enumerate() {
            self.rows[i] = convert[rs as usize];
        }
        self.shift = V::MaskArray::default();
    }

    // Build a mask for use with Vector::blend to select between
    // distm and 1-distm.
    //
    // Only the MSB of each element is used for the blend, but then we
    // shift left for the next loop.
    //
    // Each element corresponds to a row, and each bit corresponds to a column.
    // A bit is set if the row and column match.
    fn match_col(&mut self, index: usize) -> V::MaskVec {
        let mut masks = V::MaskArray::default();
        masks[0] = self.col_masks[index][self.rows[0] as usize];
        for row in 1..V::LANES {
            let mask = self.col_masks[index][self.rows[row] as usize];
            // The shifts are due to using antidiagonals.
            masks[row] = (mask >> row) | self.shift[row];
            self.shift[row] = mask << (V::MASK_BITS - row);
        }
        V::mask_from_array(masks)
    }
}

/// Perform the PairHMM forward calculation.
///
/// Operations are on horizontal stripes. The number of rows in a stripe is equal
/// to the number of vector lanes. Each stripe depends on the last row of the
/// previous stripe.
///
/// Operations within stripes are on antidiagonals. Each antidiagonal depends on
/// the previous two antidiagonals.
///
/// # Panics
/// Panics when `rs.len() == 0`.
/// Panics when the length of `q`, `i`, `d` or `c` is less than the length of `rs`.
#[inline]
unsafe fn compute<V: Vector>(
    ctx: &Context<V::Float>,
    hap: &[u8],
    rs: &[u8],
    q: &[u8],
    i: &[u8],
    d: &[u8],
    c: &[u8],
) -> V::Float
where
    V::Float: ContextFloat,
{
    let mut bitmask_vec = BitMaskVec::<V>::new(hap, &ctx.convert);

    let mode = V::set_flush_zero_mode();

    let shift_len = hap.len() + rs.len() + V::LANES;
    let mut shift_m = vec![V::Float::zero(); shift_len];
    let mut shift_x = vec![V::Float::zero(); shift_len];
    let init_y = ctx.initial_constant / num_traits::NumCast::from(hap.len()).unwrap();
    let mut shift_y = vec![init_y; shift_len];

    let mut m_t_1 = V::zero();
    let mut m_t_1_y = V::zero();
    let mut m_t_2 = V::zero();
    let mut x_t_1 = V::zero();
    let mut x_t_2 = V::zero();
    let mut y_t_1 = V::zero();
    let mut y_t_2 = V::first_element(init_y);

    assert!(rs.len() > 0);
    let mut stripe_cnt = (rs.len() + V::LANES - 1) / V::LANES;
    // The last stripe needs to be handled differently to generate the sum.
    stripe_cnt -= 1;
    let remaining_rows = rs.len() - stripe_cnt * V::LANES;

    for stripe in 0..stripe_cnt {
        let row_base = stripe * V::LANES;
        let mut p_gapm = V::FloatArray::default();
        let mut p_mm = V::FloatArray::default();
        let mut p_mx = V::FloatArray::default();
        let mut p_xx = V::FloatArray::default();
        let mut p_my = V::FloatArray::default();
        let mut p_yy = V::FloatArray::default();
        let mut distm = V::FloatArray::default();
        for r in 0..V::LANES {
            let row = row_base + r;
            let i = i[row] & 127;
            let d = d[row] & 127;
            let c = c[row] & 127;
            p_gapm[r] = V::Float::one() - ctx.ph2pr[c as usize];
            p_mm[r] = ctx.match_to_match_prob(i, d);
            p_mx[r] = ctx.ph2pr[i as usize];
            p_xx[r] = ctx.ph2pr[c as usize];
            p_my[r] = ctx.ph2pr[d as usize];
            p_yy[r] = ctx.ph2pr[c as usize];

            let q = q[row] & 127;
            distm[r] = ctx.ph2pr[q as usize];
        }
        let p_gapm = V::from_array(p_gapm);
        let p_mm = V::from_array(p_mm);
        let p_mx = V::from_array(p_mx);
        let p_xx = V::from_array(p_xx);
        let p_my = V::from_array(p_my);
        let p_yy = V::from_array(p_yy);
        let distm = V::from_array(distm);
        let one_distm = V::sub(V::splat(V::Float::one()), distm);
        let distm = V::div(distm, V::splat(V::Float::from(3)));

        bitmask_vec.init_row(&rs[row_base..], &ctx.convert);
        let mut begin_d = 0;
        let max_d = hap.len() + V::LANES - 1;
        while begin_d < max_d {
            let mut bitmask = bitmask_vec.match_col(begin_d / V::MASK_BITS);
            let num_d = cmp::min(max_d - begin_d, V::MASK_BITS);
            for d in 0..num_d {
                let shift_idx = begin_d + d;
                debug_assert!(shift_idx + V::LANES < shift_len);

                let m_t_base = V::add(
                    V::add(V::mul(m_t_2, p_mm), V::mul(x_t_2, p_gapm)),
                    V::mul(y_t_2, p_gapm),
                );
                m_t_2 = m_t_1;
                x_t_2 = x_t_1;
                y_t_2 = V::element_shift(
                    y_t_1,
                    shift_y.get_unchecked(shift_idx + V::LANES),
                    shift_y.get_unchecked_mut(shift_idx),
                );

                let distm_sel = V::blend(distm, one_distm, bitmask);
                bitmask = V::mask_shift(bitmask);
                let m_t = V::mul(m_t_base, distm_sel);

                let x_t = V::add(V::mul(m_t_1, p_mx), V::mul(x_t_1, p_xx));
                m_t_1 = V::element_shift(
                    m_t,
                    shift_m.get_unchecked(shift_idx + V::LANES),
                    shift_m.get_unchecked_mut(shift_idx),
                );

                let y_t = V::add(V::mul(m_t_1_y, p_my), V::mul(y_t_1, p_yy));
                y_t_1 = y_t;

                x_t_1 = V::element_shift(
                    x_t,
                    shift_x.get_unchecked(shift_idx + V::LANES),
                    shift_x.get_unchecked_mut(shift_idx),
                );
                m_t_1_y = m_t;
            }
            begin_d += V::MASK_BITS;
        }

        m_t_1 = V::first_element(shift_m[V::LANES - 1]);
        m_t_1_y = m_t_1;
        m_t_2 = V::zero();
        x_t_1 = V::first_element(shift_x[V::LANES - 1]);
        x_t_2 = V::zero();
        y_t_1 = V::zero();
        y_t_2 = V::zero();
    }

    // The result is the sum of M and X in the last row of the last stripe.
    // Since extracting the last row from the vector can be slow, we sum across
    // all lanes in the stripe, and then extract the last row once at the end.
    let mut sum_m = V::zero();
    let mut sum_x = V::zero();
    {
        let row_base = stripe_cnt * V::LANES;
        let mut p_gapm = V::FloatArray::default();
        let mut p_mm = V::FloatArray::default();
        let mut p_mx = V::FloatArray::default();
        let mut p_xx = V::FloatArray::default();
        let mut p_my = V::FloatArray::default();
        let mut p_yy = V::FloatArray::default();
        let mut distm = V::FloatArray::default();
        for r in 0..remaining_rows {
            let row = row_base + r;
            let i = i[row] & 127;
            let d = d[row] & 127;
            let c = c[row] & 127;
            p_gapm[r] = V::Float::one() - ctx.ph2pr[c as usize];
            p_mm[r] = ctx.match_to_match_prob(i, d);
            p_mx[r] = ctx.ph2pr[i as usize];
            p_xx[r] = ctx.ph2pr[c as usize];
            p_my[r] = ctx.ph2pr[d as usize];
            p_yy[r] = ctx.ph2pr[c as usize];

            let q = q[row] & 127;
            distm[r] = ctx.ph2pr[q as usize];
        }
        let p_gapm = V::from_array(p_gapm);
        let p_mm = V::from_array(p_mm);
        let p_mx = V::from_array(p_mx);
        let p_xx = V::from_array(p_xx);
        let p_my = V::from_array(p_my);
        let p_yy = V::from_array(p_yy);
        let distm = V::from_array(distm);
        let one_distm = V::sub(V::splat(V::Float::one()), distm);
        let distm = V::div(distm, V::splat(V::Float::from(3)));

        bitmask_vec.init_row(&rs[row_base..], &ctx.convert);
        let mut begin_d = 0;
        let max_d = hap.len() + remaining_rows - 1;
        while begin_d < max_d {
            let mut bitmask = bitmask_vec.match_col(begin_d / V::MASK_BITS);
            let num_d = cmp::min(max_d - begin_d, V::MASK_BITS);
            for d in 0..num_d {
                let distm_sel = V::blend(distm, one_distm, bitmask);
                bitmask = V::mask_shift(bitmask);
                let m_t = V::mul(
                    V::add(
                        V::add(V::mul(m_t_2, p_mm), V::mul(x_t_2, p_gapm)),
                        V::mul(y_t_2, p_gapm),
                    ),
                    distm_sel,
                );
                let x_t = V::add(V::mul(m_t_1, p_mx), V::mul(x_t_1, p_xx));
                let y_t = V::add(V::mul(m_t_1_y, p_my), V::mul(y_t_1, p_yy));
                sum_m = V::add(sum_m, m_t);
                sum_x = V::add(sum_x, x_t);

                let shift_idx = begin_d + d + V::LANES;
                m_t_2 = m_t_1;
                m_t_1 = V::element_shift_in(m_t, &shift_m[shift_idx]);
                x_t_2 = x_t_1;
                x_t_1 = V::element_shift_in(x_t, &shift_x[shift_idx]);
                y_t_2 = V::element_shift_in(y_t_1, &shift_y[shift_idx]);
                y_t_1 = y_t;
                m_t_1_y = m_t;
            }
            begin_d += V::MASK_BITS;
        }
    }
    let sum = V::add(sum_m, sum_x);
    let sums = V::to_array(sum);
    let result = sums[remaining_rows - 1].log10() - ctx.log10_initial_constant;
    V::restore_flush_zero_mode(mode);
    result
}

#[cfg(all(target_arch = "x86_64", not(feature = "c-avx")))]
mod x86_64_avx {
    use super::{compute, Forward, ForwardF32, CONTEXT32, CONTEXT64};
    use crate::vector::{AvxF32x8, AvxF64x4};

    #[target_feature(enable = "avx")]
    unsafe fn target_forward_f32x8(
        hap: &[u8],
        rs: &[u8],
        q: &[u8],
        i: &[u8],
        d: &[u8],
        c: &[u8],
    ) -> f32 {
        let ctx = &CONTEXT32;
        compute::<AvxF32x8>(ctx, hap, rs, q, i, d, c)
    }

    pub fn forward_f32x8() -> Option<ForwardF32> {
        if is_x86_feature_detected!("avx") {
            fn f(hap: &[u8], rs: &[u8], q: &[u8], i: &[u8], d: &[u8], c: &[u8]) -> f32 {
                unsafe { target_forward_f32x8(hap, rs, q, i, d, c) }
            }
            Some(f)
        } else {
            None
        }
    }

    #[target_feature(enable = "avx")]
    unsafe fn target_forward_f64x4(
        hap: &[u8],
        rs: &[u8],
        q: &[u8],
        i: &[u8],
        d: &[u8],
        c: &[u8],
    ) -> f64 {
        let ctx = &CONTEXT64;
        compute::<AvxF64x4>(ctx, hap, rs, q, i, d, c)
    }

    pub fn forward_f64x4() -> Option<Forward> {
        if is_x86_feature_detected!("avx") {
            fn f(hap: &[u8], rs: &[u8], q: &[u8], i: &[u8], d: &[u8], c: &[u8]) -> f64 {
                unsafe { target_forward_f64x4(hap, rs, q, i, d, c) }
            }
            Some(f)
        } else {
            None
        }
    }
}

#[cfg(all(target_arch = "x86_64", not(feature = "c-avx512"), feature = "nightly"))]
mod x86_64_avx512 {
    use super::{compute, Forward, ForwardF32, CONTEXT32, CONTEXT64};
    use crate::vector::{AvxF32x16, AvxF64x8};

    #[cfg(feature = "nightly")]
    #[target_feature(enable = "avx512f")]
    unsafe fn target_forward_f32x16(
        hap: &[u8],
        rs: &[u8],
        q: &[u8],
        i: &[u8],
        d: &[u8],
        c: &[u8],
    ) -> f32 {
        let ctx = &CONTEXT32;
        compute::<AvxF32x16>(ctx, hap, rs, q, i, d, c)
    }

    #[cfg(feature = "nightly")]
    pub fn forward_f32x16() -> Option<ForwardF32> {
        if is_x86_feature_detected!("avx512f") {
            fn f(hap: &[u8], rs: &[u8], q: &[u8], i: &[u8], d: &[u8], c: &[u8]) -> f32 {
                unsafe { target_forward_f32x16(hap, rs, q, i, d, c) }
            }
            Some(f)
        } else {
            None
        }
    }

    #[cfg(feature = "nightly")]
    #[target_feature(enable = "avx512f")]
    unsafe fn target_forward_f64x8(
        hap: &[u8],
        rs: &[u8],
        q: &[u8],
        i: &[u8],
        d: &[u8],
        c: &[u8],
    ) -> f64 {
        let ctx = &CONTEXT64;
        compute::<AvxF64x8>(ctx, hap, rs, q, i, d, c)
    }

    #[cfg(feature = "nightly")]
    pub fn forward_f64x8() -> Option<Forward> {
        if is_x86_feature_detected!("avx512f") {
            fn f(hap: &[u8], rs: &[u8], q: &[u8], i: &[u8], d: &[u8], c: &[u8]) -> f64 {
                unsafe { target_forward_f64x8(hap, rs, q, i, d, c) }
            }
            Some(f)
        } else {
            None
        }
    }
}

#[cfg(all(target_arch = "x86_64", feature = "c-core"))]
mod c {
    #![cfg_attr(not(all(feature = "c-avx", feature = "c-avx512")), allow(unused))]
    use super::{Forward, ForwardF32};
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

    impl Testcase {
        fn new(hap: &[u8], rs: &[u8], q: &[u8], i: &[u8], d: &[u8], c: &[u8]) -> Testcase {
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
    }

    extern "C" {
        //fn compute_avxd_c(arg1: *mut Testcase) -> f64;
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

    pub fn forward_f32x8() -> Option<ForwardF32> {
        if !is_x86_feature_detected!("avx") {
            return None;
        }
        convert_char_init();
        fn f(hap: &[u8], rs: &[u8], q: &[u8], i: &[u8], d: &[u8], c: &[u8]) -> f32 {
            let mut tc = Testcase::new(hap, rs, q, i, d, c);
            unsafe { compute_avxs(&mut tc) }
        }
        Some(f)
    }

    pub fn forward_f64x4() -> Option<Forward> {
        if !is_x86_feature_detected!("avx") {
            return None;
        }
        convert_char_init();
        fn f(hap: &[u8], rs: &[u8], q: &[u8], i: &[u8], d: &[u8], c: &[u8]) -> f64 {
            let mut tc = Testcase::new(hap, rs, q, i, d, c);
            unsafe { compute_avxd(&mut tc) }
        }
        Some(f)
    }

    pub fn forward_f32x16() -> Option<ForwardF32> {
        if !is_x86_feature_detected!("avx512f")
            || !is_x86_feature_detected!("avx512dq")
            || !is_x86_feature_detected!("avx512vl")
        {
            return None;
        }
        convert_char_init();
        fn f(hap: &[u8], rs: &[u8], q: &[u8], i: &[u8], d: &[u8], c: &[u8]) -> f32 {
            let mut tc = Testcase::new(hap, rs, q, i, d, c);
            unsafe { compute_avx512s(&mut tc) }
        }
        Some(f)
    }

    pub fn forward_f64x8() -> Option<Forward> {
        if !is_x86_feature_detected!("avx512f")
            || !is_x86_feature_detected!("avx512dq")
            || !is_x86_feature_detected!("avx512vl")
        {
            return None;
        }
        convert_char_init();
        fn f(hap: &[u8], rs: &[u8], q: &[u8], i: &[u8], d: &[u8], c: &[u8]) -> f64 {
            let mut tc = Testcase::new(hap, rs, q, i, d, c);
            unsafe { compute_avx512d(&mut tc) }
        }
        Some(f)
    }
}
