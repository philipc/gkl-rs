use std::ops;

pub(crate) trait Vector {
    type Float;
    type FloatArray: ops::IndexMut<usize, Output = Self::Float> + Default;
    type FloatVec: Copy;
    type Mask: Copy
        + Default
        + ops::Shl<usize, Output = Self::Mask>
        + ops::Shr<usize, Output = Self::Mask>
        + ops::Not<Output = Self::Mask>
        + ops::BitOr<Output = Self::Mask>
        + ops::BitOrAssign
        + From<u8>;
    type MaskArray: ops::IndexMut<usize, Output = Self::Mask> + Default;
    type MaskVec: Copy;
    type IndexArray: ops::IndexMut<usize, Output = u8> + Default;
    const MASK_BITS: usize;
    const LANES: usize;

    type Mode;
    unsafe fn set_flush_zero_mode() -> Self::Mode;
    unsafe fn restore_flush_zero_mode(mode: Self::Mode);

    // Vector with all elements set to 0.
    unsafe fn zero() -> Self::FloatVec;

    // Vector with given first element, and remaining elements set to 0.
    unsafe fn first_element(f: Self::Float) -> Self::FloatVec;

    // Vector with all elements set to given value.
    unsafe fn splat(f: Self::Float) -> Self::FloatVec;

    fn from_array(a: Self::FloatArray) -> Self::FloatVec;
    fn to_array(a: Self::FloatVec) -> Self::FloatArray;

    unsafe fn add(a: Self::FloatVec, b: Self::FloatVec) -> Self::FloatVec;
    unsafe fn sub(a: Self::FloatVec, b: Self::FloatVec) -> Self::FloatVec;
    unsafe fn mul(a: Self::FloatVec, b: Self::FloatVec) -> Self::FloatVec;
    unsafe fn div(a: Self::FloatVec, b: Self::FloatVec) -> Self::FloatVec;
    unsafe fn blend(a: Self::FloatVec, b: Self::FloatVec, mask: Self::MaskVec) -> Self::FloatVec;

    // Shift in a new first element. Shift out the previous last element.
    unsafe fn element_shift(
        x: Self::FloatVec,
        shift_in: *const Self::Float,
        shift_out: &mut Self::Float,
    ) -> Self::FloatVec;

    // Shift in a new first element. Discard the previous last element.
    unsafe fn element_shift_in(x: Self::FloatVec, shift_in: *const Self::Float) -> Self::FloatVec;

    fn mask_from_array(a: Self::MaskArray) -> Self::MaskVec;
    unsafe fn mask_shift(x: Self::MaskVec) -> Self::MaskVec;
}

impl Vector for f32 {
    type Float = f32;
    type FloatArray = [f32; 1];
    type FloatVec = f32;
    type Mask = u32;
    type MaskArray = [u32; 1];
    type MaskVec = u32;
    type IndexArray = [u8; 1];
    const MASK_BITS: usize = 32;
    const LANES: usize = 1;

    type Mode = ();

    unsafe fn set_flush_zero_mode() -> Self::Mode {
        ()
    }

    unsafe fn restore_flush_zero_mode(_mode: Self::Mode) {}

    unsafe fn zero() -> Self::FloatVec {
        0.0
    }

    unsafe fn first_element(f: Self::Float) -> Self::FloatVec {
        f
    }

    unsafe fn splat(f: Self::Float) -> Self::FloatVec {
        f
    }

    fn from_array(a: Self::FloatArray) -> Self::FloatVec {
        a[0]
    }

    fn to_array(a: Self::FloatVec) -> Self::FloatArray {
        [a]
    }

    unsafe fn add(a: Self::FloatVec, b: Self::FloatVec) -> Self::FloatVec {
        a + b
    }

    unsafe fn sub(a: Self::FloatVec, b: Self::FloatVec) -> Self::FloatVec {
        a - b
    }

    unsafe fn mul(a: Self::FloatVec, b: Self::FloatVec) -> Self::FloatVec {
        a * b
    }

    unsafe fn div(a: Self::FloatVec, b: Self::FloatVec) -> Self::FloatVec {
        a / b
    }

    unsafe fn blend(a: Self::FloatVec, b: Self::FloatVec, mask: Self::MaskVec) -> Self::FloatVec {
        if mask & (1 << 31) == 0 {
            a
        } else {
            b
        }
    }

    unsafe fn element_shift(
        x: Self::FloatVec,
        shift_in: *const Self::Float,
        shift_out: &mut Self::Float,
    ) -> Self::FloatVec {
        *shift_out = x;
        *shift_in
    }

    unsafe fn element_shift_in(_x: Self::FloatVec, shift_in: *const Self::Float) -> Self::FloatVec {
        *shift_in
    }

    fn mask_from_array(a: Self::MaskArray) -> Self::MaskVec {
        a[0]
    }

    unsafe fn mask_shift(x: Self::MaskVec) -> Self::MaskVec {
        x << 1
    }
}

impl Vector for f64 {
    type Float = f64;
    type FloatArray = [f64; 1];
    type FloatVec = f64;
    type Mask = u32;
    type MaskArray = [u32; 1];
    type MaskVec = u32;
    type IndexArray = [u8; 1];
    const MASK_BITS: usize = 32;
    const LANES: usize = 1;

    type Mode = ();

    unsafe fn set_flush_zero_mode() -> Self::Mode {
        ()
    }

    unsafe fn restore_flush_zero_mode(_mode: Self::Mode) {}

    unsafe fn zero() -> Self::FloatVec {
        0.0
    }

    unsafe fn first_element(f: Self::Float) -> Self::FloatVec {
        f
    }

    unsafe fn splat(f: Self::Float) -> Self::FloatVec {
        f
    }

    fn from_array(a: Self::FloatArray) -> Self::FloatVec {
        a[0]
    }

    fn to_array(a: Self::FloatVec) -> Self::FloatArray {
        [a]
    }

    unsafe fn add(a: Self::FloatVec, b: Self::FloatVec) -> Self::FloatVec {
        a + b
    }

    unsafe fn sub(a: Self::FloatVec, b: Self::FloatVec) -> Self::FloatVec {
        a - b
    }

    unsafe fn mul(a: Self::FloatVec, b: Self::FloatVec) -> Self::FloatVec {
        a * b
    }

    unsafe fn div(a: Self::FloatVec, b: Self::FloatVec) -> Self::FloatVec {
        a / b
    }

    unsafe fn blend(a: Self::FloatVec, b: Self::FloatVec, mask: Self::MaskVec) -> Self::FloatVec {
        if mask & (1 << 31) == 0 {
            a
        } else {
            b
        }
    }

    unsafe fn element_shift(
        x: Self::FloatVec,
        shift_in: *const Self::Float,
        shift_out: &mut Self::Float,
    ) -> Self::FloatVec {
        *shift_out = x;
        *shift_in
    }

    unsafe fn element_shift_in(_x: Self::FloatVec, shift_in: *const Self::Float) -> Self::FloatVec {
        *shift_in
    }

    fn mask_from_array(a: Self::MaskArray) -> Self::MaskVec {
        a[0]
    }

    unsafe fn mask_shift(x: Self::MaskVec) -> Self::MaskVec {
        x << 1
    }
}

#[cfg(all(target_arch = "x86_64", not(feature = "c-avx")))]
mod x86_64_avx {
    use super::Vector;
    use std::arch::x86_64::*;
    use std::mem;

    pub struct AvxF32x8;

    impl Vector for AvxF32x8 {
        type Float = f32;
        type FloatArray = [f32; 8];
        type FloatVec = __m256;
        type Mask = u32;
        type MaskArray = [u32; 8];
        type MaskVec = __m256;
        type IndexArray = [u8; 8];
        const MASK_BITS: usize = 32;
        const LANES: usize = 8;

        type Mode = u32;

        unsafe fn set_flush_zero_mode() -> Self::Mode {
            let mode = _MM_GET_FLUSH_ZERO_MODE();
            _MM_SET_FLUSH_ZERO_MODE(0x8000);
            mode
        }

        unsafe fn restore_flush_zero_mode(mode: Self::Mode) {
            _MM_SET_FLUSH_ZERO_MODE(mode);
        }

        #[target_feature(enable = "avx")]
        unsafe fn zero() -> Self::FloatVec {
            _mm256_setzero_ps()
        }

        #[target_feature(enable = "avx")]
        unsafe fn first_element(f: Self::Float) -> Self::FloatVec {
            _mm256_set_ps(0., 0., 0., 0., 0., 0., 0., f)
        }

        #[target_feature(enable = "avx")]
        unsafe fn splat(f: Self::Float) -> Self::FloatVec {
            _mm256_set1_ps(f)
        }

        fn from_array(a: Self::FloatArray) -> Self::FloatVec {
            unsafe { mem::transmute(a) }
        }

        fn to_array(a: Self::FloatVec) -> Self::FloatArray {
            unsafe { mem::transmute(a) }
        }

        #[target_feature(enable = "avx")]
        unsafe fn add(a: Self::FloatVec, b: Self::FloatVec) -> Self::FloatVec {
            _mm256_add_ps(a, b)
        }

        #[target_feature(enable = "avx")]
        unsafe fn sub(a: Self::FloatVec, b: Self::FloatVec) -> Self::FloatVec {
            _mm256_sub_ps(a, b)
        }

        #[target_feature(enable = "avx")]
        unsafe fn mul(a: Self::FloatVec, b: Self::FloatVec) -> Self::FloatVec {
            _mm256_mul_ps(a, b)
        }

        #[target_feature(enable = "avx")]
        unsafe fn div(a: Self::FloatVec, b: Self::FloatVec) -> Self::FloatVec {
            _mm256_div_ps(a, b)
        }

        #[target_feature(enable = "avx")]
        unsafe fn blend(
            a: Self::FloatVec,
            b: Self::FloatVec,
            mask: Self::MaskVec,
        ) -> Self::FloatVec {
            _mm256_blendv_ps(a, b, mask)
        }

        #[target_feature(enable = "avx")]
        unsafe fn element_shift(
            x: Self::FloatVec,
            shift_in: *const Self::Float,
            shift_out: &mut Self::Float,
        ) -> Self::FloatVec {
            /*
            let y: [f32; 8] = mem::transmute(x);
            *shift_out = y[7];
            */
            *shift_out = mem::transmute(_mm_extract_ps::<3>(_mm256_extractf128_ps::<1>(x)));
            Self::element_shift_in(x, shift_in)
        }

        #[target_feature(enable = "avx")]
        unsafe fn element_shift_in(
            x: Self::FloatVec,
            shift_in: *const Self::Float,
        ) -> Self::FloatVec {
            /*
            let x: [f32; 8] = mem::transmute(x);
            mem::transmute([*shift_in, x[0], x[1], x[2], x[3], x[4], x[5], x[6]]
            */

            // Rotate the lanes, then replace the lowest lanes.
            let x = _mm256_permute_ps::<0b10_01_00_11>(x);
            let mut x: [__m128; 2] = mem::transmute(x);
            x[1] = _mm_move_ss(x[1], x[0]);
            x[0] = _mm_move_ss(x[0], _mm_load_ss(shift_in));
            mem::transmute(x)

            /* TODO
             * The above code currently compiles into a permute + blend.
             * However, if we try to do that explicity (as shown below),
             * the compiler converts it into something slower.
             */
            /*
            let x = _mm256_permute_ps::<0b10_01_00_11>(x);
            let low = mem::transmute([_mm_load_ss(shift_in), _mm256_extractf128_ps::<0>(x)]);
            _mm256_blend_ps::<0x11>(x, low)
            */
        }

        fn mask_from_array(a: Self::MaskArray) -> Self::MaskVec {
            unsafe { mem::transmute(a) }
        }

        #[target_feature(enable = "avx")]
        unsafe fn mask_shift(x: Self::MaskVec) -> Self::MaskVec {
            // TODO: _mm256_add_epi64(x, x) is a small improvement but requires avx2
            let mut x: [__m128i; 2] = mem::transmute(x);
            x[0] = _mm_slli_epi64::<1>(x[0]);
            x[1] = _mm_slli_epi64::<1>(x[1]);
            mem::transmute(x)
        }
    }

    pub struct AvxF64x4;

    impl Vector for AvxF64x4 {
        type Float = f64;
        type FloatArray = [f64; 4];
        type FloatVec = __m256d;
        type Mask = u64;
        type MaskArray = [u64; 4];
        type MaskVec = __m256d;
        type IndexArray = [u8; 4];
        const MASK_BITS: usize = 64;
        const LANES: usize = 4;

        type Mode = u32;

        unsafe fn set_flush_zero_mode() -> Self::Mode {
            let mode = _MM_GET_FLUSH_ZERO_MODE();
            _MM_SET_FLUSH_ZERO_MODE(0x8000);
            mode
        }

        unsafe fn restore_flush_zero_mode(mode: Self::Mode) {
            _MM_SET_FLUSH_ZERO_MODE(mode);
        }

        #[target_feature(enable = "avx")]
        unsafe fn zero() -> Self::FloatVec {
            _mm256_setzero_pd()
        }

        #[target_feature(enable = "avx")]
        unsafe fn first_element(f: Self::Float) -> Self::FloatVec {
            _mm256_set_pd(0., 0., 0., f)
        }

        #[target_feature(enable = "avx")]
        unsafe fn splat(f: Self::Float) -> Self::FloatVec {
            _mm256_set1_pd(f)
        }

        fn from_array(a: Self::FloatArray) -> Self::FloatVec {
            unsafe { mem::transmute(a) }
        }

        fn to_array(a: Self::FloatVec) -> Self::FloatArray {
            unsafe { mem::transmute(a) }
        }

        #[target_feature(enable = "avx")]
        unsafe fn add(a: Self::FloatVec, b: Self::FloatVec) -> Self::FloatVec {
            _mm256_add_pd(a, b)
        }

        #[target_feature(enable = "avx")]
        unsafe fn sub(a: Self::FloatVec, b: Self::FloatVec) -> Self::FloatVec {
            _mm256_sub_pd(a, b)
        }

        #[target_feature(enable = "avx")]
        unsafe fn mul(a: Self::FloatVec, b: Self::FloatVec) -> Self::FloatVec {
            _mm256_mul_pd(a, b)
        }

        #[target_feature(enable = "avx")]
        unsafe fn div(a: Self::FloatVec, b: Self::FloatVec) -> Self::FloatVec {
            _mm256_div_pd(a, b)
        }

        #[target_feature(enable = "avx")]
        unsafe fn blend(
            a: Self::FloatVec,
            b: Self::FloatVec,
            mask: Self::MaskVec,
        ) -> Self::FloatVec {
            _mm256_blendv_pd(a, b, mask)
        }

        #[target_feature(enable = "avx")]
        unsafe fn element_shift(
            x: Self::FloatVec,
            shift_in: *const Self::Float,
            shift_out: &mut Self::Float,
        ) -> Self::FloatVec {
            /*
            let y: [f64; 4] = mem::transmute(x);
            *shift_out = y[3];
            */
            *shift_out = mem::transmute(_mm_extract_epi64::<1>(mem::transmute(
                _mm256_extractf128_pd::<1>(x),
            )));
            Self::element_shift_in(x, shift_in)
        }

        #[target_feature(enable = "avx")]
        unsafe fn element_shift_in(
            x: Self::FloatVec,
            shift_in: *const Self::Float,
        ) -> Self::FloatVec {
            /*
            let x: [f64; 4] = mem::transmute(x);
            let x: [f64; 4] = [*shift_in, x[0], x[1], x[2]];
            mem::transmute(x)
            */

            // Rotate the lanes, then replace the lowest lanes.
            let x = _mm256_permute_pd::<0b01_01>(x);
            let mut x: [__m128d; 2] = mem::transmute(x);
            x[1] = _mm_move_sd(x[1], x[0]);
            x[0] = _mm_move_sd(x[0], _mm_load_sd(shift_in));
            mem::transmute(x)

            /* Blend is slower in this case:
            let x = _mm256_permute_pd::<0b01_01>(x);
            let low = mem::transmute([_mm256_extractf128_pd::<0>(x), _mm_load_sd(shift_in)]);
            _mm256_blend_pd::<0b01_01>(x, low)
            */
        }

        fn mask_from_array(a: Self::MaskArray) -> Self::MaskVec {
            unsafe { mem::transmute(a) }
        }

        #[target_feature(enable = "avx")]
        unsafe fn mask_shift(x: Self::MaskVec) -> Self::MaskVec {
            let mut x: [__m128i; 2] = mem::transmute(x);
            x[0] = _mm_slli_epi64::<1>(x[0]);
            x[1] = _mm_slli_epi64::<1>(x[1]);
            mem::transmute(x)
        }
    }
}
#[cfg(all(target_arch = "x86_64", not(feature = "c-avx")))]
pub(crate) use x86_64_avx::*;

#[cfg(all(target_arch = "x86_64", not(feature = "c-avx512"), feature = "nightly"))]
mod x86_64_avx512 {
    use super::Vector;
    use std::arch::x86_64::*;
    use std::mem;

    #[cfg(feature = "nightly")]
    pub struct AvxF32x16;

    #[cfg(feature = "nightly")]
    impl Vector for AvxF32x16 {
        type Float = f32;
        type FloatArray = [f32; 16];
        type FloatVec = __m512;
        type Mask = u32;
        type MaskArray = [u32; 16];
        type MaskVec = __m512i;
        type IndexArray = [u8; 16];
        const MASK_BITS: usize = 32;
        const LANES: usize = 16;

        type Mode = u32;

        unsafe fn set_flush_zero_mode() -> Self::Mode {
            let mode = _MM_GET_FLUSH_ZERO_MODE();
            _MM_SET_FLUSH_ZERO_MODE(0x8000);
            mode
        }

        unsafe fn restore_flush_zero_mode(mode: Self::Mode) {
            _MM_SET_FLUSH_ZERO_MODE(mode);
        }

        #[target_feature(enable = "avx512f")]
        unsafe fn zero() -> Self::FloatVec {
            _mm512_setzero_ps()
        }

        #[target_feature(enable = "avx512f")]
        unsafe fn first_element(f: Self::Float) -> Self::FloatVec {
            _mm512_set_ps(
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., f,
            )
        }

        #[target_feature(enable = "avx512f")]
        unsafe fn splat(f: Self::Float) -> Self::FloatVec {
            _mm512_set1_ps(f)
        }

        fn from_array(a: Self::FloatArray) -> Self::FloatVec {
            unsafe { mem::transmute(a) }
        }

        fn to_array(a: Self::FloatVec) -> Self::FloatArray {
            unsafe { mem::transmute(a) }
        }

        #[target_feature(enable = "avx512f")]
        unsafe fn add(a: Self::FloatVec, b: Self::FloatVec) -> Self::FloatVec {
            _mm512_add_ps(a, b)
        }

        #[target_feature(enable = "avx512f")]
        unsafe fn sub(a: Self::FloatVec, b: Self::FloatVec) -> Self::FloatVec {
            _mm512_sub_ps(a, b)
        }

        #[target_feature(enable = "avx512f")]
        unsafe fn mul(a: Self::FloatVec, b: Self::FloatVec) -> Self::FloatVec {
            _mm512_mul_ps(a, b)
        }

        #[target_feature(enable = "avx512f")]
        unsafe fn div(a: Self::FloatVec, b: Self::FloatVec) -> Self::FloatVec {
            _mm512_div_ps(a, b)
        }

        #[target_feature(enable = "avx512f")]
        unsafe fn blend(
            a: Self::FloatVec,
            b: Self::FloatVec,
            mask: Self::MaskVec,
        ) -> Self::FloatVec {
            let bit = _mm512_set1_epi32(1 << 31);
            let mask = _mm512_test_epi32_mask(mask, bit);
            _mm512_mask_blend_ps(mask, a, b)
        }

        #[target_feature(enable = "avx512f")]
        unsafe fn element_shift(
            x: Self::FloatVec,
            shift_in: *const Self::Float,
            shift_out: &mut Self::Float,
        ) -> Self::FloatVec {
            /*
            let y: [f32; 16] = mem::transmute(x);
            *shift_out = y[15];
            */
            *shift_out = mem::transmute(_mm_extract_ps::<3>(_mm512_extractf32x4_ps::<3>(x)));
            Self::element_shift_in(x, shift_in)
        }

        #[target_feature(enable = "avx512f")]
        unsafe fn element_shift_in(
            x: Self::FloatVec,
            shift_in: *const Self::Float,
        ) -> Self::FloatVec {
            /*
            let x: [f32; 16] = mem::transmute(x);
            let x: [f32; 16] = [*shift_in, x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10], x[11], x[12], x[13], x[14]];
            mem::transmute(x)
            */
            let shift_in = _mm512_castps128_ps512(_mm_load_ss(shift_in));
            let index = mem::transmute([16u32, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]);
            _mm512_mask_permutex2var_ps(x, !0, index, shift_in)
        }

        fn mask_from_array(a: Self::MaskArray) -> Self::MaskVec {
            unsafe { mem::transmute(a) }
        }

        #[target_feature(enable = "avx512f")]
        unsafe fn mask_shift(x: Self::MaskVec) -> Self::MaskVec {
            _mm512_slli_epi32::<1>(x)
        }
    }

    #[cfg(feature = "nightly")]
    pub struct AvxF64x8;

    #[cfg(feature = "nightly")]
    impl Vector for AvxF64x8 {
        type Float = f64;
        type FloatArray = [f64; 8];
        type FloatVec = __m512d;
        type Mask = u64;
        type MaskArray = [u64; 8];
        type MaskVec = __m512i;
        type IndexArray = [u8; 8];
        const MASK_BITS: usize = 64;
        const LANES: usize = 8;

        type Mode = u32;

        unsafe fn set_flush_zero_mode() -> Self::Mode {
            let mode = _MM_GET_FLUSH_ZERO_MODE();
            _MM_SET_FLUSH_ZERO_MODE(0x8000);
            mode
        }

        unsafe fn restore_flush_zero_mode(mode: Self::Mode) {
            _MM_SET_FLUSH_ZERO_MODE(mode);
        }

        #[target_feature(enable = "avx512f")]
        unsafe fn zero() -> Self::FloatVec {
            _mm512_setzero_pd()
        }

        #[target_feature(enable = "avx512f")]
        unsafe fn first_element(f: Self::Float) -> Self::FloatVec {
            _mm512_set_pd(0., 0., 0., 0., 0., 0., 0., f)
        }

        #[target_feature(enable = "avx512f")]
        unsafe fn splat(f: Self::Float) -> Self::FloatVec {
            _mm512_set1_pd(f)
        }

        fn from_array(a: Self::FloatArray) -> Self::FloatVec {
            unsafe { mem::transmute(a) }
        }

        fn to_array(a: Self::FloatVec) -> Self::FloatArray {
            unsafe { mem::transmute(a) }
        }

        #[target_feature(enable = "avx512f")]
        unsafe fn add(a: Self::FloatVec, b: Self::FloatVec) -> Self::FloatVec {
            _mm512_add_pd(a, b)
        }

        #[target_feature(enable = "avx512f")]
        unsafe fn sub(a: Self::FloatVec, b: Self::FloatVec) -> Self::FloatVec {
            _mm512_sub_pd(a, b)
        }

        #[target_feature(enable = "avx512f")]
        unsafe fn mul(a: Self::FloatVec, b: Self::FloatVec) -> Self::FloatVec {
            _mm512_mul_pd(a, b)
        }

        #[target_feature(enable = "avx512f")]
        unsafe fn div(a: Self::FloatVec, b: Self::FloatVec) -> Self::FloatVec {
            _mm512_div_pd(a, b)
        }

        #[target_feature(enable = "avx512f")]
        unsafe fn blend(
            a: Self::FloatVec,
            b: Self::FloatVec,
            mask: Self::MaskVec,
        ) -> Self::FloatVec {
            let bit = _mm512_set1_epi64(1 << 63);
            let mask = _mm512_test_epi64_mask(mask, bit);
            _mm512_mask_blend_pd(mask, a, b)
        }

        #[target_feature(enable = "avx512f")]
        unsafe fn element_shift(
            x: Self::FloatVec,
            shift_in: *const Self::Float,
            shift_out: &mut Self::Float,
        ) -> Self::FloatVec {
            /*
            let y: [f64; 8] = mem::transmute(x);
            *shift_out = y[7];
            */
            // TODO: missing _mm512_extractf64x2_pd?
            _mm_storeh_pd(
                shift_out,
                mem::transmute(_mm512_extractf32x4_ps::<3>(mem::transmute(x))),
            );
            Self::element_shift_in(x, shift_in)
        }

        #[target_feature(enable = "avx512f")]
        unsafe fn element_shift_in(
            x: Self::FloatVec,
            shift_in: *const Self::Float,
        ) -> Self::FloatVec {
            /*
            let x: [f64; 8] = mem::transmute(x);
            let x: [f64; 8] = [*shift_in, x[0], x[1], x[2], x[3], x[4], x[5], x[6]];
            mem::transmute(x)
            */
            let shift_in = _mm512_castpd128_pd512(_mm_load_sd(shift_in));
            let index = mem::transmute([8u64, 0, 1, 2, 3, 4, 5, 6]);
            _mm512_mask_permutex2var_pd(x, !0, index, shift_in)
        }

        fn mask_from_array(a: Self::MaskArray) -> Self::MaskVec {
            unsafe { mem::transmute(a) }
        }

        #[target_feature(enable = "avx512f")]
        unsafe fn mask_shift(x: Self::MaskVec) -> Self::MaskVec {
            _mm512_slli_epi64::<1>(x)
        }
    }
}
#[cfg(all(target_arch = "x86_64", not(feature = "c-avx512"), feature = "nightly"))]
pub(crate) use x86_64_avx512::*;
