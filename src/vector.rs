use std::ops;

/// A target specific implementation of vector operations.
pub(crate) trait Vector: Copy {
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
    fn set_flush_zero_mode(self) -> Self::Mode;
    fn restore_flush_zero_mode(self, mode: Self::Mode);

    // Vector with all elements set to 0.
    fn zero(self) -> Self::FloatVec;

    // Vector with given first element, and remaining elements set to 0.
    fn first_element(self, f: Self::Float) -> Self::FloatVec;

    // Vector with all elements set to given value.
    fn splat(self, f: Self::Float) -> Self::FloatVec;

    fn from_array(self, a: Self::FloatArray) -> Self::FloatVec;
    fn to_array(self, a: Self::FloatVec) -> Self::FloatArray;

    fn add(self, a: Self::FloatVec, b: Self::FloatVec) -> Self::FloatVec;
    fn sub(self, a: Self::FloatVec, b: Self::FloatVec) -> Self::FloatVec;
    fn mul(self, a: Self::FloatVec, b: Self::FloatVec) -> Self::FloatVec;
    fn div(self, a: Self::FloatVec, b: Self::FloatVec) -> Self::FloatVec;
    fn blend(self, a: Self::FloatVec, b: Self::FloatVec, mask: Self::MaskVec) -> Self::FloatVec;

    // Shift out the previous last element.
    fn element_shift_out(self, x: Self::FloatVec, shift_out: &mut Self::Float);

    // Shift in a new first element.
    fn element_shift_in(self, x: Self::FloatVec, shift_in: &Self::Float) -> Self::FloatVec;

    fn mask_from_array(self, a: Self::MaskArray) -> Self::MaskVec;
    fn mask_shift(self, x: Self::MaskVec) -> Self::MaskVec;
}

#[derive(Clone, Copy)]
pub(crate) struct F32x1;

impl Vector for F32x1 {
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

    #[inline]
    fn set_flush_zero_mode(self) -> Self::Mode {
        ()
    }

    #[inline]
    fn restore_flush_zero_mode(self, _mode: Self::Mode) {}

    #[inline]
    fn zero(self) -> Self::FloatVec {
        0.0
    }

    #[inline]
    fn first_element(self, f: Self::Float) -> Self::FloatVec {
        f
    }

    #[inline]
    fn splat(self, f: Self::Float) -> Self::FloatVec {
        f
    }

    #[inline]
    fn from_array(self, a: Self::FloatArray) -> Self::FloatVec {
        a[0]
    }

    #[inline]
    fn to_array(self, a: Self::FloatVec) -> Self::FloatArray {
        [a]
    }

    #[inline]
    fn add(self, a: Self::FloatVec, b: Self::FloatVec) -> Self::FloatVec {
        a + b
    }

    #[inline]
    fn sub(self, a: Self::FloatVec, b: Self::FloatVec) -> Self::FloatVec {
        a - b
    }

    #[inline]
    fn mul(self, a: Self::FloatVec, b: Self::FloatVec) -> Self::FloatVec {
        a * b
    }

    #[inline]
    fn div(self, a: Self::FloatVec, b: Self::FloatVec) -> Self::FloatVec {
        a / b
    }

    #[inline]
    fn blend(self, a: Self::FloatVec, b: Self::FloatVec, mask: Self::MaskVec) -> Self::FloatVec {
        if mask & (1 << 31) == 0 {
            a
        } else {
            b
        }
    }

    #[inline]
    fn element_shift_out(self, x: Self::FloatVec, shift_out: &mut Self::Float) {
        *shift_out = x;
    }

    #[inline]
    fn element_shift_in(self, _x: Self::FloatVec, shift_in: &Self::Float) -> Self::FloatVec {
        *shift_in
    }

    #[inline]
    fn mask_from_array(self, a: Self::MaskArray) -> Self::MaskVec {
        a[0]
    }

    #[inline]
    fn mask_shift(self, x: Self::MaskVec) -> Self::MaskVec {
        x << 1
    }
}

#[derive(Clone, Copy)]
pub(crate) struct F64x1;

impl Vector for F64x1 {
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

    #[inline]
    fn set_flush_zero_mode(self) -> Self::Mode {
        ()
    }

    #[inline]
    fn restore_flush_zero_mode(self, _mode: Self::Mode) {}

    #[inline]
    fn zero(self) -> Self::FloatVec {
        0.0
    }

    #[inline]
    fn first_element(self, f: Self::Float) -> Self::FloatVec {
        f
    }

    #[inline]
    fn splat(self, f: Self::Float) -> Self::FloatVec {
        f
    }

    #[inline]
    fn from_array(self, a: Self::FloatArray) -> Self::FloatVec {
        a[0]
    }

    #[inline]
    fn to_array(self, a: Self::FloatVec) -> Self::FloatArray {
        [a]
    }

    #[inline]
    fn add(self, a: Self::FloatVec, b: Self::FloatVec) -> Self::FloatVec {
        a + b
    }

    #[inline]
    fn sub(self, a: Self::FloatVec, b: Self::FloatVec) -> Self::FloatVec {
        a - b
    }

    #[inline]
    fn mul(self, a: Self::FloatVec, b: Self::FloatVec) -> Self::FloatVec {
        a * b
    }

    #[inline]
    fn div(self, a: Self::FloatVec, b: Self::FloatVec) -> Self::FloatVec {
        a / b
    }

    #[inline]
    fn blend(self, a: Self::FloatVec, b: Self::FloatVec, mask: Self::MaskVec) -> Self::FloatVec {
        if mask & (1 << 31) == 0 {
            a
        } else {
            b
        }
    }

    #[inline]
    fn element_shift_out(self, x: Self::FloatVec, shift_out: &mut Self::Float) {
        *shift_out = x;
    }

    #[inline]
    fn element_shift_in(self, _x: Self::FloatVec, shift_in: &Self::Float) -> Self::FloatVec {
        *shift_in
    }

    #[inline]
    fn mask_from_array(self, a: Self::MaskArray) -> Self::MaskVec {
        a[0]
    }

    #[inline]
    fn mask_shift(self, x: Self::MaskVec) -> Self::MaskVec {
        x << 1
    }
}

#[cfg(all(target_arch = "x86_64", not(feature = "c-avx")))]
mod x86_64_avx {
    use super::Vector;
    use std::arch::x86_64::*;
    use std::mem;

    #[derive(Clone, Copy)]
    pub struct AvxF32x8(());

    impl AvxF32x8 {
        #[inline]
        pub fn new() -> Option<Self> {
            if is_x86_feature_detected!("avx") {
                Some(AvxF32x8(()))
            } else {
                None
            }
        }

        #[inline]
        pub unsafe fn new_unchecked() -> Self {
            AvxF32x8(())
        }
    }

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

        #[inline]
        fn set_flush_zero_mode(self) -> Self::Mode {
            unsafe {
                let mode = _MM_GET_FLUSH_ZERO_MODE();
                _MM_SET_FLUSH_ZERO_MODE(0x8000);
                mode
            }
        }

        #[inline]
        fn restore_flush_zero_mode(self, mode: Self::Mode) {
            unsafe {
                _MM_SET_FLUSH_ZERO_MODE(mode);
            }
        }

        #[inline]
        fn zero(self) -> Self::FloatVec {
            unsafe { _mm256_setzero_ps() }
        }

        #[inline]
        fn first_element(self, f: Self::Float) -> Self::FloatVec {
            unsafe { _mm256_set_ps(0., 0., 0., 0., 0., 0., 0., f) }
        }

        #[inline]
        fn splat(self, f: Self::Float) -> Self::FloatVec {
            unsafe { _mm256_set1_ps(f) }
        }

        #[inline]
        fn from_array(self, a: Self::FloatArray) -> Self::FloatVec {
            unsafe { mem::transmute(a) }
        }

        #[inline]
        fn to_array(self, a: Self::FloatVec) -> Self::FloatArray {
            unsafe { mem::transmute(a) }
        }

        #[inline]
        fn add(self, a: Self::FloatVec, b: Self::FloatVec) -> Self::FloatVec {
            unsafe { _mm256_add_ps(a, b) }
        }

        #[inline]
        fn sub(self, a: Self::FloatVec, b: Self::FloatVec) -> Self::FloatVec {
            unsafe { _mm256_sub_ps(a, b) }
        }

        #[inline]
        fn mul(self, a: Self::FloatVec, b: Self::FloatVec) -> Self::FloatVec {
            unsafe { _mm256_mul_ps(a, b) }
        }

        #[inline]
        fn div(self, a: Self::FloatVec, b: Self::FloatVec) -> Self::FloatVec {
            unsafe { _mm256_div_ps(a, b) }
        }

        #[inline]
        fn blend(
            self,
            a: Self::FloatVec,
            b: Self::FloatVec,
            mask: Self::MaskVec,
        ) -> Self::FloatVec {
            unsafe { _mm256_blendv_ps(a, b, mask) }
        }

        #[inline]
        fn element_shift_out(self, x: Self::FloatVec, shift_out: &mut Self::Float) {
            /*
            let y: [f32; 8] = mem::transmute(x);
            *shift_out = y[7];
            */
            unsafe {
                *shift_out = mem::transmute(_mm_extract_ps::<3>(_mm256_extractf128_ps::<1>(x)));
            }
        }

        #[inline]
        fn element_shift_in(self, x: Self::FloatVec, shift_in: &Self::Float) -> Self::FloatVec {
            /*
            let x: [f32; 8] = mem::transmute(x);
            mem::transmute([*shift_in, x[0], x[1], x[2], x[3], x[4], x[5], x[6]]
            */

            // Rotate the lanes, then replace the lowest lanes.
            unsafe {
                let x = _mm256_permute_ps::<0b10_01_00_11>(x);
                let mut x: [__m128; 2] = mem::transmute(x);
                x[1] = _mm_move_ss(x[1], x[0]);
                x[0] = _mm_move_ss(x[0], _mm_load_ss(shift_in));
                mem::transmute(x)
            }

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

        #[inline]
        fn mask_from_array(self, a: Self::MaskArray) -> Self::MaskVec {
            unsafe { mem::transmute(a) }
        }

        #[inline]
        fn mask_shift(self, x: Self::MaskVec) -> Self::MaskVec {
            // TODO: _mm256_add_epi64(x, x) is a small improvement but requires avx2
            unsafe {
                let mut x: [__m128i; 2] = mem::transmute(x);
                x[0] = _mm_slli_epi64::<1>(x[0]);
                x[1] = _mm_slli_epi64::<1>(x[1]);
                mem::transmute(x)
            }
        }
    }

    #[derive(Clone, Copy)]
    pub struct AvxF64x4(());

    impl AvxF64x4 {
        #[inline]
        pub fn new() -> Option<Self> {
            if is_x86_feature_detected!("avx") {
                Some(AvxF64x4(()))
            } else {
                None
            }
        }

        #[inline]
        pub fn new_unchecked() -> Self {
            AvxF64x4(())
        }
    }

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

        #[inline]
        fn set_flush_zero_mode(self) -> Self::Mode {
            unsafe {
                let mode = _MM_GET_FLUSH_ZERO_MODE();
                _MM_SET_FLUSH_ZERO_MODE(0x8000);
                mode
            }
        }

        #[inline]
        fn restore_flush_zero_mode(self, mode: Self::Mode) {
            unsafe {
                _MM_SET_FLUSH_ZERO_MODE(mode);
            }
        }

        #[inline]
        fn zero(self) -> Self::FloatVec {
            unsafe { _mm256_setzero_pd() }
        }

        #[inline]
        fn first_element(self, f: Self::Float) -> Self::FloatVec {
            unsafe { _mm256_set_pd(0., 0., 0., f) }
        }

        #[inline]
        fn splat(self, f: Self::Float) -> Self::FloatVec {
            unsafe { _mm256_set1_pd(f) }
        }

        #[inline]
        fn from_array(self, a: Self::FloatArray) -> Self::FloatVec {
            unsafe { mem::transmute(a) }
        }

        #[inline]
        fn to_array(self, a: Self::FloatVec) -> Self::FloatArray {
            unsafe { mem::transmute(a) }
        }

        #[inline]
        fn add(self, a: Self::FloatVec, b: Self::FloatVec) -> Self::FloatVec {
            unsafe { _mm256_add_pd(a, b) }
        }

        #[inline]
        fn sub(self, a: Self::FloatVec, b: Self::FloatVec) -> Self::FloatVec {
            unsafe { _mm256_sub_pd(a, b) }
        }

        #[inline]
        fn mul(self, a: Self::FloatVec, b: Self::FloatVec) -> Self::FloatVec {
            unsafe { _mm256_mul_pd(a, b) }
        }

        #[inline]
        fn div(self, a: Self::FloatVec, b: Self::FloatVec) -> Self::FloatVec {
            unsafe { _mm256_div_pd(a, b) }
        }

        #[inline]
        fn blend(
            self,
            a: Self::FloatVec,
            b: Self::FloatVec,
            mask: Self::MaskVec,
        ) -> Self::FloatVec {
            unsafe { _mm256_blendv_pd(a, b, mask) }
        }

        #[inline]
        fn element_shift_out(self, x: Self::FloatVec, shift_out: &mut Self::Float) {
            /*
            let y: [f64; 4] = mem::transmute(x);
            *shift_out = y[3];
            */
            unsafe {
                *shift_out = mem::transmute(_mm_extract_epi64::<1>(mem::transmute(
                    _mm256_extractf128_pd::<1>(x),
                )));
            }
        }

        #[inline]
        fn element_shift_in(self, x: Self::FloatVec, shift_in: &Self::Float) -> Self::FloatVec {
            /*
            let x: [f64; 4] = mem::transmute(x);
            let x: [f64; 4] = [*shift_in, x[0], x[1], x[2]];
            mem::transmute(x)
            */

            // Rotate the lanes, then replace the lowest lanes.
            unsafe {
                let x = _mm256_permute_pd::<0b01_01>(x);
                let mut x: [__m128d; 2] = mem::transmute(x);
                x[1] = _mm_move_sd(x[1], x[0]);
                x[0] = _mm_move_sd(x[0], _mm_load_sd(shift_in));
                mem::transmute(x)
            }

            /* Blend is slower in this case:
            let x = _mm256_permute_pd::<0b01_01>(x);
            let low = mem::transmute([_mm256_extractf128_pd::<0>(x), _mm_load_sd(shift_in)]);
            _mm256_blend_pd::<0b01_01>(x, low)
            */
        }

        #[inline]
        fn mask_from_array(self, a: Self::MaskArray) -> Self::MaskVec {
            unsafe { mem::transmute(a) }
        }

        #[inline]
        fn mask_shift(self, x: Self::MaskVec) -> Self::MaskVec {
            unsafe {
                let mut x: [__m128i; 2] = mem::transmute(x);
                x[0] = _mm_slli_epi64::<1>(x[0]);
                x[1] = _mm_slli_epi64::<1>(x[1]);
                mem::transmute(x)
            }
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

    #[derive(Clone, Copy)]
    pub struct AvxF32x16(());

    impl AvxF32x16 {
        #[inline]
        pub fn new() -> Option<Self> {
            if is_x86_feature_detected!("avx512f") {
                Some(AvxF32x16(()))
            } else {
                None
            }
        }

        #[inline]
        pub fn new_unchecked() -> Self {
            AvxF32x16(())
        }
    }

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

        #[inline]
        fn set_flush_zero_mode(self) -> Self::Mode {
            unsafe {
                let mode = _MM_GET_FLUSH_ZERO_MODE();
                _MM_SET_FLUSH_ZERO_MODE(0x8000);
                mode
            }
        }

        #[inline]
        fn restore_flush_zero_mode(self, mode: Self::Mode) {
            unsafe {
                _MM_SET_FLUSH_ZERO_MODE(mode);
            }
        }

        #[inline]
        fn zero(self) -> Self::FloatVec {
            unsafe { _mm512_setzero_ps() }
        }

        #[inline]
        fn first_element(self, f: Self::Float) -> Self::FloatVec {
            unsafe {
                _mm512_set_ps(
                    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., f,
                )
            }
        }

        #[inline]
        fn splat(self, f: Self::Float) -> Self::FloatVec {
            unsafe { _mm512_set1_ps(f) }
        }

        #[inline]
        fn from_array(self, a: Self::FloatArray) -> Self::FloatVec {
            unsafe { mem::transmute(a) }
        }

        #[inline]
        fn to_array(self, a: Self::FloatVec) -> Self::FloatArray {
            unsafe { mem::transmute(a) }
        }

        #[inline]
        fn add(self, a: Self::FloatVec, b: Self::FloatVec) -> Self::FloatVec {
            unsafe { _mm512_add_ps(a, b) }
        }

        #[inline]
        fn sub(self, a: Self::FloatVec, b: Self::FloatVec) -> Self::FloatVec {
            unsafe { _mm512_sub_ps(a, b) }
        }

        #[inline]
        fn mul(self, a: Self::FloatVec, b: Self::FloatVec) -> Self::FloatVec {
            unsafe { _mm512_mul_ps(a, b) }
        }

        #[inline]
        fn div(self, a: Self::FloatVec, b: Self::FloatVec) -> Self::FloatVec {
            unsafe { _mm512_div_ps(a, b) }
        }

        #[inline]
        fn blend(
            self,
            a: Self::FloatVec,
            b: Self::FloatVec,
            mask: Self::MaskVec,
        ) -> Self::FloatVec {
            unsafe {
                let bit = _mm512_set1_epi32(1 << 31);
                let mask = _mm512_test_epi32_mask(mask, bit);
                _mm512_mask_blend_ps(mask, a, b)
            }
        }

        #[inline]
        fn element_shift_out(self, x: Self::FloatVec, shift_out: &mut Self::Float) {
            /*
            let y: [f32; 16] = mem::transmute(x);
            *shift_out = y[15];
            */
            unsafe {
                *shift_out = mem::transmute(_mm_extract_ps::<3>(_mm512_extractf32x4_ps::<3>(x)));
            }
        }

        #[inline]
        fn element_shift_in(self, x: Self::FloatVec, shift_in: &Self::Float) -> Self::FloatVec {
            /*
            let x: [f32; 16] = mem::transmute(x);
            let x: [f32; 16] = [*shift_in, x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10], x[11], x[12], x[13], x[14]];
            mem::transmute(x)
            */
            unsafe {
                let shift_in = _mm512_castps128_ps512(_mm_load_ss(shift_in));
                let index =
                    mem::transmute([16u32, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]);
                _mm512_mask_permutex2var_ps(x, !0, index, shift_in)
            }
        }

        #[inline]
        fn mask_from_array(self, a: Self::MaskArray) -> Self::MaskVec {
            unsafe { mem::transmute(a) }
        }

        #[inline]
        fn mask_shift(self, x: Self::MaskVec) -> Self::MaskVec {
            unsafe { _mm512_slli_epi32::<1>(x) }
        }
    }

    #[derive(Clone, Copy)]
    pub struct AvxF64x8(());

    impl AvxF64x8 {
        #[inline]
        pub fn new() -> Option<Self> {
            if is_x86_feature_detected!("avx512f") {
                Some(AvxF64x8(()))
            } else {
                None
            }
        }

        #[inline]
        pub fn new_unchecked() -> Self {
            AvxF64x8(())
        }
    }

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

        #[inline]
        fn set_flush_zero_mode(self) -> Self::Mode {
            unsafe {
                let mode = _MM_GET_FLUSH_ZERO_MODE();
                _MM_SET_FLUSH_ZERO_MODE(0x8000);
                mode
            }
        }

        #[inline]
        fn restore_flush_zero_mode(self, mode: Self::Mode) {
            unsafe {
                _MM_SET_FLUSH_ZERO_MODE(mode);
            }
        }

        #[inline]
        fn zero(self) -> Self::FloatVec {
            unsafe { _mm512_setzero_pd() }
        }

        #[inline]
        fn first_element(self, f: Self::Float) -> Self::FloatVec {
            unsafe { _mm512_set_pd(0., 0., 0., 0., 0., 0., 0., f) }
        }

        #[inline]
        fn splat(self, f: Self::Float) -> Self::FloatVec {
            unsafe { _mm512_set1_pd(f) }
        }

        #[inline]
        fn from_array(self, a: Self::FloatArray) -> Self::FloatVec {
            unsafe { mem::transmute(a) }
        }

        #[inline]
        fn to_array(self, a: Self::FloatVec) -> Self::FloatArray {
            unsafe { mem::transmute(a) }
        }

        #[inline]
        fn add(self, a: Self::FloatVec, b: Self::FloatVec) -> Self::FloatVec {
            unsafe { _mm512_add_pd(a, b) }
        }

        #[inline]
        fn sub(self, a: Self::FloatVec, b: Self::FloatVec) -> Self::FloatVec {
            unsafe { _mm512_sub_pd(a, b) }
        }

        #[inline]
        fn mul(self, a: Self::FloatVec, b: Self::FloatVec) -> Self::FloatVec {
            unsafe { _mm512_mul_pd(a, b) }
        }

        #[inline]
        fn div(self, a: Self::FloatVec, b: Self::FloatVec) -> Self::FloatVec {
            unsafe { _mm512_div_pd(a, b) }
        }

        #[inline]
        fn blend(
            self,
            a: Self::FloatVec,
            b: Self::FloatVec,
            mask: Self::MaskVec,
        ) -> Self::FloatVec {
            unsafe {
                let bit = _mm512_set1_epi64(1 << 63);
                let mask = _mm512_test_epi64_mask(mask, bit);
                _mm512_mask_blend_pd(mask, a, b)
            }
        }

        #[inline]
        fn element_shift_out(self, x: Self::FloatVec, shift_out: &mut Self::Float) {
            /*
            let y: [f64; 8] = mem::transmute(x);
            *shift_out = y[7];
            */
            // TODO: missing _mm512_extractf64x2_pd?
            unsafe {
                _mm_storeh_pd(
                    shift_out,
                    mem::transmute(_mm512_extractf32x4_ps::<3>(mem::transmute(x))),
                );
            }
        }

        #[inline]
        fn element_shift_in(self, x: Self::FloatVec, shift_in: &Self::Float) -> Self::FloatVec {
            /*
            let x: [f64; 8] = mem::transmute(x);
            let x: [f64; 8] = [*shift_in, x[0], x[1], x[2], x[3], x[4], x[5], x[6]];
            mem::transmute(x)
            */
            unsafe {
                let shift_in = _mm512_castpd128_pd512(_mm_load_sd(shift_in));
                let index = mem::transmute([8u64, 0, 1, 2, 3, 4, 5, 6]);
                _mm512_mask_permutex2var_pd(x, !0, index, shift_in)
            }
        }

        #[inline]
        fn mask_from_array(self, a: Self::MaskArray) -> Self::MaskVec {
            unsafe { mem::transmute(a) }
        }

        #[inline]
        fn mask_shift(self, x: Self::MaskVec) -> Self::MaskVec {
            unsafe { _mm512_slli_epi64::<1>(x) }
        }
    }
}
#[cfg(all(target_arch = "x86_64", not(feature = "c-avx512"), feature = "nightly"))]
pub(crate) use x86_64_avx512::*;
