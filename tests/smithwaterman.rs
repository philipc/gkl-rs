use gkl::smithwaterman::{OverhangStrategy, Parameters};

fn test_one(
    ref_array: &[u8],
    alt_array: &[u8],
    parameters: Parameters,
    overhang: OverhangStrategy,
    expected: &[u8],
    expected_offset: usize,
) {
    if let Some(f) = gkl::smithwaterman::align_avx2() {
        let (cigar, offset) = f(ref_array, alt_array, parameters, overhang).unwrap();
        assert_eq!(cigar, expected);
        assert_eq!(offset, expected_offset);
    }

    if let Some(f) = gkl::smithwaterman::align_avx512() {
        let (cigar, offset) = f(ref_array, alt_array, parameters, overhang).unwrap();
        assert_eq!(cigar, expected);
        assert_eq!(offset, expected_offset);
    }

    if let Some(f) = gkl::smithwaterman::align() {
        let (cigar, offset) = f(ref_array, alt_array, parameters, overhang).unwrap();
        assert_eq!(cigar, expected);
        assert_eq!(offset, expected_offset);
    }
}

#[test]
fn single_element() {
    test_one(
        b"C",
        b"C",
        Parameters::new(3, -2, -2, -1),
        OverhangStrategy::Ignore,
        b"1M",
        0,
    );
}

#[test]
fn two_element() {
    test_one(
        b"AD",
        b"AT",
        Parameters::new(3, -5, -2, -1),
        OverhangStrategy::Ignore,
        b"1M1I",
        0,
    );
}
