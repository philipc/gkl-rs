use gkl::smithwaterman::{Align, OverhangStrategy, Parameters};

fn test_one(
    ref_array: &[u8],
    alt_array: &[u8],
    parameters: Parameters,
    overhang: OverhangStrategy,
    expected_offset: usize,
    expected: &[u8],
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
        0,
        b"1M",
    );
}

#[test]
fn two_element() {
    test_one(
        b"AD",
        b"AT",
        Parameters::new(3, -5, -2, -1),
        OverhangStrategy::Ignore,
        0,
        b"1M1I",
    );
}

#[test]
fn read_aligned_to_ref_complex_alignment() {
    test_one(
        b"AAAGGACTGACTG",
        b"ACTGACTGACTG",
        Parameters::new(3, -1, -4, -3),
        OverhangStrategy::SoftClip,
        1,
        b"12M",
    );
}

#[test]
fn odd_no_alignment() {
    let ref1 = b"AAAGACTACTG";
    let read1 = b"AACGGACACTG";
    test_one(
        ref1,
        read1,
        Parameters::new(50, -100, -220, -12),
        OverhangStrategy::SoftClip,
        1,
        b"2M2I3M1D4M",
    );
    test_one(
        ref1,
        read1,
        Parameters::new(200, -50, -300, -22),
        OverhangStrategy::SoftClip,
        0,
        b"11M",
    );
}

#[test]
fn indels_at_start_and_end() {
    let m = &b"CCCCC"[..];
    let ref_array = &[&b"AAA"[..], m].concat();
    let alt_array = &[m, &b"GGG"[..]].concat();
    let parameters = Parameters::new(3, -1, -4, -3);
    test_one(
        ref_array,
        alt_array,
        parameters,
        OverhangStrategy::SoftClip,
        3,
        b"5M3S",
    );
}

#[test]
fn degenerate_alignment_with_indels_at_both_ends() {
    let ref_array = b"TGTGTGTGTGTGTGACAGAGAGAGAGAGAGAGAGAGAGAGAGAGA";
    let alt_array = b"ACAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGA";
    let parameters = Parameters::new(25, -50, -110, -6);
    test_one(
        ref_array,
        alt_array,
        parameters,
        OverhangStrategy::SoftClip,
        14,
        b"31M20S",
    );
}

// This test is designed to ensure that the indels are correctly placed
// if the region flanking these indels is extended by a varying amount.
// It checks for problems caused by floating point rounding leading to different
// paths being selected.
#[test]
fn identical_alignments_with_differing_flank_lengths() {
    // Create two versions of the same sequence with different flanking regions.
    let padded_ref = &b"GCGTCGCAGTCTTAAGGCCCCGCCTTTTCAGACAGCTTCCGCTGGGCCTGGGCCGCTGCGGGGCGGTCACGGCCCCTTTAAGCCTGAGCCCCGCCCCCTGGCTCCCCGCCCCCTCTTCTCCCCTCCCCCAAGCCAGCACCTGGTGCCCCGGCGGGTCGTGCGGCGCGGCGCTCCGCGGTGAGCGCCTGACCCCGAGGGGGCCCGGGGCCGCGTCCCTGGGCCCTCCCCACCCTTGCGGTGGCCTCGCGGGTCCCAGGGGCGGGGCTGGAGCGGCAGCAGGGCCGGGGAGATGGGCGGTGGGGAGCGCGGGAGGGACCGGGCCGAGCCGGGGGAAGGGCTCCGGTGACT"[..];
    let padded_hap = &b"GCGTCGCAGTCTTAAGGCCCCGCCTTTTCAGACAGCTTCCGCTGGGCCTGGGCCGCTGCGGGGCGGTCACGGCCCCTTTAAGCCTGAGCCCCGCCCCCTGGCTCCCCGCCCCCTCTTCTCCCCTCCCCCAAGCCAGCACCTGGTGCCCCGGCGGGTCGTGCGGCGCGGCGCTCCGCGGTGAGCGCCTGACCCCGAGGGCCGGGCCCTCCCCACCCTTGCGGTGGCCTCGCGGGTCCCAGGGGCGGGGCTGGAGCGGCAGCAGGGCCGGGGAGATGGGCGGTGGGGAGCGCGGGAGGGACCGGGCCGAGCCGGGGGAAGGGCTCCGGTGACT"[..];
    let not_padded_ref = &b"CTTTAAGCCTGAGCCCCGCCCCCTGGCTCCCCGCCCCCTCTTCTCCCCTCCCCCAAGCCAGCACCTGGTGCCCCGGCGGGTCGTGCGGCGCGGCGCTCCGCGGTGAGCGCCTGACCCCGAGGGGGCCCGGGGCCGCGTCCCTGGGCCCTCCCCACCCTTGCGGTGGCCTCGCGGGTCCCAGGGGCGGGGCTGGAGCGGCAGCAGGGCCGGGGAGATGGGCGGTGGGGAGCGCGGGAGGGA"[..];
    let not_padded_hap = &b"CTTTAAGCCTGAGCCCCGCCCCCTGGCTCCCCGCCCCCTCTTCTCCCCTCCCCCAAGCCAGCACCTGGTGCCCCGGCGGGTCGTGCGGCGCGGCGCTCCGCGGTGAGCGCCTGACCCCGAGGGCCGGGCCCTCCCCACCCTTGCGGTGGCCTCGCGGGTCCCAGGGGCGGGGCTGGAGCGGCAGCAGGGCCGGGGAGATGGGCGGTGGGGAGCGCGGGAGGGA"[..];

    // a simplified version of the getCigar routine in the haplotype caller to align these
    let sw_pad = b"NNNNNNNNNN";
    let paddeds_ref = &[sw_pad, padded_ref, sw_pad].concat();
    let paddeds_hap = &[sw_pad, padded_hap, sw_pad].concat();
    let not_paddeds_ref = &[sw_pad, not_padded_ref, sw_pad].concat();
    let not_paddeds_hap = &[sw_pad, not_padded_hap, sw_pad].concat();
    let parameters = Parameters::new(200, -150, -260, -11);
    let test_one = |f: Align| {
        let padded_alignment = f(
            paddeds_ref,
            paddeds_hap,
            parameters,
            OverhangStrategy::SoftClip,
        )
        .unwrap();
        let not_padded_alignment = f(
            not_paddeds_ref,
            not_paddeds_hap,
            parameters,
            OverhangStrategy::SoftClip,
        )
        .unwrap();
        // Now verify that the two sequences have the same alignment and not match positions.
        let padded_elements = padded_alignment
            .0
            .split_inclusive(|c| *c < b'0' || *c > b'9')
            .collect::<Vec<_>>();
        let not_padded_elements = not_padded_alignment
            .0
            .split_inclusive(|c| *c < b'0' || *c > b'9')
            .collect::<Vec<_>>();
        for (pc, npc) in padded_elements.iter().zip(not_padded_elements.iter()) {
            if pc.last().copied() != Some(b'M') || npc.last().copied() != Some(b'M') {
                assert_eq!(pc, npc);
            }
        }
    };
    if let Some(f) = gkl::smithwaterman::align_avx2() {
        test_one(f);
    }
    if let Some(f) = gkl::smithwaterman::align_avx512() {
        test_one(f);
    }
}

#[test]
fn substring_match() {
    let alt_array = &b"CCCCC"[..];
    let ref_array = &[&b"AAA"[..], alt_array].concat();
    let parameters = Parameters::new(3, -1, -4, -3);
    test_one(
        ref_array,
        alt_array,
        parameters,
        OverhangStrategy::SoftClip,
        3,
        b"5M",
    );
    test_one(
        ref_array,
        alt_array,
        parameters,
        OverhangStrategy::Indel,
        0,
        b"3D5M",
    );
    test_one(
        ref_array,
        alt_array,
        parameters,
        OverhangStrategy::LeadingIndel,
        0,
        b"3D5M",
    );
    test_one(
        ref_array,
        alt_array,
        parameters,
        OverhangStrategy::Ignore,
        3,
        b"5M",
    );
}

#[test]
fn substring_match_long() {
    let ref_array = b"ATAGAAAATAGTTTTTGGAAATATGGGTGAAGAGACATCTCCTCTTATGGAAAAAGGGATTCTAGAATTTAACAATAAATATTCCCAACTTTCCCCAAGGCTTTAAAATCTACCTTGAAGGAGCAGCTGATGTATTTCTAGAACAGACTTAGGTGTCTTGGTGTGGCCTGTAAAGAGATACTGTCTTTCTCTTTTGAGTGTAAGAGAGAAAGGACAGTCTACTCAATAAAGAGTGCTGGGAAAACTGAATATCCACACACAGAATAATAAAACTAGATCCTATCTCTCACCATATACAAAGATCAACTCAAAACAAATTAAAGACCTAAATGTAAGACAAGAAATTATAAAACTACTAGAAAAAAACACAAGGGAAATGCTTCAGGACATTGGC";
    let alt_array = b"AAAAAAA";
    let parameters = Parameters::new(3, -1, -4, -3);
    test_one(
        ref_array,
        alt_array,
        parameters,
        OverhangStrategy::SoftClip,
        359,
        b"7M",
    );
    test_one(
        ref_array,
        alt_array,
        parameters,
        OverhangStrategy::Indel,
        0,
        b"1M358D6M29D",
    );
    test_one(
        ref_array,
        alt_array,
        parameters,
        OverhangStrategy::LeadingIndel,
        0,
        b"1M1D6M",
    );
    test_one(
        ref_array,
        alt_array,
        parameters,
        OverhangStrategy::Ignore,
        359,
        b"7M",
    );
}
