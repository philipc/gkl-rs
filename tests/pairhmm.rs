use std::cmp;
use std::fs::File;
use std::io::{BufRead, BufReader};

fn test_one(hap: &[u8], rs: &[u8], q: &[u8], i: &[u8], d: &[u8], c: &[u8], expected: f64) {
    println!("expected: {}", expected);
    if let Some(fp) = gkl::pairhmm_avxs(hap, rs, q, i, d, c) {
        println!("{}", fp.log10());
        assert!((fp.log10() as f64 - expected).abs() < 1e-5);
    }

    if let Some(fp) = gkl::pairhmm_avxd(hap, rs, q, i, d, c) {
        println!("{}", fp.log10());
        assert!((fp.log10() - expected).abs() < 1e-5);
    }

    if let Some(fp) = gkl::pairhmm_avx512s(hap, rs, q, i, d, c) {
        println!("{}", fp.log10());
        assert!((fp.log10() as f64 - expected).abs() < 1e-5);
    }

    if let Some(fp) = gkl::pairhmm_avx512d(hap, rs, q, i, d, c) {
        println!("{}", fp.log10());
        assert!((fp.log10() - expected).abs() < 1e-5);
    }
}

#[test]
fn simple() {
    test_one(
        b"ACGT",
        b"ACGT",
        b"++++",
        b"++++",
        b"++++",
        b"++++",
        -6.022797e-01,
    );
}

#[test]
fn data_file() {
    let file = File::open("tests/pairhmm-testdata.txt").unwrap();
    let lines = BufReader::new(file).lines();
    for line in lines {
        let line = line.unwrap();
        if line.starts_with('#') {
            continue;
        }
        let mut tokens = line.split_whitespace();
        let hap = tokens.next().unwrap().as_bytes();
        let rs = tokens.next().unwrap().as_bytes();
        let parse_qual = |tokens: &mut std::str::SplitWhitespace, min| -> Vec<u8> {
            tokens
                .next()
                .unwrap()
                .as_bytes()
                .iter()
                .copied()
                .map(|b| cmp::max(min, b - 33))
                .collect()
        };
        let q = &parse_qual(&mut tokens, 6);
        let i = &parse_qual(&mut tokens, 0);
        let d = &parse_qual(&mut tokens, 0);
        let c = &parse_qual(&mut tokens, 0);
        let expected = tokens.next().unwrap().parse::<f64>().unwrap();
        test_one(hap, rs, q, i, d, c, expected);
    }
}
