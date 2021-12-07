use criterion::{criterion_group, criterion_main, Criterion, SamplingMode};

use std::cmp;
use std::fs::File;
use std::io::{BufRead, BufReader};

criterion_main!(benches);
criterion_group!(benches, bench);

fn parse_qual(tokens: &mut std::str::SplitWhitespace, min: u8) -> Vec<u8> {
    tokens
        .next()
        .unwrap()
        .as_bytes()
        .iter()
        .copied()
        .map(|b| cmp::max(min, b - 33))
        .collect()
}

fn bench(c: &mut Criterion) {
    let in_file = File::open("tests/data/10s.in").unwrap();
    let mut in_lines = BufReader::new(in_file).lines();
    let mut tests = Vec::new();
    while let Some(line) = in_lines.next() {
        let line = line.unwrap();
        let mut tokens = line.split_whitespace();
        let rs_num = tokens.next().unwrap().parse::<usize>().unwrap();
        let hap_num = tokens.next().unwrap().parse::<usize>().unwrap();
        let mut rs = Vec::new();
        for _ in 0..rs_num {
            let line = in_lines.next().unwrap().unwrap();
            let mut tokens = line.split_whitespace();
            let rs_bytes = tokens.next().unwrap().as_bytes().to_vec();
            let q = &parse_qual(&mut tokens, 6);
            let i = &parse_qual(&mut tokens, 0);
            let d = &parse_qual(&mut tokens, 0);
            let c = &parse_qual(&mut tokens, 0);
            rs.push((rs_bytes, q.to_vec(), i.to_vec(), d.to_vec(), c.to_vec()));
        }
        let mut hap = Vec::new();
        for _ in 0..hap_num {
            hap.push(in_lines.next().unwrap().unwrap());
        }
        tests.push((rs, hap));
    }

    macro_rules! bench {
        ($f: expr) => {
            for (rs, hap) in &tests {
                for (rs, q, i, d, c) in rs.iter() {
                    for hap in hap.iter() {
                        $f(hap.as_bytes(), rs, q, i, d, c);
                    }
                }
            }
        };
    }

    if let Some(f) = gkl::pairhmm::forward_f32x8() {
        c.bench_function("forward_f32x8", |b| b.iter(|| bench!(f)));
    }

    if let Some(f) = gkl::pairhmm::forward_f32x16() {
        c.bench_function("forward_f32x16", |b| b.iter(|| bench!(f)));
    }

    if let Some(f) = gkl::pairhmm::forward_f64x4() {
        c.bench_function("forward_f64x4", |b| b.iter(|| bench!(f)));
    }

    if let Some(f) = gkl::pairhmm::forward_f64x8() {
        c.bench_function("forward_f64x8", |b| b.iter(|| bench!(f)));
    }

    {
        let f = gkl::pairhmm::forward;
        c.bench_function("forward_any", |b| b.iter(|| bench!(f)));
    }

    let mut group = c.benchmark_group("slow");
    group.sample_size(10).sampling_mode(SamplingMode::Flat);

    {
        let f = gkl::pairhmm::forward_f32x1();
        group.bench_function("forward_f32x1", |b| b.iter(|| bench!(f)));
    }

    {
        let f = gkl::pairhmm::forward_f64x1();
        group.bench_function("forward_f64x1", |b| b.iter(|| bench!(f)));
    }

    group.finish();
}
