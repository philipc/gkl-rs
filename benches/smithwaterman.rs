use criterion::{criterion_group, criterion_main, Criterion, SamplingMode};

use std::fs::File;
use std::io::{BufRead, BufReader};

criterion_main!(benches);
criterion_group!(benches, bench);

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
            rs.push(rs_bytes);
        }
        let mut hap = Vec::new();
        for _ in 0..hap_num {
            hap.push(in_lines.next().unwrap().unwrap());
        }
        tests.push((rs, hap));
    }

    let parameters = gkl::smithwaterman::Parameters::new(1, -4, -6, -1);
    macro_rules! bench {
        ($f: expr) => {
            for (rs, hap) in &tests {
                for rs in rs.iter() {
                    for hap in hap.iter() {
                        $f(
                            hap.as_bytes(),
                            rs,
                            parameters,
                            gkl::smithwaterman::OverhangStrategy::Indel,
                        )
                        .unwrap();
                    }
                }
            }
        };
    }

    if let Some(f) = gkl::smithwaterman::align_i32x8() {
        c.bench_function("align_i32x8", |b| b.iter(|| bench!(f)));
    }

    if let Some(f) = gkl::smithwaterman::align_i32x16() {
        c.bench_function("align_i32x16", |b| b.iter(|| bench!(f)));
    }

    let mut group = c.benchmark_group("slow");
    group.sample_size(10).sampling_mode(SamplingMode::Flat);

    {
        let f = gkl::smithwaterman::align_i32x1();
        group.bench_function("align_i32x1", |b| b.iter(|| bench!(f)));
    }

    group.finish();
}
