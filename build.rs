fn main() {
    cc::Build::new()
        .file("gkl/pairhmm/avx_impl.cc")
        .file("gkl/pairhmm/avx512_impl.cc")
        .file("gkl/pairhmm/pairhmm_common.cc")
        .flag("-mavx")
        .flag("-mavx2")
        .flag("-mavx512f")
        .flag("-mavx512dq")
        .flag("-mavx512vl")
        .include("gkl/pairhmm")
        .compile("gkl");
}
