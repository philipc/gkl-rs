//! Rust bindings for the Genomics Kernel Library.
//!
//! Provides:
//! - AVX and AVX-512 versions of the PairHMM forward algorithm
//! - AVX2 and AVX-512 versions of the Smith-Waterman sequence alignment algorithm
#![deny(missing_docs)]

pub mod pairhmm;
pub mod smithwaterman;
