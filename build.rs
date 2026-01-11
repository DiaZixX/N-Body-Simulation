// fn main() {
//     // Compile the CUDA kernel with nvcc into a static object and link it.
//     use std::process::Command;
//     use std::env;
//
//     let out_dir = env::var("OUT_DIR").unwrap();
//     let src = "src/cuda/kernel.cu";
//     let obj = format!("{}/kernel.o", out_dir);
//
//     // nvcc must be available in PATH
//     let status = Command::new("nvcc")
//         .args(&["-c", src, "-o", &obj, "-Xcompiler", "-fPIC"])
//         .status()
//         .expect("failed to run nvcc");
//
//     if !status.success() {
//         panic!("nvcc failed");
//     }
//
//     println!("cargo:rustc-link-search=native={}", out_dir);
//     println!("cargo:rustc-link-lib=static=kernel");
// }

extern crate cc;

fn main() {
    /*
    cc::Build::new()
        .cuda(true)
        .flag("-cudart=static")
        .flag("-O3")
        .flag("-gencode=arch=compute_86,code=sm_86")
        .file("src/cuda/kernel.cu")
        .compile("compute_forces_gpu");

    /* Link CUDA Runtime (libcudart.so) */

    // Add link directory
    // - This path depends on where you install CUDA (i.e. depends on your Linux distribution)
    // - This should be set by `$LIBRARY_PATH`
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-lib=cudart");

    /* Optional: Link CUDA Driver API (libcuda.so) */

    // println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64/stub");
    // println!("cargo:rustc-link-lib=cuda");
    */
}
