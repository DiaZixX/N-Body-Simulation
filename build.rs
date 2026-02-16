use std::env;
use std::path::PathBuf;

fn main() {
    // Détection de la feature vec3
    let is_vec3 = env::var("CARGO_FEATURE_VEC3").is_ok();

    // Configuration CUDA
    let cuda_path = env::var("CUDA_PATH")
        .or_else(|_| env::var("CUDA_HOME"))
        .unwrap_or_else(|_| "/usr/local/cuda".to_string());

    println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
    println!("cargo:rustc-link-lib=cudart");

    // Compilation CUDA
    let mut nvcc = cc::Build::new();
    nvcc.cuda(true);
    nvcc.flag("-cudart=static");
    nvcc.flag("-O3");

    // Détecter compute capability (adapter selon votre GPU)
    // RTX 3080/3090 = sm_86, RTX 4090 = sm_89
    nvcc.flag("-gencode=arch=compute_86,code=sm_86");

    if is_vec3 {
        nvcc.define("VEC3", None);
    }

    nvcc.file("src/cuda/kernel.cu");
    nvcc.compile("nbody_cuda");

    println!("cargo:rerun-if-changed=src/cuda/kernel.cu");
}
