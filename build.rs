fn main() {
    println!("cargo:rustc-link-search=native=.");
    println!("cargo:rustc-link-lib=static=mandelbrot_cuda");
    println!("cargo:rustc-link-lib=dylib=cuda");
    println!("cargo:rustc-link-lib=dylib=cudart");
}
