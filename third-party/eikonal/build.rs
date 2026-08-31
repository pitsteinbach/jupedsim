fn main() {
    cxx_build::bridge("src/lib.rs")
        .std("c++17")
        .compile("eikonal-cxx");

    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=src/fsm.rs");
}
