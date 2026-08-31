fn main() {
    cxx_build::bridge("src/lib.rs")
        .std("c++17")
        .compile("floorfield-cxx");

    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=src/geometry.rs");
    println!("cargo:rerun-if-changed=src/mesh.rs");
}
