#![allow(clippy::uninlined_format_args)]

// Based on llm-chain implementation
// https://github.com/sobelio/llm-chain/blob/main/llm-chain-llama/sys/build.rs
extern crate bindgen;

use std::env;
use std::path::PathBuf;

fn main() {
    let (_host, target_arch, _target_os) = get_build_target();
    let target = env::var("TARGET").unwrap();
    // Link C++ standard library
    if let Some(cpp_stdlib) = get_cpp_link_stdlib(&target) {
        println!("cargo:rustc-link-lib=dylib={}", cpp_stdlib);
        println!("cargo:rustc-link-arg=-l{}", cpp_stdlib);
    }
    // Link macOS Accelerate framework for matrix calculations
    if target.contains("apple") {
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }
    println!("cargo:rustc-link-search={}", env::var("OUT_DIR").unwrap());
    println!("cargo:rustc-link-lib=static=llama");
    println!("cargo:rerun-if-changed=wrapper.h");

    env::set_var("CXXFLAGS", "-fPIC");
    env::set_var("CFLAGS", "-fPIC");
    env::set_var("CMAKE_SYSTEM_PROCESSOR", target_arch);

    if env::var("LLAMA_DONT_GENERATE_BINDINGS").is_ok() {
        let _: u64 = std::fs::copy(
            "src/bindings.rs",
            env::var("OUT_DIR").unwrap() + "/bindings.rs",
        )
        .expect("Failed to copy bindings.rs");
    } else {
        let bindings = bindgen::Builder::default()
            .header("wrapper.h")
            .clang_arg("-I./llama.cpp")
            .parse_callbacks(Box::new(bindgen::CargoCallbacks))
            .generate();

        match bindings {
            Ok(b) => {
                let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
                b.write_to_file(out_path.join("bindings.rs"))
                    .expect("Couldn't write bindings!");
            }
            Err(e) => {
                println!("cargo:warning=Unable to generate bindings: {}", e);
                println!("cargo:warning=Using bundled bindings.rs, which may be out of date");
                // copy src/bindings.rs to OUT_DIR
                std::fs::copy(
                    "src/bindings.rs",
                    env::var("OUT_DIR").unwrap() + "/bindings.rs",
                )
                .expect("Unable to copy bindings.rs");
            }
        }
    };

    // stop if we're on docs.rs
    if env::var("DOCS_RS").is_ok() {
        return;
    }

    // build lib
    env::set_current_dir("llama.cpp").expect("Unable to change directory to whisper.cpp");
    _ = std::fs::remove_dir_all("build");
    _ = std::fs::create_dir("build");
    env::set_current_dir("build").expect("Unable to change directory to llama.cpp build");


    let code = std::process::Command::new("cmake")
        .arg("..")
        .arg("-DCMAKE_BUILD_TYPE=Release")
        // .arg("-DBUILD_SHARED_LIBS=OFF")
        .arg("-DLLAMA_ALL_WARNINGS=OFF")
        .arg("-DLLAMA_ALL_WARNINGS_3RD_PARTY=OFF")
        .arg("-DLLAMA_BUILD_TESTS=OFF")
        .arg("-DLLAMA_BUILD_EXAMPLES=OFF")
        // .arg("-DLLAMA_STATIC=ON")
        .status()
        .expect("Failed to generate build script");
    if code.code() != Some(0) {
        panic!("Failed to generate build script");
    }

    let code = std::process::Command::new("cmake")
        .arg("--build")
        .arg(".")
        .arg("--config Release")
        .status()
        .expect("Failed to build lib");
    if code.code() != Some(0) {
        panic!("Failed to build lib");
    }

    // move libllama.a to where Cargo expects it (OUT_DIR)
    #[cfg(target_os = "windows")]
    {
        std::fs::copy(
            "Release/llama.lib",
            format!("{}/llama.lib", env::var("OUT_DIR").unwrap()),
        )
        .expect("Failed to copy lib");
    }

    #[cfg(not(target_os = "windows"))]
    {
        std::fs::copy(
            "libllama.a",
            format!("{}/libllama.a", env::var("OUT_DIR").unwrap()),
        )
        .expect("Failed to copy lib");
    }
    // clean the llama build directory to prevent Cargo from complaining during crate publish
    _ = std::fs::remove_dir_all("build");
}

// From https://github.com/alexcrichton/cc-rs/blob/fba7feded71ee4f63cfe885673ead6d7b4f2f454/src/lib.rs#L2462
fn get_cpp_link_stdlib(target: &str) -> Option<&'static str> {
    if target.contains("msvc") {
        None
    } else if target.contains("apple") || target.contains("freebsd") || target.contains("openbsd") {
        Some("c++")
    } else if target.contains("android") {
        Some("c++_shared")
    } else {
        Some("stdc++")
    }
}

fn get_build_target() -> (String, String, String) {
    let target = env::var("TARGET").unwrap();
    let target_triple = target.split('-').collect::<Vec<&str>>();
    let target_arch = target_triple[0];
    let target_os = target_triple[2];
    println!("target_arch: {}", target_arch);
    println!("target_os: {}", target_os);
    let host = env::var("HOST").unwrap();
    println!("host: {}", host);

    (host, target_arch.to_string(), target_os.to_string())
}