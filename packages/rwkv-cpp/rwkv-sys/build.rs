#![allow(clippy::uninlined_format_args)]

// Based on llm-chain implementation
// https://github.com/sobelio/llm-chain/blob/main/llm-chain-llama/sys/build.rs
extern crate bindgen;

use platforms::{Arch, Platform, OS};
use std::env;
use std::path::PathBuf;

fn main() {
    let target = env::var("TARGET").unwrap();
    let platform = Platform::find(&target).unwrap();
    env::set_var("RUSTFLAGS", "-C target-feature=+crt-static");
    env::set_var("CXXFLAGS", "-fPIC");
    env::set_var("CFLAGS", "-fPIC");

    // Link macOS Accelerate framework for matrix calculations
    if platform.target_os == OS::MacOS {
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }

    println!("cargo:rustc-link-search={}", env::var("OUT_DIR").unwrap());
    println!("cargo:rustc-link-lib=static=rwkv");
    println!("cargo:rerun-if-changed=wrapper.h");

    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_arg("-I./rwkv.cpp")
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

    // stop if we're on docs.rs
    if env::var("DOCS_RS").is_ok() {
        return;
    }

    // build lib
    env::set_current_dir("rwkv.cpp").expect("Unable to change directory to rwkv.cpp");
    _ = std::fs::remove_dir_all("build");
    _ = std::fs::create_dir("build");
    env::set_current_dir("build").expect("Unable to change directory to rwkv.cpp build");

    let mut command = std::process::Command::new("cmake");
    let command = command
        .arg("..")
        .arg("-DCMAKE_BUILD_TYPE=Release")
        .arg("-DRWKV_ALL_WARNINGS=OFF");

    if platform.target_os == OS::MacOS {
        if platform.target_arch == Arch::AArch64 {
            command
                .arg("-DAPPLE=ON")
                .arg("-DRWKV_ACCELERATE=ON")
                .arg("-DCMAKE_SYSTEM_NAME=Darwin")
                .arg("-DCMAKE_SYSTEM_PROCESSOR=apple-m1")
                .arg("-DCMAKE_OSX_ARCHITECTURES=arm64")
                .arg("-DRWKV_NATIVE=OFF");
        } else {
            command
                .arg("-DAPPLE=ON")
                .arg("-DRWKV_ACCELERATE=ON")
                .arg("-DCMAKE_SYSTEM_NAME=Darwin")
                .arg("-DCMAKE_SYSTEM_PROCESSOR=x86_64");
        }
    } else {
        // os except macOS will enable static linking
        command.arg("-DRWKV_STATIC=ON");
    }

    let code = command.status().expect("Failed to generate build script");
    if code.code() != Some(0) {
        panic!("Failed to generate build script");
    }

    let code = std::process::Command::new("cmake")
        .arg("--build")
        .arg(".")
        .args(["--config", "Release"])
        .status()
        .expect("Failed to build lib");
    if code.code() != Some(0) {
        panic!("Failed to build lib");
    }

    // move librwkv.a to where Cargo expects it (OUT_DIR)
    #[cfg(target_os = "windows")]
    {
        std::fs::copy(
            "Release/rwkv.lib",
            format!("{}/rwkv.lib", env::var("OUT_DIR").unwrap()),
        )
        .expect("Failed to copy lib");
    }

    #[cfg(not(target_os = "windows"))]
    {
        std::fs::copy(
            "librwkv.a",
            format!("{}/librwkv.a", env::var("OUT_DIR").unwrap()),
        )
        .expect("Failed to copy lib");
    }
    // clean the rwkv build directory to prevent Cargo from complaining during crate publish
    _ = std::fs::remove_dir_all("build");
}
