#![allow(clippy::uninlined_format_args)]

// Based on llm-chain implementation
// https://github.com/sobelio/llm-chain/blob/main/llm-chain-llama/sys/build.rs
extern crate bindgen;

use dirs::home_dir;
use platforms::{Arch, Platform, OS};
use std::env;
use std::path::PathBuf;

struct BuildLinkInfo {
    link_type: String,
    #[cfg(target_os = "windows")]
    link_extension_windows: String,
    link_extension_nix: String,
    link_out_dir: String,
    cmake_link_flag: Vec<String>,
}

#[cfg(not(feature = "dynamic"))]
fn get_link_info() -> BuildLinkInfo {
    BuildLinkInfo {
        link_type: "static".to_owned(),
        #[cfg(target_os = "windows")]
        link_extension_windows: "lib".to_owned(),
        link_extension_nix: "a".to_owned(),
        link_out_dir: env::var("OUT_DIR").unwrap(),
        cmake_link_flag: vec!["-DLLAMA_STATIC=ON".to_owned()],
    }
}

#[cfg(feature = "dynamic")]
fn get_link_info() -> BuildLinkInfo {
    BuildLinkInfo {
        link_type: "dylib".to_owned(),
        #[cfg(target_os = "windows")]
        link_extension_windows: "dll".to_owned(),
        link_extension_nix: "so".to_owned(),
        link_out_dir: env::var("OUT_DIR").unwrap(),
        cmake_link_flag: vec![
            "-DLLAMA_STATIC=OFF".to_owned(),
            "-DBUILD_SHARED_LIBS=ON".to_owned(),
        ],
    }
}

fn main() {
    let home_dir = home_dir().unwrap();
    let llama_node_dir = home_dir.join(".llama-node");

    if !llama_node_dir.exists() {
        std::fs::create_dir(&llama_node_dir).expect("Unable to create .llama-node directory");
    }

    let target = env::var("TARGET").unwrap();
    let platform = Platform::find(&target).unwrap();
    env::set_var("CXXFLAGS", "-fPIC");
    env::set_var("CFLAGS", "-fPIC");

    // Link macOS Accelerate framework for matrix calculations
    if platform.target_os == OS::MacOS {
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }

    let build_link_info = get_link_info();

    println!("cargo:rustc-link-search={}", build_link_info.link_out_dir);
    println!("cargo:rustc-link-lib={}=llama", build_link_info.link_type);
    println!("cargo:rerun-if-changed=wrapper.h");

    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_arg("-I./llama.cpp")
        .clang_arg("-xc++")
        .clang_arg("-std=c++11")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .generate();

    match bindings {
        Ok(b) => {
            let out_path = PathBuf::from(build_link_info.link_out_dir.clone());
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
    env::set_current_dir("llama.cpp").expect("Unable to change directory to llama.cpp");
    _ = std::fs::remove_dir_all("build");
    _ = std::fs::create_dir("build");
    env::set_current_dir("build").expect("Unable to change directory to llama.cpp build");

    let mut command = std::process::Command::new("cmake");
    let command = command
        .arg("..")
        .arg("-DCMAKE_BUILD_TYPE=Release")
        .arg("-DLLAMA_ALL_WARNINGS=OFF")
        .arg("-DLLAMA_ALL_WARNINGS_3RD_PARTY=OFF")
        .arg("-DLLAMA_BUILD_TESTS=OFF")
        .arg("-DLLAMA_BUILD_EXAMPLES=OFF");

    for flag in build_link_info.cmake_link_flag {
        command.arg(&flag);
    }

    #[cfg(feature = "cublas")]
    {
        command
            .arg("-DLLAMA_CUBLAS=ON")
            .arg("-DCMAKE_POSITION_INDEPENDENT_CODE=ON");
    }

    if platform.target_os == OS::MacOS {
        if platform.target_arch == Arch::AArch64 {
            command
                .arg("-DAPPLE=ON")
                .arg("-DLLAMA_ACCELERATE=ON")
                .arg("-DCMAKE_SYSTEM_NAME=Darwin")
                .arg("-DCMAKE_SYSTEM_PROCESSOR=apple-m1")
                .arg("-DCMAKE_OSX_ARCHITECTURES=arm64")
                .arg("-DLLAMA_NATIVE=OFF");
        } else {
            command
                .arg("-DAPPLE=ON")
                .arg("-DLLAMA_ACCELERATE=ON")
                .arg("-DCMAKE_SYSTEM_NAME=Darwin")
                .arg("-DCMAKE_SYSTEM_PROCESSOR=x86_64");
        }
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

    // move libllama.a to where Cargo expects it (OUT_DIR)
    #[cfg(target_os = "windows")]
    {
        std::fs::copy(
            format!("Release/llama.{}", build_link_info.link_extension_windows),
            format!(
                "{}/llama.{}",
                build_link_info.link_out_dir, build_link_info.link_extension_windows
            ),
        )
        .expect("Failed to copy lib");

        #[cfg(feature = "dynamic")]
        {
            // move libllama.dll to llama_node_dir
            std::fs::copy(
                format!("Release/llama.{}", build_link_info.link_extension_windows),
                format!(
                    "{}/llama.{}",
                    llama_node_dir.display(),
                    build_link_info.link_extension_windows
                ),
            )
            .expect("Failed to copy lib");
        }
    }

    #[cfg(not(target_os = "windows"))]
    {
        std::fs::copy(
            format!("libllama.{}", build_link_info.link_extension_nix),
            format!(
                "{}/libllama.{}",
                build_link_info.link_out_dir, build_link_info.link_extension_nix
            ),
        )
        .expect("Failed to copy lib");

        #[cfg(feature = "dynamic")]
        {
            // move libllama.so to llama_node_dir
            std::fs::copy(
                format!("libllama.{}", build_link_info.link_extension_nix),
                format!(
                    "{}/libllama.{}",
                    llama_node_dir.display(),
                    build_link_info.link_extension_nix
                ),
            )
            .expect("Failed to copy lib");
        }
    }
    // clean the llama build directory to prevent Cargo from complaining during crate publish
    _ = std::fs::remove_dir_all("build");
}
