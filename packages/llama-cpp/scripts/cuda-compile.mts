import { exec, execSync } from "child_process";

const checkClang: () => boolean = () => {
    try {
        process.stdout.write("Checking clang...");
        execSync("clang --version");
        console.log("✅");
        return true;
    } catch (error) {
        return false;
    }
};

const checkGcc: () => boolean = () => {
    try {
        process.stdout.write("Checking gcc...");
        execSync("gcc --version");
        console.log("✅");
        return true;
    } catch (error) {
        return false;
    }
};

const checkEnv = () => {
    // check if rustc is installed and available
    try {
        process.stdout.write("Checking rustc...");
        execSync("rustc --version");
        console.log("✅");
    } catch (error) {
        console.log("❌");
        console.error("rustc is not installed or not available in PATH");
        console.log("Please install rustc from https://rustup.rs/");
        process.exit(1);
    }

    // check if cargo is installed and available
    try {
        process.stdout.write("Checking cargo...");
        execSync("cargo --version");
        console.log("✅");
    } catch (error) {
        console.log("❌");
        console.error("cargo is not installed or not available in PATH");
        console.log("Please install cargo from https://rustup.rs/");
        process.exit(1);
    }

    // check if cmake is installed and available
    try {
        process.stdout.write("Checking cmake...");
        execSync("cmake --version");
        console.log("✅");
    } catch (error) {
        console.log("❌");
        console.error("cmake is not installed or not available in PATH");
        console.log(
            "Please install cmake from https://cmake.org/install/ or your package manager. Make sure to add it to PATH."
        );
        process.exit(1);
    }

    // check if llvm is installed and available
    try {
        process.stdout.write("Checking llvm...");
        execSync("llvm-config --version");
        console.log("✅");
    } catch (error) {
        console.log("❌");
        console.error("llvm is not installed or not available in PATH");
        console.log(
            "Please install llvm from https://releases.llvm.org/download.html or your package manager. Make sure to add it to PATH."
        );
        process.exit(1);
    }

    // check if clang or gcc is installed and available
    if (!checkClang() && !checkGcc()) {
        console.log("❌");

        console.error("clang or gcc is not installed or not available in PATH");

        // install clang
        console.log(
            "Please install clang from https://releases.llvm.org/download.html or your package manager. Make sure to add it to PATH."
        );

        // or install gcc
        console.log(
            "Alternatively, you can install gcc from https://gcc.gnu.org/install/ or your package manager. Make sure to add it to PATH."
        );

        process.exit(1);
    }

    // check if nvcc is installed and available
    try {
        process.stdout.write("Checking nvcc...");
        execSync("nvcc --version");
        console.log("✅");
    } catch (error) {
        console.log("❌");
        console.error("nvcc is not installed or not available in PATH");
        console.log(
            "Please install nvcc from https://developer.nvidia.com/cuda-downloads or your package manager. Make sure to add it to PATH."
        );
        process.exit(1);
    }
};

const compile = () => {
    const buildProcess = exec(
        `napi build --platform --release --features=cublas`
    );
    buildProcess.stdout?.pipe(process.stdout);
    buildProcess.stderr?.pipe(process.stderr);

    return new Promise<boolean>((resolve, reject) => {
        buildProcess.on("close", (code) => {
            if (code !== 0) {
                reject(code);
            } else {
                resolve(true);
            }
        });
    });
};

const postCompile = async () => {
    const homeDir = process.env.HOME || process.env.USERPROFILE;
    const extension = process.platform === "win32" ? ".dll" : ".so";
    const libPath = `${homeDir}/.llama-node/libllama${extension}`;

    // check if libllama.so exists
    try {
        process.stdout.write("Checking libllama...");
        execSync(`ls ${libPath}`);
        console.log("✅");
    } catch (error) {
        console.error("libllama is not found");
        console.log(
            "Please make sure that libllama is compiled and installed under ~/.llama-node/"
        );
        process.exit(1);
    }

    // check if libllama.so is under the LD_LIBRARY_PATH
    try {
        process.stdout.write("Checking LD_LIBRARY_PATH...");
        execSync(`echo $LD_LIBRARY_PATH | grep ${homeDir}/.llama-node`)
        console.log("✅");
    } catch (error) {
        console.log("\n\n");
        console.log("libllama is not under LD_LIBRARY_PATH");
        console.log("add this to your .bashrc or .zshrc:");
        console.log(
            `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.llama-node`
        );
    }
};

const run = async () => {
    console.log("Checking environment...\n");
    checkEnv();

    console.log("\n\n");

    console.log("Compiling...\n");
    await compile();

    console.log("\n\n");

    console.log("Post-compiling...\n");
    await postCompile();
    console.log("Compile successful!");
};

run();
