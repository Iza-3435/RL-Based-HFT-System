#!/bin/bash

# Setup script for Custom Replay Buffer project
set -e

echo "Setting up Custom Replay Buffer development environment..."

# Detect OS
OS=""
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
else
    echo "Unsupported OS: $OSTYPE"
    exit 1
fi

# Install dependencies based on OS
install_dependencies() {
    if [[ "$OS" == "linux" ]]; then
        echo "Installing dependencies for Linux..."
        sudo apt-get update
        sudo apt-get install -y \
            build-essential \
            cmake \
            clang-format \
            cppcheck \
            libjemalloc-dev \
            valgrind \
            gdb
    elif [[ "$OS" == "macos" ]]; then
        echo "Installing dependencies for macOS..."
        if ! command -v brew &> /dev/null; then
            echo "Homebrew not found. Please install Homebrew first."
            exit 1
        fi
        brew install \
            cmake \
            clang-format \
            cppcheck \
            jemalloc
    fi
}

# Create build directory
setup_build() {
    echo "Setting up build directory..."
    mkdir -p build
    cd build
    cmake -DCMAKE_BUILD_TYPE=Release ..
    make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
    cd ..
}

# Run initial tests
run_tests() {
    echo "Running initial tests..."
    cd build
    if [[ -f "ctest" ]] || command -v ctest &> /dev/null; then
        ctest --output-on-failure
    else
        echo "CTest not available, skipping tests"
    fi
    cd ..
}

# Make scripts executable
make_executable() {
    chmod +x scripts/*.sh
}

# Main setup flow
main() {
    make_executable
    
    if [[ "$1" == "--skip-deps" ]]; then
        echo "Skipping dependency installation..."
    else
        install_dependencies
    fi
    
    setup_build
    
    if [[ "$1" != "--skip-tests" ]]; then
        run_tests
    fi
    
    echo "Setup complete!"
    echo "You can now run:"
    echo "  make release    - Build release version"
    echo "  make debug      - Build debug version"
    echo "  make test       - Run tests"
    echo "  make benchmark  - Run benchmarks"
}

main "$@"