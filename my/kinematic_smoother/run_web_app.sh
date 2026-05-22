#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -d "build" ]; then
    echo "Creating build directory and running cmake..."
    mkdir build
fi

cd build
cmake .. -DBUILD_PYTHON=ON -DBUILD_TESTS=OFF
make -j$(nproc) py_kinematic_smoother
cd ..

export PYTHONPATH="$SCRIPT_DIR/build:$SCRIPT_DIR:$PYTHONPATH"
python3 web/app.py
