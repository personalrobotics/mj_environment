#!/bin/bash
# Helper script to run MuJoCo demos with mjpython on macOS
# Automatically handles DYLD_LIBRARY_PATH for uv-managed Python installations

# Find the actual Python executable (through uv if needed)
if command -v uv &> /dev/null; then
    PYTHON_EXEC=$(uv run python -c "import sys; print(sys.executable)" 2>/dev/null)
else
    PYTHON_EXEC=$(python -c "import sys; print(sys.executable)" 2>/dev/null)
fi

if [ -z "$PYTHON_EXEC" ]; then
    echo "[ERROR] Could not find Python executable"
    exit 1
fi

# Resolve symlinks to find the actual Python binary
PYTHON_REAL=$(readlink -f "$PYTHON_EXEC" 2>/dev/null || python -c "import os; print(os.path.realpath('$PYTHON_EXEC'))" 2>/dev/null || echo "$PYTHON_EXEC")
PYTHON_DIR=$(dirname "$PYTHON_REAL")

# Try to find the lib directory - check common locations
PYTHON_LIB=""
if [ -d "$PYTHON_DIR/../lib" ]; then
    PYTHON_LIB="$PYTHON_DIR/../lib"
elif [ -d "$HOME/.local/share/uv/python" ]; then
    # For uv: find the actual cpython installation directory
    # Extract version from Python executable path or version output
    PYTHON_VERSION_FULL=$(python --version 2>&1 | awk '{print $2}')
    PYTHON_VERSION=$(echo "$PYTHON_VERSION_FULL" | cut -d. -f1,2)
    PYTHON_MINOR=$(echo "$PYTHON_VERSION_FULL" | cut -d. -f3)
    
    # Try to find matching cpython installation
    UV_PYTHON_DIR=$(find "$HOME/.local/share/uv/python" -type d -name "cpython-${PYTHON_VERSION_FULL}*" 2>/dev/null | head -1)
    if [ -z "$UV_PYTHON_DIR" ]; then
        # Fallback: find any cpython directory
        UV_PYTHON_DIR=$(find "$HOME/.local/share/uv/python" -type d -name "cpython-*" 2>/dev/null | head -1)
    fi
    
    if [ -n "$UV_PYTHON_DIR" ] && [ -d "$UV_PYTHON_DIR/lib" ]; then
        PYTHON_LIB="$UV_PYTHON_DIR/lib"
    fi
fi

# Set DYLD_LIBRARY_PATH if we found the library path
if [ -n "$PYTHON_LIB" ] && [ -d "$PYTHON_LIB" ]; then
    export DYLD_LIBRARY_PATH="$PYTHON_LIB"
    echo "[INFO] Using Python library path: $PYTHON_LIB"
else
    echo "[WARN] Could not automatically find Python library path."
    echo "[WARN] Attempting to run without DYLD_LIBRARY_PATH..."
fi

# Run mjpython with the demo
if [ -f ".venv/bin/mjpython" ]; then
    .venv/bin/mjpython "$@"
else
    echo "[ERROR] mjpython not found in .venv/bin/"
    echo "[ERROR] Make sure you've installed the package: uv pip install -e ."
    exit 1
fi

