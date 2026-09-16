#!/usr/bin/env bash
set -e

# SonarVision Unified Application Launcher
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
FRONTEND_DIST="$SCRIPT_DIR/frontend/dist"

usage() {
    echo "Usage: ./start.sh [options]"
    echo ""
    echo "Options:"
    echo "  --dev            Run backend and frontend dev server concurrently (hot reloading)"
    echo "  --build          Force rebuild frontend assets before starting"
    echo "  --host HOST      Host address to bind (default: 0.0.0.0)"
    echo "  --port PORT      Port number to bind (default: 8000)"
    echo "  --help           Show this help message"
    echo ""
}

DEV_MODE=false
FORCE_BUILD=false
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dev)
            DEV_MODE=true
            shift
            ;;
        --build)
            FORCE_BUILD=true
            shift
            ;;
        --host)
            HOST="$2"
            EXTRA_ARGS+=("--host" "$2")
            shift 2
            ;;
        --port)
            PORT="$2"
            EXTRA_ARGS+=("--port" "$2")
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# Ensure frontend build exists for unified serving
# --- Python environment bootstrap (fresh-clone safe) ---
if [ ! -x "$SCRIPT_DIR/.venv/bin/python" ]; then
    echo "[0/2] No virtualenv found — creating .venv and installing dependencies..."
    echo "      (this runs once; later launches reuse it)"
    python3 -m venv "$SCRIPT_DIR/.venv" || {
        echo "ERROR: could not create .venv. On Debian/Ubuntu run: sudo apt install python3-venv"
        exit 1
    }
    "$SCRIPT_DIR/.venv/bin/pip" install --upgrade pip
    "$SCRIPT_DIR/.venv/bin/pip" install -r "$SCRIPT_DIR/requirements.txt" || {
        echo "ERROR: dependency install failed. See output above."
        exit 1
    }
fi
PYTHON="$SCRIPT_DIR/.venv/bin/python"

# --- Node.js precheck (only needed when a frontend build must run) ---
need_build=false
if [ "$DEV_MODE" = true ]; then
    need_build=true
elif [ ! -d "$FRONTEND_DIST" ] || [ "$FORCE_BUILD" = true ]; then
    need_build=true
fi
if [ "$need_build" = true ] && ! command -v npm >/dev/null 2>&1; then
    echo "ERROR: Node.js/npm is required to build/run the frontend."
    echo "       Install Node.js 20+ from https://nodejs.org/ then re-run ./start.sh"
    exit 1
fi

# --- Model weights check (warn, don't fail: engine falls back to Demo Mode) ---
if ! ls "$SCRIPT_DIR"/models/*.onnx >/dev/null 2>&1; then
    echo "WARNING: no ONNX weights found in models/ — the app will start in DEMO mode"
    echo "         (sample analysis works, detections are simulated, nothing is real)."
    echo "         For real inference, download weights first:"
    echo "           \$SCRIPT_DIR/.venv/bin/hf download Dinoman1221/sonarvision-yolov8-esi-v6 yolo_esi_v6_fp16.onnx --local-dir models/"
    echo "           cp models/yolo_esi_v6_fp16.onnx models/yolo_esi_fp16.onnx"
fi

if [ "$DEV_MODE" = false ]; then
    if [ ! -d "$FRONTEND_DIST" ] || [ "$FORCE_BUILD" = true ]; then
        echo "[1/2] Building frontend production bundle..."
        (cd frontend && npm install && npm run build)
    else
        echo "[1/2] Frontend build verified."
    fi
    echo "[2/2] Launching unified SonarVision web app on http://${HOST}:${PORT}..."
    exec "$PYTHON" run_app.py --host "$HOST" --port "$PORT" "${EXTRA_ARGS[@]}"
else
    echo "[Dev Mode] Starting backend (port $PORT) and frontend dev server (port 5173)..."
    
    cleanup() {
        echo ""
        echo "Shutting down servers..."
        kill 0
    }
    trap cleanup SIGINT SIGTERM EXIT

    "$PYTHON" run_app.py --host "$HOST" --port "$PORT" "${EXTRA_ARGS[@]}" &
    BACKEND_PID=$!

    (cd frontend && npm run dev) &
    FRONTEND_PID=$!

    wait
fi
