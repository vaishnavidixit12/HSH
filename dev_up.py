# -----------------------------------------------------------------------------
# dev_up.py — Dev Orchestrator for HS Single-Pipe Solver
# Boots FastAPI (uvicorn) + Streamlit UI, validates YAML, exposes logs,
# and works cleanly in GitHub Codespaces.
# Key details:
#   - Binds API to API_HOST (0.0.0.0 in devcontainer.json) for port forwarding
#   - Health probe always connects via 127.0.0.1 (since 0.0.0.0 is not connectable)
#   - HEALTH_URL built after .env is loaded
# -----------------------------------------------------------------------------

from __future__ import annotations
import atexit
import os
import sys
import time
import socket
import subprocess
from pathlib import Path

# ---------------------- CONFIG (base defaults) ----------------------
PROJECT_ROOT = Path(__file__).parent.resolve()
API_APP = "api.main:app"              # uvicorn import path for FastAPI app
API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8000"))
UI_PORT  = int(os.getenv("UI_PORT",  "8501"))
UI_FILE  = PROJECT_ROOT / "ui" / "app.py"
CATALOG_PATH = os.getenv("CATALOG_PATH", str(PROJECT_ROOT / "examples" / "catalog_hs.yaml"))
PYTHONPATH_APPEND = str(PROJECT_ROOT / "src")

# ---------------------- HELPERS ----------------------
def echo(msg: str): print(f"[dev_up] {msg}", flush=True)
def fail(msg: str, code: int = 1): echo(f"❌ {msg}"); sys.exit(code)

def run_shell(cmd: str) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)

def kill_port(port: int):
    echo(f"Checking for existing process on port {port}...")
    if sys.platform.startswith("win"):
        result = run_shell(f"netstat -ano | findstr :{port}")
        for line in result.stdout.splitlines():
            if "LISTENING" in line:
                pid = line.strip().split()[-1]
                echo(f"Killing PID {pid} on port {port}")
                os.system(f"taskkill /PID {pid} /F >nul 2>&1")
    else:
        result = run_shell(f"lsof -t -i:{port}")
        for pid in result.stdout.splitlines():
            echo(f"Killing PID {pid} on port {port}")
            os.system(f"kill -9 {pid} >/dev/null 2>&1")

def check_port_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) != 0

def wait_for_api(url: str, timeout: float = 60.0) -> bool:
    import urllib.request
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(0.4)
    return False

def validate_yaml(path: str):
    try:
        import yaml
    except ImportError:
        fail("PyYAML missing → pip install PyYAML")
    p = Path(path)
    if not p.exists():
        fail(f"Catalog YAML not found: {p}")
    try:
        with p.open("r", encoding="utf-8") as f:
            yaml.safe_load(f)
    except Exception as e:
        fail(f"YAML validation failed:\n{e}")

def load_dotenv_if_present():
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        echo("Loaded .env file")

def ensure_env():
    if not os.getenv("OPENAI_API_KEY"):
        fail("OPENAI_API_KEY not set. Add to .env or set a Codespaces secret.")
    # prefer a client-connectable API_URL
    bind_host = os.getenv("API_HOST", API_HOST)
    bind_port = int(os.getenv("API_PORT", API_PORT))
    client_host = "127.0.0.1" if bind_host in ("0.0.0.0", "0") else bind_host
    os.environ.setdefault("OPENAI_MODEL", "gpt-5")
    os.environ.setdefault("API_URL", f"http://{client_host}:{bind_port}")
    os.environ.setdefault("CATALOG_PATH", CATALOG_PATH)

def which_or_fail(pkg: str, hint: str):
    try:
        __import__(pkg)
    except Exception:
        fail(f"{pkg} missing → {hint}")

# ---------------------- STARTERS ----------------------
def start_uvicorn() -> subprocess.Popen:
    env = os.environ.copy()
    env["PYTHONPATH"] = (env.get("PYTHONPATH", "") + os.pathsep + PYTHONPATH_APPEND).strip(os.pathsep)
    host = env.get("API_HOST", API_HOST)
    port = env.get("API_PORT", str(API_PORT))
    cmd = [sys.executable, "-m", "uvicorn", API_APP, "--host", host, "--port", str(port), "--reload"]
    echo(f"▶ Starting API → {' '.join(cmd)}")
    return subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

def start_streamlit() -> subprocess.Popen:
    """Launch Streamlit UI on the configured port."""
    env = os.environ.copy()
    # Helpful defaults if not already set
    env.setdefault("STREAMLIT_SERVER_ADDRESS", "0.0.0.0")
    env.setdefault("STREAMLIT_SERVER_PORT", str(UI_PORT))
    env.setdefault("STREAMLIT_SERVER_HEADLESS", "true")
    env.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")

    cmd = [
        sys.executable, "-m", "streamlit", "run", str(UI_FILE),
        "--server.port", str(UI_PORT),
        "--server.address", env["STREAMLIT_SERVER_ADDRESS"],
        "--server.headless", env["STREAMLIT_SERVER_HEADLESS"]
    ]
    echo(f"▶ Starting UI → {' '.join(cmd)}")
    return subprocess.Popen(
        cmd, cwd=str(PROJECT_ROOT), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )


# ---------------------- MAIN ----------------------
def main():
    echo("🚀 Launching HS Solver Dev Environment...")

    load_dotenv_if_present()
    ensure_env()

    # Resolve env *after* .env load
    bind_host = os.getenv("API_HOST", API_HOST)
    bind_port = int(os.getenv("API_PORT", API_PORT))
    probe_host = "127.0.0.1" if bind_host in ("0.0.0.0", "0") else bind_host
    health_url = f"http://{probe_host}:{bind_port}/health"
    catalog_path = os.getenv("CATALOG_PATH", CATALOG_PATH)

    # Deps
    which_or_fail("uvicorn",  "pip install uvicorn[standard]")
    which_or_fail("streamlit","pip install streamlit")
    which_or_fail("yaml",     "pip install PyYAML")

    echo(f"Validating {catalog_path} ...")
    validate_yaml(catalog_path)
    echo("✅ YAML OK")

    # Fresh ports
    kill_port(bind_port)
    kill_port(UI_PORT)
    if not check_port_free(probe_host, bind_port): fail(f"Port {bind_port} still in use after cleanup.")
    if not check_port_free(probe_host, UI_PORT):  fail(f"Port {UI_PORT} still in use after cleanup.")

    # Start processes
    api = start_uvicorn()
    ui  = None

    def cleanup():
        for proc in (ui, api):
            if proc and proc.poll() is None:
                proc.terminate()
                time.sleep(0.5)
                if proc.poll() is None:
                    proc.kill()
    atexit.register(cleanup)

    echo("⌛ Waiting for API /health ...")
    if not wait_for_api(health_url, timeout=60):
        # If health fails, try to dump a few API lines to help debug
        try:
            if api and api.stdout:
                echo("Last API logs:")
                for _ in range(20):
                    line = api.stdout.readline()
                    if not line: break
                    print(f"[API] {line}", end="")
        finally:
            fail("API failed to become ready in time.")

    echo("✅ API ready")

    ui = start_streamlit()
    # after starting streamlit
    echo("⌛ Waiting for Streamlit to emit 'Running on' ...")
    deadline = time.time() + 45
    ok = False
    if ui and ui.stdout:
        while time.time() < deadline:
            line = ui.stdout.readline()
            if line:
                print(f"[UI] {line}", end="")
                if "Running on" in line or "Network URL" in line:
                    ok = True
                    break
            else:
                time.sleep(0.2)
    if not ok:
        echo("⚠️ Streamlit didn’t confirm in time; check logs above.")

    echo(f"🌐 UI running at: http://localhost:{UI_PORT}")
    echo(f"📘 API docs: http://localhost:{bind_port}/docs")

    try:
        while True:
            for name, proc in [("API", api), ("UI", ui)]:
                if proc and proc.stdout:
                    line = proc.stdout.readline()
                    if line:
                        print(f"[{name}] {line}", end="")
            if (api and api.poll() is not None) or (ui and ui.poll() is not None):
                break
            time.sleep(0.2)
    except KeyboardInterrupt:
        echo("🛑 Ctrl+C pressed — shutting down...")
    finally:
        cleanup()
        echo("✅ All processes stopped cleanly.")

if __name__ == "__main__":
    main()
