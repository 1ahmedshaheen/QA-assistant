import subprocess
import time
import sys
import os

def kill_port(port):
    """Kills any process listening on the specified port (Windows)."""
    try:
        import subprocess
        # Find PID using port
        output = subprocess.check_output(f'netstat -ano | findstr :{port}', shell=True).decode()
        for line in output.strip().split('\n'):
            if 'LISTENING' in line:
                pid = line.strip().split()[-1]
                print(f"Cleaning up port {port} (Found stale process {pid})...")
                subprocess.run(['taskkill', '/F', '/PID', pid, '/T'], capture_output=True)
    except Exception:
        pass # Port probably free or access denied

def run_project():
    """Launches FastAPI backend and Gradio frontend concurrently."""
    print("Starting Smart Contract Assistant Project...")

    # Kill any stale processes on the required ports
    kill_port(9012)
    kill_port(8090)

    # Ensure venv is used if available
    python_exe = sys.executable
    if os.path.exists("venv/Scripts/python.exe"):
        python_exe = os.path.abspath("venv/Scripts/python.exe")
    elif os.path.exists("venv/bin/python"):
        python_exe = os.path.abspath("venv/bin/python")

    # 1. Start FastAPI Backend
    print("Launching FastAPI Backend (LangServe)...")
    backend_process = subprocess.Popen(
        [python_exe, "-m", "app.api.server"],
        env={**os.environ, "PYTHONIOENCODING": "utf-8"}
    )

    # Wait for backend to initialize
    print("Waiting for backend to warm up...")
    time.sleep(5)

    # 2. Start Gradio Frontend
    print("Launching Gradio Frontend...")
    frontend_process = subprocess.Popen(
        [python_exe, "frontend/gradio_app.py"],
        env={**os.environ, "PYTHONIOENCODING": "utf-8"}
    )

    try:
        while True:
            time.sleep(1)
            if backend_process.poll() is not None:
                print("Backend process terminated unexpectedly.")
                break
            if frontend_process.poll() is not None:
                print("Frontend process terminated unexpectedly.")
                break
    except KeyboardInterrupt:
        print("\nShutting down project...")
    finally:
        backend_process.terminate()
        frontend_process.terminate()
        print("Shutdown complete.")

if __name__ == "__main__":
    run_project()
