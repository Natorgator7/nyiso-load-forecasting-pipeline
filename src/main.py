
import subprocess
import sys
from pathlib import Path


def run_script(script_name: str) -> None:
    script_path = Path(__file__).resolve().parent / script_name

    if not script_path.exists():
        raise FileNotFoundError(f"Missing script: {script_path}")

    print(f"\n=== Running {script_name} ===")
    result = subprocess.run([sys.executable, str(script_path)], check=False)

    if result.returncode != 0:
        raise RuntimeError(f"{script_name} failed with exit code {result.returncode}")


if __name__ == "__main__":
    scripts = ["ingest.py", "features.py", "train.py", "evaluation.py"]

    try:
        for script in scripts:
            run_script(script)
        print("\nPipeline completed successfully.")
    except Exception as exc:
        print(f"\nPipeline stopped: {exc}")
        sys.exit(1)
# %%
