import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def verify_inputs():
    manifest = json.loads((ROOT / "MANIFEST.json").read_text())
    for entry in manifest["inputs"]:
        path = ROOT / entry["file"]
        if not path.is_file():
            raise SystemExit(f"Missing input: {entry['file']}. Keep the complete project folder together.")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != entry["sha256"]:
            raise SystemExit(f"Input checksum mismatch: {entry['file']}")
    print(f"Verified {len(manifest['inputs'])} input files.", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Compare starvation responses within a reference regulatory hierarchy.")
    parser.add_argument("--check-inputs", action="store_true", help="Verify bundled data and exit without analysis.")
    parser.add_argument("--check-sequences", action="store_true", help="Regenerate orthology checks using a local DIAMOND installation.")
    arguments = parser.parse_args()
    verify_inputs()
    if arguments.check_inputs:
        return
    missing = [name for name in ["numpy", "pandas", "matplotlib", "Bio", "pydeseq2"]
               if importlib.util.find_spec(name) is None]
    if missing:
        raise SystemExit("Missing Python packages. Run: python -m pip install -r requirements.txt")
    if arguments.check_sequences and shutil.which("diamond") is None:
        raise SystemExit("Sequence checks require DIAMOND on PATH; omit --check-sequences to use the bundled checks.")
    stages = ["prepare.py"]
    if arguments.check_sequences:
        stages.append("check_sequences.py")
    stages.extend(["analyse.py", "plot_results.py", "validate.py"])
    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    for stage in stages:
        log_path = logs / (Path(stage).stem + ".log")
        print(f"Running {stage}; log: {log_path.relative_to(ROOT)}", flush=True)
        with log_path.open("w") as output:
            result = subprocess.run([sys.executable, str(ROOT / "code" / stage)], cwd=ROOT,
                                    stdout=output, stderr=subprocess.STDOUT)
        if result.returncode:
            raise SystemExit(f"{stage} failed. Details: {log_path}")
    print("Finished. Inspect results/figures/ and results/validation.json.", flush=True)


if __name__ == "__main__":
    main()
