"""
One-shot runner: generate data → train all models → launch Flask app.
"""
import subprocess, sys

python = sys.executable

print("=" * 55)
print("  Step 1/3 — Generating dataset")
print("=" * 55)
subprocess.run([python, "generate_data.py"], check=True)

print("\n" + "=" * 55)
print("  Step 2/3 — Training models (this may take a few minutes)")
print("=" * 55)
subprocess.run([python, "-m", "src.train"], check=True)

print("\n" + "=" * 55)
print("  Step 3/3 — Launching Flask app at http://127.0.0.1:5000")
print("=" * 55)
subprocess.run([python, "app.py"], check=True)
