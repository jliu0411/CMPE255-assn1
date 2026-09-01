from pathlib import Path
import shutil, subprocess, sys
ROOT=Path(__file__).resolve().parents[1]; dest=ROOT/"data/raw"
if not shutil.which("kaggle"): raise SystemExit("Kaggle CLI not found. Install it with `pip install kaggle`, then configure your API credentials.")
dest.mkdir(parents=True,exist_ok=True)
subprocess.run(["kaggle","competitions","download","-c","nyc-taxi-trip-duration","-p",str(dest)],check=True)
for archive in dest.glob("*.zip"): shutil.unpack_archive(archive,dest)
print(f"Competition data is ready in {dest}")
