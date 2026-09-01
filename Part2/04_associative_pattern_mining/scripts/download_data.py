from pathlib import Path
import shutil,subprocess
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'data/raw';out.mkdir(parents=True,exist_ok=True)
if not shutil.which('kaggle'):raise SystemExit('Install the Kaggle CLI and configure credentials first.')
subprocess.run(['kaggle','datasets','download','-d','divanshu22/online-retail-dataset','-p',str(out),'--unzip'],check=True);print(f'Downloaded data to {out}. Pass the resulting CSV/XLSX path to src.mine.')
