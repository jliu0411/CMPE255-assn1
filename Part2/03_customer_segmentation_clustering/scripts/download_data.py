from pathlib import Path
import shutil,subprocess
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'data/raw';out.mkdir(parents=True,exist_ok=True)
if not shutil.which('kaggle'):raise SystemExit('Install the Kaggle CLI and configure credentials first.')
subprocess.run(['kaggle','datasets','download','-d','imakash3011/customer-personality-analysis','-p',str(out),'--unzip'],check=True);print(f'Data downloaded to {out}')
