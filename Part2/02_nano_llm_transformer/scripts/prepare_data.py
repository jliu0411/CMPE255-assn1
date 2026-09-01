from pathlib import Path
import hashlib,json,re
ROOT=Path(__file__).resolve().parents[1];raw=ROOT/"data/raw/corpus.txt";out=ROOT/"data/processed";out.mkdir(parents=True,exist_ok=True)
if not raw.exists(): raise SystemExit(f"Missing {raw}")
text=raw.read_text(encoding="utf-8");text=re.sub(r"\r\n?","\n",text);blocks=[];seen=set()
for block in text.split("\n\n"):
    clean=re.sub(r"[ \t]+"," ",block).strip()
    key=hashlib.sha256(clean.lower().encode()).hexdigest()
    if len(clean)>=20 and key not in seen:blocks.append(clean);seen.add(key)
text="\n\n".join(blocks);cut=int(len(text)*.9);cut=text.rfind("\n",0,cut);train,val=text[:cut],text[cut:]
(out/"train.bin").write_bytes(train.encode());(out/"val.bin").write_bytes(val.encode());stats={"characters":len(text),"utf8_bytes":len(text.encode()),"documents":len(blocks),"train_bytes":len(train.encode()),"validation_bytes":len(val.encode()),"sha256":hashlib.sha256(text.encode()).hexdigest()};(out/"stats.json").write_text(json.dumps(stats,indent=2));print(json.dumps(stats,indent=2))
