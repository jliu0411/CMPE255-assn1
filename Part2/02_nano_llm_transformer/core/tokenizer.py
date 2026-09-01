from pathlib import Path

class ByteTokenizer:
    """UTF-8 byte tokenizer: fixed vocabulary, reversible, and dependency free."""
    vocab_size=256
    def encode(self,text:str)->list[int]: return list(text.encode("utf-8",errors="replace"))
    def decode(self,ids:list[int])->str: return bytes(i for i in ids if 0<=i<256).decode("utf-8",errors="replace")
    def save(self,path): Path(path).write_text('{"type":"utf8-byte","vocab_size":256}',encoding="utf-8")
