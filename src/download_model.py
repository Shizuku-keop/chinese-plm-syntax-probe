"""从 ModelScope 下载模型文件到 models/<名称>/。

用法: python src/download_model.py <modelscope_id>
例: python src/download_model.py AI-ModelScope/bert-base-chinese
"""
import json
import shutil
import sys
import urllib.request
from pathlib import Path

KEEP = {
    "config.json",
    "vocab.txt",
    "tokenizer_config.json",
    "tokenizer.json",
    "model.safetensors",
    "special_tokens_map.json",
}


def main(model_id: str) -> None:
    out = Path("models") / model_id.split("/")[-1]
    out.mkdir(parents=True, exist_ok=True)
    api = f"https://modelscope.cn/api/v1/models/{model_id}/repo/files"
    with urllib.request.urlopen(api, timeout=60) as r:
        data = json.load(r)
    files = {f.get("Path") or f.get("path"): f.get("Size") or 0 for f in data["Data"]["Files"]}
    weight = "model.safetensors" if "model.safetensors" in files else "pytorch_model.bin"
    wanted = [n for n in files if n in KEEP or n == weight]
    for name in wanted:
        dest = out / name
        size = files[name]
        if dest.exists() and dest.stat().st_size == size:
            print(f"skip {name} (已存在)")
            continue
        url = f"https://modelscope.cn/models/{model_id}/resolve/master/{name}"
        print(f"下载 {name} ({size / 1e6:.1f} MB)...", flush=True)
        with urllib.request.urlopen(url, timeout=120) as r, open(dest, "wb") as w:
            shutil.copyfileobj(r, w, length=1024 * 1024)
        ok = dest.stat().st_size == size
        print(f"  {'OK' if ok else 'SIZE MISMATCH'} {name} -> {dest.stat().st_size} bytes", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
