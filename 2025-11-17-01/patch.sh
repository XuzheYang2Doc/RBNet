
WRP="$CONDA_PREFIX/lib/python3.10/site-packages/mmengine/model/wrappers/__init__.py"
cp "$WRP" "${WRP}.bak_fix_mmf"

python - <<'PY'
import os, re, pathlib

p = pathlib.Path(os.environ["CONDA_PREFIX"]) / "lib/python3.10/site-packages/mmengine/model/wrappers/__init__.py"
txt = p.read_text()

# 把你之前加的 except ... pass 替换成：定义一个占位符 MMFullyShardedDataParallel
pattern = r"(except Exception:[^\n]*FSDP[^\n]*\n)([ \t]+)pass\s*\n"
m = re.search(pattern, txt)
if not m:
    raise SystemExit("没找到之前插入的 FSDP except/pass 片段。请把 wrappers/__init__.py 相关片段贴我。")

indent = m.group(2)
replacement = (
    m.group(1) +
    f"{indent}class MMFullyShardedDataParallel:\n"
    f"{indent}    \"\"\"Placeholder when FSDP is unavailable in this PyTorch build.\"\"\"\n"
    f"{indent}    def __init__(self, *args, **kwargs):\n"
    f"{indent}        raise RuntimeError(\n"
    f"{indent}            'MMFullyShardedDataParallel(FSDP) is not available: '\n"
    f"{indent}            'this PyTorch wheel lacks torch._C._distributed_c10d.'\n"
    f"{indent}        )\n"
)

txt2 = re.sub(pattern, replacement, txt, count=1)
p.write_text(txt2)
print("Patched:", p)
PY

