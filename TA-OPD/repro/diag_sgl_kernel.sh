#!/usr/bin/env bash
# 诊断 sgl_kernel import 失败的具体原因
# 用法: bash repro/diag_sgl_kernel.sh
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_env.sh"
activate_env

export LD_LIBRARY_PATH="${CONDA_PREFIX:-$(python3 -c 'import sys; print(sys.prefix)')}/lib:${LD_LIBRARY_PATH:-}"

python3 << 'PYEOF'
import sys, os, traceback, subprocess, pathlib, importlib.util

print("=== Python & LD_LIBRARY_PATH ===")
print(f"Python: {sys.executable}")
print("LD_LIBRARY_PATH:")
for p in os.environ.get("LD_LIBRARY_PATH", "").split(":"):
    if p: print(f"  {p}")
print()

# 1. torch ABI
import torch
print(f"=== torch ===")
print(f"version: {torch.__version__}")
print(f"_GLIBCXX_USE_CXX11_ABI: {torch._C._GLIBCXX_USE_CXX11_ABI}")
print(f"cuda: {torch.version.cuda}")
print()

# 2. sgl-kernel 包元信息
print("=== pip show sgl-kernel ===")
r = subprocess.run(["pip", "show", "sgl-kernel"], capture_output=True, text=True)
print(r.stdout or "(not installed)")

# 3. import sgl_kernel
print("=== import sgl_kernel ===")
try:
    import sgl_kernel
    print(f"OK: {sgl_kernel.__file__}")
except Exception:
    traceback.print_exc()
print()

# 4. 看 so 文件 + ldd
print("=== sgl_kernel 的 .so 文件 (ldd) ===")
try:
    spec = importlib.util.find_spec("sgl_kernel")
    if spec and spec.origin:
        so_dir = pathlib.Path(spec.origin).parent
        for f in sorted(so_dir.rglob("*.so")):
            print(f"\n[{f}]")
            out = subprocess.run(["ldd", str(f)], capture_output=True, text=True).stdout
            for line in out.splitlines():
                if "not found" in line or "=>" in line or "linux-vdso" in line:
                    print(f"  {line.strip()}")
    else:
        print("sgl_kernel 没找到 spec (包可能根本没装上)")
except Exception:
    traceback.print_exc()

# 5. 系统 vs conda libstdc++
print()
print("=== libstdc++.so.6 GLIBCXX 版本对比 ===")
for label, path in [
    ("system", "/lib/x86_64-linux-gnu/libstdc++.so.6"),
    ("conda",  f"{os.environ.get('CONDA_PREFIX', '')}/lib/libstdc++.so.6"),
]:
    if not os.path.exists(path):
        print(f"{label}: {path} (不存在)")
        continue
    r = subprocess.run(["strings", path], capture_output=True, text=True)
    versions = sorted([l for l in r.stdout.splitlines() if l.startswith("GLIBCXX_")])
    print(f"{label}: {path}")
    print(f"  最高: {versions[-3:] if versions else '(无)'}")
PYEOF

echo ""
echo "========================================="
echo " 把上面所有输出贴给 QoderCN, 我帮你定位根因。"
echo "========================================="
