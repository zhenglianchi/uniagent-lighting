#!/usr/bin/env bash
# 修复 Python 3.10 下 verl 的 StrEnum 兼容问题（UCloud 版）
# 用法：bash /home/ubuntu/fix_strenum_ucloud.sh
set -e

E=/home/ubuntu/miniforge3/envs/swe-rl
"$E/bin/pip" install -q StrEnum 2>&1 | tail -1 || true

"$E/bin/python" - <<'PY'
import pathlib

old = "from enum import StrEnum"
new = "try:\n    from enum import StrEnum\nexcept ImportError:\n    from strenum import StrEnum"
count = 0
for p in pathlib.Path("/home/ubuntu/uni-agent/verl/verl").rglob("*.py"):
    s = p.read_text()
    if old in s:
        p.write_text(s.replace(old, new))
        print("patched:", p)
        count += 1
print("total patched:", count)
PY

"$E/bin/python" -c "import verl.utils.tokenizer.continuous_token_wiring; print('StrEnum 补丁 OK')" 2>&1 | tail -1
