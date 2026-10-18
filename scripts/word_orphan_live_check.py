"""真实转换客户端异常退出后，验证空闲 worker 回收队列及转换器输出回收。"""

import argparse
import atexit
import base64
import hashlib
import json
import subprocess
import time
import uuid
from pathlib import Path

from word_converter_check import word


def run(container, source, user=None, expected=0):
    command = ["docker", "exec", "-i"]
    if user:
        command += ["--user", user]
    command += [container, "python", "-"]
    result = subprocess.run(command, input=source, capture_output=True, text=True, encoding="utf-8", timeout=90)
    if result.returncode != expected:
        raise RuntimeError({"exit": result.returncode, "stderr": result.stderr[-1000:]})
    return result.stdout.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("使用新的输出路径")
    identities = [uuid.uuid4().hex for _ in range(3)]
    identity, fresh, linked = identities
    data = word()
    encoded = base64.b64encode(data).decode()
    worker = "bidwb-worker-parse-1"
    converter = "bidwb-word-converter-1"
    def cleanup_own_test_files():
        try:
            run(worker, f'''from pathlib import Path
for identity in {identities!r}:
    for suffix in ('docx','request.json','request.tmp'):
        (Path('/word-input')/(identity+'.'+suffix)).unlink(missing_ok=True)
Path('/tmp/{linked}-sentinel').unlink(missing_ok=True)
''')
        except RuntimeError:
            pass
    atexit.register(cleanup_own_test_files)
    crash = f'''
import base64, os, uuid
from pathlib import Path
from types import SimpleNamespace
from bidwb.word_conversion import convert_word
uuid.uuid4 = lambda: SimpleNamespace(hex={identity!r})
def checkpoint():
    if Path('/word-input/{identity}.request.json').exists():
        os._exit(23)
convert_word(base64.b64decode({encoded!r}), 'synthetic-orphan.docx', input_dir=Path('/word-input'), output_dir=Path('/word-output'), renderer_fingerprint=os.environ['BIDWB_WORD_RENDERER_FINGERPRINT'], checkpoint=checkpoint)
'''
    run(worker, crash, expected=23)
    print("客户端在发布请求后异常退出，等待转换回执。", flush=True)
    read = f"from pathlib import Path; print((Path('/word-output') / '{identity}.json').exists())"
    deadline = time.monotonic() + 75
    while time.monotonic() < deadline:
        if run(worker, read) == "True":
            break
        time.sleep(0.5)
    else:
        raise AssertionError("转换回执未就绪")
    prepare = f'''
import os, time
from pathlib import Path
from bidwb.word_conversion import ORPHAN_INPUT_SECONDS
root=Path('/word-input')
for suffix in ('docx','request.json'):
    p=root / ('{identity}.'+suffix)
    assert p.exists()
    os.utime(p,(time.time()-ORPHAN_INPUT_SECONDS-60,)*2)
for suffix in ('docx','request.tmp'):
    (root / ('{fresh}.'+suffix)).write_bytes(b'fresh synthetic operation')
(root / '{linked}.docx').write_bytes(b'old synthetic source')
os.utime(root / '{linked}.docx',(time.time()-ORPHAN_INPUT_SECONDS-60,)*2)
target=Path('/tmp/{linked}-sentinel')
target.write_bytes(b'outside synthetic sentinel')
(root / '{linked}.request.json').symlink_to(target)
print('prepared')
'''
    run(worker, prepare)
    print("仅将本次合成遗留输入的时间戳提前，等待空闲 worker 的周期回收。", flush=True)
    absent = f"from pathlib import Path; print(not any(Path('/word-input').glob('{identity}.*')))"
    deadline = time.monotonic() + 75
    while time.monotonic() < deadline:
        if run(worker, absent) == "True":
            break
        time.sleep(0.5)
    else:
        raise AssertionError("空闲 worker 未回收旧输入")
    verify = f'''
from pathlib import Path
root=Path('/word-input')
assert (root/'{fresh}.docx').exists() and (root/'{fresh}.request.tmp').exists()
assert (root/'{linked}.docx').exists() and (root/'{linked}.request.json').is_symlink()
assert Path('/tmp/{linked}-sentinel').read_bytes()==b'outside synthetic sentinel'
for identity in {identities!r}:
    for suffix in ('docx','request.json','request.tmp'):
        (root/(identity+'.'+suffix)).unlink(missing_ok=True)
Path('/tmp/{linked}-sentinel').unlink()
print('fresh_and_nonregular_preserved')
'''
    run(worker, verify)
    run(converter, f'''
import os,time
from pathlib import Path
for identity in {identities!r}:
    for suffix in ('json','pdf'):
        path=Path('/word-output')/(identity+'.'+suffix)
        if path.exists(): os.utime(path,(time.time()-180,)*2)
''', user="65534")
    deadline = time.monotonic() + 10
    empty = f"from pathlib import Path; print(not any((Path('/word-output')/(i+'.'+s)).exists() for i in {identities!r} for s in ('json','pdf')))"
    while time.monotonic() < deadline:
        if run(worker, empty) == "True":
            break
        time.sleep(0.5)
    else:
        raise AssertionError("输出未回收")
    result = {"synthetic": True, "model_calls": 0, "child_process_exit": 23, "source_sha256": hashlib.sha256(data).hexdigest(), "accelerated_file_timestamps": True, "idle_worker_reclaimed_inputs": True, "fresh_group_preserved": True, "nonregular_group_preserved": True, "outside_sentinel_preserved": True, "converter_reclaimed_outputs": True, "operation_ids": identities}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
