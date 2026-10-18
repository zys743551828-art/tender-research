"""用合成 DOCX 验证无网络转换、回退与逐任务隔离；不访问业务资料、模型或数据库。"""

import argparse
import base64
import hashlib
import io
import json
import subprocess
import time
import uuid
import zipfile
from pathlib import Path


def word(body=None, font="Noto Sans CJK SC", external=False):
    if body is None:
        body = f'<w:p><w:r><w:rPr><w:rFonts w:ascii="{font}" w:eastAsia="{font}"/></w:rPr><w:t>施工组织设计，工期180日历天，费用15.12‰。</w:t></w:r></w:p>'
    parts = {
        "[Content_Types].xml": '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>',
        "_rels/.rels": '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="r1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>',
        "word/document.xml": '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>' + body + "</w:body></w:document>",
    }
    if external:
        parts["word/_rels/document.xml.rels"] = (
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="url" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink" Target="https://invalid.example/word" TargetMode="External"/></Relationships>'
        )
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, text in parts.items():
            info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, text.encode())
    return output.getvalue()


DRIVER = r"""
import base64,hashlib,json,sys,time,uuid
from pathlib import Path
payload=json.load(sys.stdin); source=base64.b64decode(payload['data']); identity=uuid.uuid4().hex
profile=json.loads(Path('/word-output/renderer.json').read_text())
request={'source_sha256':hashlib.sha256(source).hexdigest(),'source_bytes':len(source),'filename':'合成验证.docx','renderer_fingerprint':profile['fingerprint']}
request.update(payload.get('overrides',{}))
original=Path('/word-input')/(identity+'.docx'); path=Path('/word-input')/(identity+'.request.json')
original.write_bytes(source); temporary=path.with_suffix('.tmp'); temporary.write_text(json.dumps(request,ensure_ascii=False),encoding='utf-8'); temporary.replace(path)
try:
    deadline=time.monotonic()+100; output=Path('/word-output')/(identity+'.json')
    while not output.exists() and time.monotonic()<deadline: time.sleep(.25)
    report=json.loads(output.read_text()); pdf=Path('/word-output')/(identity+'.pdf')
    assert report['operation_id']==identity
    print(json.dumps({'report':report,'pdf':base64.b64encode(pdf.read_bytes()).decode() if pdf.exists() else None},ensure_ascii=False))
finally:
    path.unlink(missing_ok=True); original.unlink(missing_ok=True)
"""

ISOLATION = r"""
import json,os,tempfile
from pathlib import Path
from landlock import restrict
job=Path(tempfile.mkdtemp(prefix='word-',dir='/work')); other=Path(tempfile.mkdtemp(prefix='word-',dir='/work'))
(job/'own').write_text('own'); (other/'other').write_text('other')
parent=os.getpid(); read,write=os.pipe(); pid=os.fork()
if pid==0:
    os.close(read); restrict(job); result={'own_read':(job/'own').read_text()=='own'}
    for name,path in [('other_job',str(other/'other')),('queue','/word-input'),('parent_app','/app/renderer-manifest.json')]:
        try:
            os.listdir(path) if name=='queue' else Path(path).read_bytes()
            result[name+'_denied']=False
        except PermissionError: result[name+'_denied']=True
    try:
        os.kill(parent,0); result['parent_signal_denied']=False
    except PermissionError: result['parent_signal_denied']=True
    os.write(write,json.dumps(result).encode()); os._exit(0)
os.close(write); data=os.read(read,8192); os.waitpid(pid,0); print(data.decode()); assert all(json.loads(data).values())
"""


CLIENT = r"""
import base64,json,sys
from pathlib import Path
sys.path.insert(0,'/clientcode')
from word_conversion import convert_word
payload=json.load(sys.stdin)
profile=json.loads(Path('/word-output/renderer.json').read_text())
result=convert_word(base64.b64decode(payload['data']),'外部改稿.docx',input_dir=Path('/word-input'),output_dir=Path('/word-output'),renderer_fingerprint=profile['fingerprint'],checkpoint=lambda:None)
print(json.dumps({'report':result.report,'pdf':base64.b64encode(result.pdf).decode() if result.pdf else None},ensure_ascii=False))
assert not list(Path('/word-input').glob('*.request.json'))
"""


def docker(*arguments, data=None, check=True):
    return subprocess.run(["docker", *arguments], input=data, capture_output=True, text=True, encoding="utf-8", timeout=150, check=check)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="bidwb-word:7.4.7")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--client-module", type=Path, help="核对产品客户端时传入 backend/src/bidwb/word_conversion.py")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    identity = "bidwb-word-check-" + uuid.uuid4().hex[:12]
    input_volume, output_volume = identity + "-input", identity + "-output"
    results = []
    try:
        for volume in (input_volume, output_volume):
            docker("volume", "create", volume)
        docker(
            "run",
            "--rm",
            "-d",
            "--name",
            identity,
            "--init",
            "--read-only",
            "--network",
            "none",
            "--user",
            "65534:65534",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges:true",
            "--memory",
            "1g",
            "--cpus",
            "1",
            "--pids-limit",
            "128",
            "--tmpfs",
            "/tmp:ro,noexec,nosuid,nodev,size=1m,mode=0555",
            "--tmpfs",
            "/work:rw,noexec,nosuid,nodev,size=256m,uid=65534,gid=65534,mode=0700",
            "--mount",
            f"type=volume,source={input_volume},target=/word-input,readonly",
            "--mount",
            f"type=volume,source={output_volume},target=/word-output",
            args.image,
        )
        for _ in range(40):
            ready = docker("exec", identity, "python", "/app/service.py", "--health", check=False)
            if ready.returncode == 0:
                break
            time.sleep(0.25)
        assert ready.returncode == 0, ready.stderr
        cases = [
            ("plain-chinese", word(), {}, "converted"),
            ("table", word("<w:tbl><w:tr><w:tc><w:p><w:r><w:t>施工工期180天</w:t></w:r></w:p></w:tc></w:tr></w:tbl>"), {}, "needs_review"),
            ("missing-font", word(font="BidwbMissingFontSynthetic"), {}, "needs_review"),
            ("page-field", word('<w:p><w:fldSimple w:instr="PAGE"/></w:p>'), {}, "failed"),
            ("external-link", word(external=True), {}, "failed"),
            ("invalid-container", b"invalid docx", {}, "failed"),
            ("wrong-sha", word(), {"source_sha256": "0" * 64}, "failed"),
            ("wrong-renderer", word(), {"renderer_fingerprint": "0" * 64}, "failed"),
        ]
        for name, source, overrides, expected in cases:
            response = docker(
                "run",
                "-i",
                "--rm",
                "--network",
                "none",
                "--read-only",
                "--user",
                "10001:10001",
                "--cap-drop",
                "ALL",
                "--security-opt",
                "no-new-privileges:true",
                "--mount",
                f"type=volume,source={input_volume},target=/word-input",
                "--mount",
                f"type=volume,source={output_volume},target=/word-output,readonly",
                "--entrypoint",
                "python",
                args.image,
                "-c",
                DRIVER,
                data=json.dumps({"data": base64.b64encode(source).decode(), "overrides": overrides}),
            )
            result = json.loads(response.stdout)
            report = result["report"]
            results.append({"case": name, **report})
            assert report["state"] == expected, report
            if expected != "failed":
                pdf = base64.b64decode(result["pdf"])
                assert hashlib.sha256(pdf).hexdigest() == report["pdf_sha256"] and len(pdf) == report["pdf_bytes"]
                assert report["source_sha256"] == hashlib.sha256(source).hexdigest()
                (args.output / f"{name}.pdf").write_bytes(pdf)
            else:
                assert not report["conversion_attempted"] and result["pdf"] is None, report
            print(json.dumps({"case": name, "state": report["state"], "gaps": report.get("gaps", [])}, ensure_ascii=False), flush=True)
        if args.client_module:
            client_path = args.client_module.resolve(strict=True).as_posix()
            for name, source in [("client-plain", word()), ("client-field", word('<w:p><w:fldSimple w:instr="PAGE"/></w:p>'))]:
                response = docker(
                    "run",
                    "-i",
                    "--rm",
                    "--network",
                    "none",
                    "--read-only",
                    "--user",
                    "10001:10001",
                    "--cap-drop",
                    "ALL",
                    "--security-opt",
                    "no-new-privileges:true",
                    "--mount",
                    f"type=volume,source={input_volume},target=/word-input",
                    "--mount",
                    f"type=volume,source={output_volume},target=/word-output,readonly",
                    "--mount",
                    f"type=bind,source={client_path},target=/clientcode/word_conversion.py,readonly",
                    "--entrypoint",
                    "python",
                    args.image,
                    "-c",
                    CLIENT,
                    data=json.dumps({"data": base64.b64encode(source).decode()}),
                )
                result = json.loads(response.stdout)
                expected = "converted" if name == "client-plain" else "failed"
                assert result["report"]["state"] == expected, result["report"]
                assert bool(result["pdf"]) == (name == "client-plain")
                results.append({"case": name, **result["report"]})
                print(json.dumps({"case": name, "state": expected}, ensure_ascii=False), flush=True)
        isolation = json.loads(docker("exec", identity, "python", "-c", ISOLATION).stdout)
        assert all(isolation.values())
        invalid = docker("run", "--rm", "--network", "none", args.image, check=False)
        assert invalid.returncode != 0 and "word_container_privileges_invalid" in invalid.stderr
        (args.output / "isolation.json").write_text(json.dumps({**isolation, "unsafe_start_refused": True}, indent=2) + "\n", encoding="utf-8")
    finally:
        (args.output / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        docker("stop", identity, check=False)
        for volume in (input_volume, output_volume):
            docker("volume", "rm", volume, check=False)


if __name__ == "__main__":
    main()
