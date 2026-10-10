"""扫描页 OCR 对照；模型本机推理，引用保留像素坐标且须复核。"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import tempfile
import time
from pathlib import Path

import pdfplumber
import pypdfium2

from . import paths
from .rule_evidence import normalized


def run(datasets: Path, output: Path):
    import rapidocr
    from rapidocr import RapidOCR
    start = time.perf_counter()
    manifest = json.loads((datasets/"公开规则资料/资料清单.json").read_text())
    cases = [("PUB-004",26,["造价咨询工作服务费收费标准","15.12万元","10.8万元","2.16万元"]),
             ("PUB-008",88,["40%","30%","工程重点难点保障体系"]),
             ("PUB-012",3,[])]
    engine = RapidOCR()
    results = []
    with tempfile.TemporaryDirectory(prefix="ocr-probe-") as tmp:
        for fid,page_no,anchors in cases:
            row = next(r for r in manifest["file_records"] if r["file_id"] == fid)
            path = paths.original(datasets/"公开规则资料", row["relative_path"])
            if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
                raise ValueError("OCR 原件校验失败")
            pdf = pypdfium2.PdfDocument(path)
            image = pdf[page_no-1].render(scale=2).to_pil(); image_path = Path(tmp)/f"{fid}-{page_no}.png"
            image.save(image_path); pdf.close()
            began = time.perf_counter(); prediction = engine(str(image_path))
            elapsed = time.perf_counter()-began
            texts = list(prediction.txts) if prediction.txts is not None else []
            boxes = prediction.boxes.tolist() if prediction.boxes is not None else []
            scores = [float(x) for x in prediction.scores] if prediction.scores is not None else []
            with pdfplumber.open(path) as document:
                p = document.pages[page_no-1]; plain = p.crop(p.bbox).extract_text() or ""
            text = normalized("".join(texts))
            results.append({"file_id":fid,"source_sha256":row["sha256"],"pdf_page":page_no,
                            "image_sha256":hashlib.sha256(image_path.read_bytes()).hexdigest(),
                            "image_dimensions":list(image.size),"render_scale":2,"crop_text":plain,
                            "ocr_lines":[{"text":t,"bbox_pixels_top_left":b,"confidence":c,"kind":"ocr_excerpt_needs_review"} for t,b,c in zip(texts,boxes,scores)],
                            "anchor_checks":[{"anchor":a,"ocr_contains":normalized(a) in text,"crop_contains":normalized(a) in normalized(plain)} for a in anchors],
                            "blank_page_no_ocr_text":not texts if not anchors else None,"elapsed_seconds":round(elapsed,3)})
    model_dir = Path(rapidocr.__file__).parent
    result = {"experiment_id":"ocr-probe","adapter":"rapidocr-onnx-cpu", "evaluation_kind":"three_known_page_probes_not_ocr_accuracy_benchmark",
              "runtime":{"platform":platform.platform(),"packages":{n:importlib.metadata.version(n) for n in ("rapidocr","onnxruntime","numpy","opencv_python")}},
              "model_sha256":{str(p.relative_to(model_dir)):hashlib.sha256(p.read_bytes()).hexdigest() for p in model_dir.rglob("*.onnx")},
              "pages":results,"elapsed_seconds":round(time.perf_counter()-start,3),
              "limits":["人工挑选少量锚点，不代表字错率或表格结构准确率", "OCR文字和数字仍需复核", "本实验没有向外部OCR服务发送项目文件"]}
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"pages":[{"file_id":p["file_id"],"page":p["pdf_page"],"lines":len(p["ocr_lines"]),"checks":p["anchor_checks"],"blank":p["blank_page_no_ocr_text"]} for p in results]},ensure_ascii=False))
    return result


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets",type=Path,default=paths.DATASETS)
    p.add_argument("--output",type=Path,default=paths.RESULTS/"ocr-probe.json")
    a=p.parse_args();run(a.datasets,a.output)
