"""对固定物理页作本地解析探针；锚点命中不代表表格、图意或全文质量通过。"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import time
import traceback
import unicodedata
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalize(text: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text))


def without_embedded_images(text: str) -> str:
    """内嵌图片不是文字；原始 JSON 仍保存图片引用供版面核对。"""
    return re.sub(r"<img\b[^>]*>", "[image]", text, flags=re.IGNORECASE)


def configure_cache(root: Path) -> None:
    for key, child in {
        "HF_HOME": "huggingface", "TORCH_HOME": "torch", "XDG_CACHE_HOME": "cache",
        "MODELSCOPE_CACHE": "modelscope", "MINERU_HOME": "mineru", "PADDLE_PDX_CACHE_HOME": "paddlex",
    }.items():
        os.environ[key] = str(root / child)
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["MINERU_MODEL_SMALL_BACKEND"] = "onnx"
    os.environ["MINERU_MODEL_SOURCE"] = "local"
    os.environ["OMP_NUM_THREADS"] = "4"


def adapter(engine: str):
    if engine == "pdfium":
        import pypdfium2 as pdfium

        def parse(path, number, raw):
            with pdfium.PdfDocument(path) as pdf, closing(pdf[number - 1]) as page, closing(page.get_textpage()) as textpage:
                return textpage.get_text_range(), {"table_structure": "unsupported", "figure_semantics": "unsupported"}
        return parse
    if engine == "docling":
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import OcrMode, PdfPipelineOptions, RapidOcrOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption

        options = PdfPipelineOptions()
        options.do_ocr = True
        options.do_table_structure = True
        options.enable_remote_services = False
        options.do_picture_description = False
        options.ocr_options = RapidOcrOptions(lang=["ch"], mode=OcrMode.FULL_PAGE)
        converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)})

        def parse(path, number, raw):
            result = converter.convert(path, page_range=(number, number))
            write_json(raw, result.document.export_to_dict())
            return result.document.export_to_markdown(), {"conversion_status": str(result.status), "figure_semantics": "unverified"}
        return parse
    if engine == "rapidocr":
        from rapidocr import RapidOCR

        pipeline = RapidOCR(params={"Global.model_root_dir": str(Path(os.environ["XDG_CACHE_HOME"]) / "rapidocr")})

        def parse(path, number, raw):
            import pypdfium2 as pdfium

            with pdfium.PdfDocument(path) as pdf, closing(pdf[number - 1]) as page:
                result = pipeline(page.render(scale=2).to_pil())
            write_json(raw, {"texts": list(result.txts or []),
                             "boxes": result.boxes.tolist() if result.boxes is not None else [],
                             "scores": result.scores.tolist() if hasattr(result.scores, "tolist") else result.scores})
            return "\n".join(result.txts or []), {"input": "144dpi_render", "table_structure": "unsupported", "figure_semantics": "unsupported"}
        return parse
    if engine == "paddle":
        from paddleocr import PPStructureV3

        pipeline = PPStructureV3(
            device="cpu", enable_mkldnn=False, use_doc_orientation_classify=False, use_doc_unwarping=False,
            use_textline_orientation=False, use_formula_recognition=False, use_chart_recognition=False,
            use_seal_recognition=False, use_table_recognition=True,
        )

        def parse(path, number, raw):
            import pypdfium2 as pdfium

            image = raw.with_suffix(".png")
            with pdfium.PdfDocument(path) as pdf, closing(pdf[number - 1]) as page:
                page.render(scale=2).to_pil().save(image)
            results = list(pipeline.predict(str(image)))
            payload = [result.json for result in results]
            write_json(raw, payload)
            text = "\n".join(result.markdown["markdown_texts"] for result in results)
            return text, {"input": "144dpi_render", "enable_mkldnn": False, "figure_semantics": "unverified"}
        return parse
    if engine == "mineru":
        executable = Path(sys.executable).parent / ("mineru-kit.exe" if os.name == "nt" else "mineru-kit")

        def parse(path, number, raw):
            with raw.with_suffix(".log").open("w", encoding="utf-8") as log:
                subprocess.run([
                    str(executable), "parse", str(path), "-o", str(raw), "--pages", str(number),
                    "--format", "middle_json", "--tier", "basic", "--disable-image-analysis",
                ], stdout=log, stderr=subprocess.STDOUT, check=True, timeout=300)
            payload = json.loads(raw.read_text(encoding="utf-8"))
            # Only content strings count as text; model metadata and coordinates cannot satisfy anchors.
            texts = []

            def visit(node):
                if isinstance(node, dict):
                    for key, value in node.items():
                        if key in {"content", "text", "html"} and isinstance(value, str):
                            texts.append(value)
                        else:
                            visit(value)
                elif isinstance(node, list):
                    for item in node:
                        visit(item)
            visit(payload)
            return "\n".join(texts), {"tier": "basic", "figure_semantics": "unverified"}
        return parse
    raise ValueError(engine)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=["pdfium", "docling", "paddle", "mineru", "rapidocr"], required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--datasets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--sample", help="仅跑一个样本 ID，用于故障定位")
    args = parser.parse_args()
    configure_cache(args.cache.resolve())
    output = args.output.resolve() / args.engine
    output.mkdir(parents=True, exist_ok=True)
    protocol_bytes = args.protocol.read_bytes()
    protocol = json.loads(protocol_bytes)
    samples = [s for s in protocol["samples"] if args.sample is None or s["id"] == args.sample]
    if not samples:
        raise ValueError("No matching samples")
    # Verify all source bytes before loading/downloading models.
    verified = {}
    for sample in samples:
        path = args.datasets.resolve() / sample["source_path"]
        if path not in verified:
            verified[path] = hashlib.sha256(path.read_bytes()).hexdigest()
        if verified[path] != sample["source_sha256"]:
            raise ValueError(f"Source hash mismatch: {sample['id']}")
    started_at = datetime.now(UTC).isoformat()
    start = time.perf_counter()
    init_error = None
    try:
        parse = adapter(args.engine)
    except Exception as exc:  # noqa: BLE001 -- 第三方引擎异常作为本轮可审计失败，不吞掉或冒充成功。
        init_error = f"{type(exc).__name__}: {exc}"
        (output / "initialization-error.log").write_text(traceback.format_exc(), encoding="utf-8")
    summary = {
        "schema": "parsing-probe-result-1", "engine": args.engine,
        "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "initialization_seconds": round(time.perf_counter() - start, 3),
        "initialization_error": init_error,
        "started_at": started_at, "platform": platform.platform(),
        "python": sys.version, "cache": str(args.cache.resolve()),
        "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        "scope": protocol["scope"], "samples": [],
    }
    write_json(output / "summary.json", summary)
    for sample in samples:
        started = time.perf_counter()
        row = {"id": sample["id"], "physical_page": sample["physical_page"], "status": "failed",
               "structure_review": "not_evaluated", "figure_semantics_review": sample["figure_semantics_review"]}
        try:
            if init_error:
                raise RuntimeError(init_error)
            text, details = parse(args.datasets.resolve() / sample["source_path"], sample["physical_page"], output / f"{sample['id']}.json")
            text = without_embedded_images(text)
            (output / f"{sample['id']}.md").write_text(text, encoding="utf-8")
            row.update(status="parsed", chars=len(text), details=details,
                       anchors=[{"expected": anchor, "found": normalize(anchor) in normalize(text)} for anchor in sample["anchors"]])
        except Exception as exc:  # noqa: BLE001 -- 一页失败不能抹去其他页的实测记录。
            row["error"] = f"{type(exc).__name__}: {exc}"
            (output / f"{sample['id']}-error.log").write_text(traceback.format_exc(), encoding="utf-8")
        row["seconds"] = round(time.perf_counter() - started, 3)
        summary["samples"].append(row)
        summary["total_seconds"] = round(time.perf_counter() - start, 3)
        write_json(output / "summary.json", summary)
        print(json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
