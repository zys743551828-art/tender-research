"""将本地安全张量 BGE 模型导出为归一化 CLS ONNX，并与 CPU 原模型逐项对照。"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoModel, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    root = args.model.resolve()
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True, trust_remote_code=False)
    encoder = AutoModel.from_pretrained(root, local_files_only=True, trust_remote_code=False, use_safetensors=True, attn_implementation="eager")

    class Embedding(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = encoder

        def forward(self, input_ids, attention_mask, token_type_ids):
            hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=False)[0]
            return torch.nn.functional.normalize(hidden[:, 0], p=2, dim=1)

    model = Embedding().eval()
    names = ["input_ids", "attention_mask", "token_type_ids"]
    example = tokenizer(["施工组织设计", "为这个句子生成表示以用于检索相关文章：管网工期120天"], padding=True, return_tensors="pt")
    output = root / "model.onnx"
    with torch.no_grad():
        torch.onnx.export(model, tuple(example[n] for n in names), str(output), input_names=names, output_names=["embeddings"],
                          dynamic_axes={**{n: {0: "batch", 1: "sequence"} for n in names}, "embeddings": {0: "batch"}},
                          opset_version=17, dynamo=False, external_data=False)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 2
    session = ort.InferenceSession(str(output), sess_options=options, providers=["CPUExecutionProvider"])
    checks = []
    for texts in [["施工"], ["工期120天", "GB 50300-2013 验收"], ["施工组织设计。" * 100]]:
        tokens = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            expected = model(*(tokens[n] for n in names)).numpy()
        actual = session.run(None, {n: tokens[n].numpy() for n in names})[0]
        error = float(np.max(np.abs(expected - actual)))
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-4)
        checks.append({"batch": len(texts), "tokens": int(tokens["input_ids"].shape[1]), "max_abs_error": error,
                       "dimensions": int(actual.shape[1]), "norms": np.linalg.norm(actual, axis=1).tolist()})
    files = ["model.safetensors", "model.onnx", "tokenizer.json", "tokenizer_config.json", "config.json", "vocab.txt"]
    manifest = {"model_id": "BAAI/bge-small-zh-v1.5", "source": "AI-ModelScope/bge-small-zh-v1.5", "dimensions": 512,
                "pooling": "CLS", "normalize": True, "max_tokens": 512, "query_prefix": "为这个句子生成表示以用于检索相关文章：",
                "files": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in files},
                "conversion_checks": checks, "packages": {n: importlib.metadata.version(n) for n in ["torch", "transformers", "tokenizers", "onnx", "onnxruntime"]}}
    (root / "embedding-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"model": str(output), "checks": checks}, ensure_ascii=False))


if __name__ == "__main__":
    main()
