"""核对产品分词器输入与 Transformers 本地输入，以及 ONNX 输出的动态批量和截断标记。"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tender-app/backend/src"))
from bidwb.embeddings import MAX_TOKENS, MODEL_FINGERPRINT, QUERY_PREFIX, LocalEmbedder  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    runtime = LocalEmbedder(args.model)
    checks = []
    for query in (False, True):
        for texts in (["施工"], ["GB 50300-2013：验收15.12‰", "工期180天"], ["施工组织设计。" * 100]):
            batch = [(QUERY_PREFIX if query else "") + text for text in texts]
            expected = tokenizer(batch, padding=True, truncation=True, max_length=MAX_TOKENS, return_tensors="np")
            actual = runtime._tokenizer.encode_batch(batch)
            for name, field in (("input_ids", "ids"), ("attention_mask", "attention_mask"), ("token_type_ids", "type_ids")):
                np.testing.assert_array_equal(expected[name], np.asarray([getattr(token, field) for token in actual]))
            encoded = runtime.encode(texts, query=query)
            reference = runtime._session.run(["embeddings"], dict(expected))[0]
            np.testing.assert_allclose(encoded.vectors, reference, atol=1e-6)
            norms = np.linalg.norm(np.asarray(encoded.vectors), axis=1)
            np.testing.assert_allclose(norms, 1.0, atol=1e-5)
            checks.append({"query": query, "batch": len(texts), "tokens": int(expected["input_ids"].shape[1]),
                           "truncated": encoded.truncated, "token_parity": True, "output_parity": True, "norms": norms.tolist()})
    assert runtime.encode([]).vectors == []
    output = {"model_fingerprint": MODEL_FINGERPRINT, "checks": checks,
              "packages": {name: importlib.metadata.version(name) for name in ("transformers", "tokenizers", "onnxruntime", "numpy")}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    main()
