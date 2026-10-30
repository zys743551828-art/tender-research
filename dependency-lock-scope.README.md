# Base dependency lock scope

`requirements-base.lock.txt` targets **CPython 3.11, Linux amd64, glibc** and
locks the complete transitive closure of the six exact dependencies in
`pyproject.toml` `[project].dependencies`. Each selected wheel has a SHA256
obtained from real PyPI download metadata. It is not a cross-platform lock.

Resolution was performed on 2026-10-30 with Python 3.11.18 in official Python
Linux amd64 image manifest
`sha256:061c190f3d5815b341a18dfe8a45771f8ba68c15a607ea56997fde3009ac4eec`.
Metadata source: `https://pypi.org/simple`; wheel source:
`https://files.pythonhosted.org/`.

Install in a fresh Linux/Python 3.11 environment:

```sh
python -m pip --isolated install --require-hashes -r requirements-base.lock.txt
python -m pip check
```

The lock excludes the `dev` and `store` extras, build tooling, and the separate
`requirements-ocr.txt` and `requirements-orchestration.txt` environments.
Those files do not constitute one jointly validated installation: OCR pins
`numpy==2.5.3`, while base pins `numpy==2.3.5`; orchestration pins
`pydantic==2.13.3`, while base pins `pydantic==2.13.5`. Installing those exact
declarations together would conflict. This task preserves their declarations
and does not guess a reconciled version.

The new base lock is not yet wired into CI or an installation entrypoint.
Hash installation and `pip check` establish package availability and dependency
consistency; they do not establish research workflow, OCR, orchestration,
model, or production acceptance. No model was invoked.

Evidence remains in the isolated workspace at
`../evidence-local-20261030/dependency-locks/`: environment, resolver inputs,
`research-base-report.json`, resolve/install/check logs, `status.json`, and the
Docker exact-resource cleanup receipt. The official Python image is retained
for dependency verification. No global packages or existing source declarations
were changed.
