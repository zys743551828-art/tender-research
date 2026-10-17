"""批量入库测试；PDF、企业与分数均为合成夹具，只证明程序行为。"""
from __future__ import annotations

import csv
import hashlib
import json
import random
from pathlib import Path

import pytest

from tender_research import intake
from tender_research.intake import COLUMNS, TAIL_COLUMNS, InputError, run


def make_pdf(path: Path, pages: list[str]):
    """手写最小 PDF：Helvetica 文本，每页一段；空字符串的页没有文字层。"""
    objects = ["<< /Type /Catalog /Pages 2 0 R >>", None, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    kids = []
    for text in pages:
        lines = [text[i:i + 90] for i in range(0, len(text), 90)]
        ops = "".join(f"({line.replace(chr(92), '').replace('(', '').replace(')', '')}) Tj T* " for line in lines)
        stream = f"BT /F1 9 Tf 12 TL 30 800 Td {ops}ET" if lines else "0 0 m 10 10 l S"
        objects.append(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")
        objects.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Contents {len(objects)} 0 R "
                       "/Resources << /Font << /F1 3 0 R >> >> >>")
        kids.append(f"{len(objects)} 0 R")
    objects[1] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(kids)} >>"
    out, offsets = bytearray(b"%PDF-1.4\n"), []
    for i, obj in enumerate(objects, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n{obj}\nendobj\n".encode("latin-1")
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += "".join(f"{o:010d} 00000 n \n" for o in offsets).encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(out))


def words(seed: int, n: int) -> list[str]:
    rng = random.Random(seed)
    return ["".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(rng.randint(3, 8))) for _ in range(n)]


def body_pages(seed: int, pages: int = 3, mutate: float = 0.0) -> list[str]:
    """合成正文；mutate>0 时在固定种子文本上替换一部分词，模拟同一模板改写。"""
    base = words(seed, 120 * pages)
    rng = random.Random(seed + 999)
    text = [w if rng.random() >= mutate else w[::-1] + "x" for w in base]
    return [" ".join(text[i * 120:(i + 1) * 120]) for i in range(pages)]


def row(**overrides) -> dict:
    base = {"样本编号": "", "项目名称": "合肥市示例路道路工程", "招标编号": "TEST-A", "标段": "", "地区": "安徽省合肥市",
            "工程类型": "市政道路", "评分办法": "综合评估法", "投标企业": "示例甲市政工程有限公司", "本公司": "是",
            "提交日期": "2026-04-20", "招标文件": "A招标.pdf", "技术标正文": "A甲.pdf", "技术分": "52.3",
            "技术满分": "60", "成绩口径": "技术总分", "聚合方式": "评委平均", "成绩来源文件": "", "成绩来源页码": "",
            "成绩来源类型": "公司台账", "结果": "入围", "成绩取得日期": "2026-05-12", "备注": "",
            "同场技术分": "", "入围分数线": ""}
    base.update(overrides)
    return base


def write_sheet(path: Path, rows: list[dict], tail: bool = True):
    columns = list(COLUMNS) + (list(TAIL_COLUMNS) if tail else [])
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(columns)
        for r in rows:
            w.writerow([r.get(c, "") for c in columns])


def tree_digest(root: Path) -> dict:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


@pytest.fixture
def env(tmp_path, monkeypatch):
    """两个项目：A 项目 3 家、B 项目 2 家；库中已有 1.2 版 CASE-001 与一个 CASE-004 目录。原件缓存在 tmp 下。"""
    inbox, datasets = tmp_path / "收件", tmp_path / "datasets"
    monkeypatch.setenv("TENDER_DATASETS_CACHE", str(tmp_path / "cache"))
    for name in ("ENDPOINT", "REGION", "BUCKET", "ACCESS_KEY_ID", "SECRET_ACCESS_KEY"):
        monkeypatch.delenv(f"TENDER_STORE_{name}", raising=False)
        monkeypatch.delenv(f"TENDER_STORE_{name}_FILE", raising=False)
    make_pdf(inbox / "A招标.pdf", ["tender A rules " + " ".join(words(1, 200))])
    make_pdf(inbox / "A补遗一.pdf", ["addendum A " + " ".join(words(2, 60))])
    make_pdf(inbox / "B招标.pdf", ["tender B rules " + " ".join(words(3, 200))])
    make_pdf(inbox / "评分表A.pdf", ["score table " + " ".join(words(4, 60))])
    for name, seed in (("A甲.pdf", 10), ("A乙.pdf", 20), ("sub/A丙.pdf", 30), ("B甲.pdf", 40), ("B丁.pdf", 50)):
        make_pdf(inbox / name, body_pages(seed))
    legacy = datasets / "CASE-001_广德_污水管网改造及新建"
    legacy.mkdir(parents=True)
    (legacy / "资料清单.json").write_text(json.dumps({"schema_version": "1.2", "case_id": "CASE-001"}), encoding="utf-8")
    (datasets / "CASE-004_合肥_旧样本").mkdir()
    (datasets / "README.md").write_text("# 案例库\n", encoding="utf-8")
    rows = [
        row(招标文件="A招标.pdf;A补遗一.pdf", 成绩来源文件="评分表A.pdf", 成绩来源页码="3", 成绩来源类型="官方原件",
            同场技术分="52.3;48;/", 入围分数线="50"),
        row(投标企业="示例乙建设有限公司", 本公司="否", 技术标正文="A乙.pdf", 技术分="48", 结果="未入围",
            招标文件="A招标.pdf;A补遗一.pdf"),
        row(投标企业="示例丙集团股份有限公司", 本公司="否", 技术标正文="A丙.pdf", 技术分="/", 结果="",
            招标文件="A招标.pdf;A补遗一.pdf"),
        row(项目名称="合肥市示例管网（二期）改造项目", 招标编号="TEST-B", 标段="二标段", 工程类型="排水管网",
            评分办法="技术入围后价格竞争", 招标文件="B招标.pdf", 技术标正文="B甲.pdf", 技术分="41", 技术满分="50",
            提交日期="2026/6/1", 成绩取得日期=""),
        row(项目名称="合肥市示例管网（二期）改造项目", 招标编号="TEST-B", 标段="二标段", 工程类型="排水管网",
            评分办法="技术入围后价格竞争", 招标文件="b招标.PDF", 技术标正文="B丁", 投标企业="示例丁工程有限公司",
            本公司="否", 技术分="39.5", 技术满分="50", 提交日期="2026-06-01", 成绩取得日期=""),
    ]
    sheet = tmp_path / "样本登记表.csv"
    write_sheet(sheet, rows)
    return {"inbox": inbox, "datasets": datasets, "sheet": sheet, "rows": rows, "tmp": tmp_path,
            "cache": tmp_path / "cache"}


def rerun(env, rows, **kw):
    write_sheet(env["sheet"], rows)
    return run(env["sheet"], env["inbox"], env["datasets"], **kw)


def pairs(env) -> dict:
    return json.loads((env["datasets"] / "评分配对集.json").read_text(encoding="utf-8"))


def test_happy_path_two_projects(env):
    inbox_before = tree_digest(env["inbox"])
    legacy_before = tree_digest(env["datasets"] / "CASE-001_广德_污水管网改造及新建")
    out = run(env["sheet"], env["inbox"], env["datasets"])
    assert [r.status for r in out.rows] == ["new"] * 5
    assert [r.sample_id for r in out.rows] == ["CASE-005", "CASE-006", "CASE-007", "CASE-008", "CASE-009"]
    assert tree_digest(env["inbox"]) == inbox_before
    assert tree_digest(env["datasets"] / "CASE-001_广德_污水管网改造及新建") == legacy_before
    assert (env["datasets"] / "README.md").read_text(encoding="utf-8") == "# 案例库\n"

    case = env["datasets"] / "CASE-005_合肥_示例路道路"
    cached = env["cache"] / "CASE-005_合肥_示例路道路"
    assert sorted(p.name for p in case.rglob("*")) == ["README.md", "资料清单.json"]   # 仓库不含原件
    names = sorted(p.relative_to(cached).as_posix() for p in cached.rglob("*") if p.is_file())
    assert names == ["01-招标资料/CASE-005_招标文件_日期未注明.pdf", "01-招标资料/CASE-005_补遗1_日期未注明.pdf",
                     "02-投标文件/CASE-005_技术标_示例甲市政_20260420.pdf", "03-评审结果/CASE-005_成绩来源_20260512.pdf"]
    assert not (env["cache"] / "CASE-006_合肥_示例路道路" / "03-评审结果").exists()
    m = json.loads((case / "资料清单.json").read_text(encoding="utf-8"))
    assert m["schema_version"] == "1.3" and m["assessment"]["reported_results"] == []
    assert [r["role"] for r in m["file_records"]] == ["tender", "addendum", "body", "score_source"]
    assert m["file_records"][0]["original_filename"] == "A招标.pdf"
    body_sha = m["file_records"][2]["sha256"]
    assert m["file_records"][2]["storage"] == {"key": f"originals/sha256/{body_sha[:2]}/{body_sha}.pdf"}
    assert m["file_records"][2]["minhash"]["scheme"] == intake.MINHASH_SCHEME
    off = m["assessment"]["official_results"][0]
    assert off == {"result_id": "CASE-005-OFF-1", "target": "technical_total", "value": 52.3, "maximum": 60.0,
                   "aggregation": "评委平均", "source_kind": "official_document", "source_file_id": "FILE-004",
                   "source_pdf_page": 3, "known_at": "2026-05-12", "stage_result": "shortlisted",
                   "verification_status": "verified", "submission_file_id": "FILE-003",
                   "submission_sha256": m["file_records"][2]["sha256"]}
    assert m["file_records"][2]["sha256"] == hashlib.sha256((env["inbox"] / "A甲.pdf").read_bytes()).hexdigest()
    assert "CASE-005｜合肥示例路道路" in (case / "README.md").read_text(encoding="utf-8")

    p = pairs(env)
    assert p["schema"] == "score-pairs/1" and len(p["samples"]) == 5
    by_id = {s["sample_id"]: s for s in p["samples"]}
    a, b = by_id["CASE-005"], by_id["CASE-008"]
    assert a["project_key"] == "TEST-A|" and b["project_key"] == "TEST-B|二标段"
    assert {by_id[i]["template_group"] for i in ("CASE-005", "CASE-006", "CASE-007")} == {a["template_group"]}
    assert by_id["CASE-009"]["template_group"] == b["template_group"] != a["template_group"]
    assert [f["role"] for f in a["tender_files"]] == ["tender", "addendum"]
    assert a["body_file"]["text_layer"] is True and a["body_file"]["pages"] == 3
    assert p["datasets_root"] == str(env["cache"].resolve())
    assert (Path(p["datasets_root"]) / a["body_file"]["path"]).is_file()
    assert a["body_file"]["storage_key"] == m["file_records"][2]["storage"]["key"]
    assert a["tender_files"][0]["storage_key"] == m["file_records"][0]["storage"]["key"]
    assert a["label"]["value"] == 52.3 and a["eligible"] and a["exclusion"] is None
    assert by_id["CASE-006"]["label"]["verification_status"] == "pending"
    assert by_id["CASE-007"]["label"] is None and by_id["CASE-007"]["exclusion"] == "missing_label"
    assert b["submitted_at"] == "2026-06-01" and b["scoring_scheme"] == "技术入围后价格竞争"
    assert p["projects"]["TEST-A|"]["field_scores"] == [52.3, 48.0, None]
    assert p["projects"]["TEST-A|"]["shortlist_cutoff"] == 50.0
    assert p["projects"]["TEST-A|"]["field_source"]["source_file_id"] == "FILE-004"
    assert p["projects"]["TEST-B|二标段"]["field_scores"] is None
    report = (env["datasets"] / "入库报告.md").read_text(encoding="utf-8")
    assert "| 5 | 5 | 5 | 0 | 0 |" in report and "CASE-001_广德_污水管网改造及新建" in report
    assert "待上传" in report and m["file_records"][2]["storage"]["key"] in report


def test_idempotent_rerun(env):
    run(env["sheet"], env["inbox"], env["datasets"])
    first = tree_digest(env["datasets"])
    out = run(env["sheet"], env["inbox"], env["datasets"])
    assert [r.status for r in out.rows] == ["skipped"] * 5
    assert [r.sample_id for r in out.rows] == ["CASE-005", "CASE-006", "CASE-007", "CASE-008", "CASE-009"]
    second = tree_digest(env["datasets"])
    changed = {k for k in first.keys() | second.keys() if first.get(k) != second.get(k)}
    assert changed <= {"评分配对集.json", "入库报告.md"}
    samples = pairs(env)["samples"]
    rows = [row(**r) for r in env["rows"]]
    rows[0]["样本编号"] = "CASE-005"   # 回填编号后同样跳过
    out = rerun(env, rows)
    assert out.rows[0].status == "skipped" and pairs(env)["samples"] == samples


def test_dry_run_writes_nothing(env, capsys):
    before = tree_digest(env["tmp"])
    code = intake.main(["--sheet", str(env["sheet"]), "--inbox", str(env["inbox"]), "--datasets", str(env["datasets"]),
                        "--dry-run"])
    assert code == 0 and tree_digest(env["tmp"]) == before
    printed = capsys.readouterr().out
    assert "试运行" in printed and "CASE-009" in printed


def test_missing_and_ambiguous_files(env):
    make_pdf(env["inbox"] / "x" / "重名.pdf", body_pages(70))
    make_pdf(env["inbox"] / "y" / "重名.pdf", body_pages(71))
    make_pdf(env["inbox"] / "z" / "同内容.pdf", body_pages(72))
    make_pdf(env["inbox"] / "w" / "同内容.pdf", body_pages(72))
    rows = [row(技术标正文="不存在.pdf"), row(技术标正文="重名.pdf", 投标企业="甲二"),
            row(技术标正文="y/重名.pdf", 投标企业="甲三"), row(技术标正文="同内容.pdf", 投标企业="甲四"),
            row(招标文件="A招标.pdf;缺补遗.pdf", 技术标正文="A乙.pdf", 投标企业="甲五")]
    out = rerun(env, rows)
    assert [r.status for r in out.rows] == ["rejected", "rejected", "new", "new", "rejected"]
    cats = {(i.row, i.category) for i in out.issues}
    assert {(2, "missing_file"), (3, "ambiguous_file"), (6, "missing_file")} <= cats
    assert out.rows[2].sample_id == "CASE-005"   # 拒收行不占编号
    report = (env["datasets"] / "入库报告.md").read_text(encoding="utf-8")
    assert "### 缺文件（2，拒收）" in report and "### 文件名歧义（1，拒收）" in report
    assert "| 3 |" in report and "子目录/文件名.pdf" in report
    assert intake.main(["--sheet", str(env["sheet"]), "--inbox", str(env["inbox"]),
                        "--datasets", str(env["datasets"])]) == 1


def test_label_exclusions(env):
    make_pdf(env["inbox"] / "扫描件.pdf", ["", "", "p3"])
    make_pdf(env["inbox"] / "C1.pdf", body_pages(61))
    make_pdf(env["inbox"] / "C2.pdf", body_pages(62))
    rows = [row(技术标正文="扫描件.pdf"),
            row(技术标正文="C1.pdf", 投标企业="甲2", 技术分="61"),
            row(技术标正文="C2.pdf", 投标企业="甲3", 技术分="88.5", 成绩口径="综合总分")]
    out = rerun(env, rows)
    assert [r.status for r in out.rows] == ["new"] * 3
    s = {x["sample_id"]: x for x in pairs(env)["samples"]}
    assert s["CASE-005"]["exclusion"] == "no_text_layer" and s["CASE-005"]["body_file"]["text_layer"] is False
    assert s["CASE-006"]["exclusion"] == "score_out_of_range" and s["CASE-006"]["label"]["value"] == 61
    assert s["CASE-007"]["exclusion"] == "composite_not_technical" and s["CASE-007"]["label"] is None
    assert not any(x["eligible"] for x in s.values())
    assert {(i.row, i.category) for i in out.issues} >= {(2, "no_text_layer"), (3, "score_out_of_range"), (4, "composite")}
    m = json.loads(next(env["datasets"].glob("CASE-007_*/资料清单.json")).read_text(encoding="utf-8"))
    assert m["assessment"]["official_results"][0]["target"] == "overall_total"
    readme = next(env["datasets"].glob("CASE-005_*/README.md")).read_text(encoding="utf-8")
    assert "无文字层" in readme


def test_unpublished_score_keeps_null(env):
    out = rerun(env, [row(技术分="/"), row(技术标正文="A乙.pdf", 投标企业="乙", 技术分="")])
    assert [r.status for r in out.rows] == ["new", "new"]
    m = json.loads(next(env["datasets"].glob("CASE-005_*/资料清单.json")).read_text(encoding="utf-8"))
    assert m["assessment"]["official_results"][0]["value"] is None
    assert [s["exclusion"] for s in pairs(env)["samples"]] == ["missing_label", "missing_label"]
    assert [(i.row, i.sample) for i in out.issues if i.category == "missing_label"] == [(2, "CASE-005"), (3, "CASE-006")]


def test_duplicate_body_and_template_groups(env):
    make_pdf(env["inbox"] / "近似.pdf", body_pages(10, mutate=0.15))   # A甲 的改写
    make_pdf(env["inbox"] / "无关.pdf", body_pages(99))
    common = {"项目名称": "其他项目", "招标文件": "B招标.pdf"}
    rows = [row(),
            row(招标编号="TEST-C", 投标企业="复制企业", 技术标正文="A甲.pdf", **common),
            row(招标编号="TEST-D", 投标企业="近似企业", 技术标正文="近似.pdf", **common),
            row(招标编号="TEST-E", 投标企业="无关企业", 技术标正文="无关.pdf", **common)]
    out = rerun(env, rows)
    assert [r.status for r in out.rows] == ["new"] * 4
    s = {x["sample_id"]: x for x in pairs(env)["samples"]}
    assert s["CASE-006"]["exclusion"] == "duplicate_body" and s["CASE-006"]["duplicate_of"] == "CASE-005"
    assert s["CASE-005"]["template_group"] == s["CASE-006"]["template_group"] == s["CASE-007"]["template_group"]
    assert s["CASE-008"]["template_group"] != s["CASE-005"]["template_group"]
    assert any(i.category == "duplicate_body" and i.row == 3 for i in out.issues)
    # 再次运行：重复正文的样本按项目＋企业＋SHA 识别为已入库
    out = rerun(env, rows)
    assert [r.status for r in out.rows] == ["skipped"] * 4


def test_minhash_estimates_jaccard():
    a = " ".join(body_pages(5, pages=4))
    near = " ".join(body_pages(5, pages=4, mutate=0.1))
    far = " ".join(body_pages(6, pages=4))
    sa, sn, sf = intake.minhash(a), intake.minhash(near), intake.minhash(far)
    assert intake.jaccard(sa, intake.minhash(a)) == 1.0
    assert intake.jaccard(sa, sn) >= 0.5 > intake.jaccard(sa, sf)
    assert intake.minhash("abc") is None


def test_explicit_ids_and_conflicts(env):
    rows = [row(样本编号="CASE-020"), row(样本编号="CASE-020", 技术标正文="A乙.pdf", 投标企业="乙"),
            row(技术标正文="B甲.pdf", 投标企业="丙"), row(样本编号="CASE-001", 技术标正文="B丁.pdf", 投标企业="丁"),
            row(技术标正文="A甲.pdf")]
    out = rerun(env, rows)
    assert [r.status for r in out.rows] == ["new", "rejected", "new", "rejected", "rejected"]
    assert out.rows[2].sample_id == "CASE-021"
    assert {i.category for i in out.issues if i.row in (3, 5, 6)} == {"id_conflict"}
    assert len(list(env["datasets"].glob("CASE-001_*"))) == 1


def test_validation_errors(env):
    rows = [row(工程类型="道路"), row(本公司="Y"), row(提交日期="20260420"), row(技术满分="0"),
            row(技术分="五十"), row(项目名称=""), row(成绩来源页码="第3页"), row(技术标正文="A甲.pdf;A乙.pdf")]
    out = rerun(env, rows)
    assert all(r.status == "rejected" for r in out.rows)
    assert all(any(i.category == "field_invalid" for i in r.issues) for r in out.rows)
    assert pairs(env)["samples"] == []


def test_field_scores_conflict_and_own_score(env):
    rows = [row(同场技术分="52.3;48", 入围分数线="50"),
            row(技术标正文="A乙.pdf", 投标企业="乙", 技术分="48", 同场技术分="48;52.3"),
            row(技术标正文="A丙.pdf", 投标企业="丙", 技术分="45"),
            row(招标编号="TEST-B", 招标文件="B招标.pdf", 技术标正文="B甲.pdf", 同场技术分="40;41"),
            row(招标编号="TEST-B", 招标文件="B招标.pdf", 技术标正文="B丁.pdf", 投标企业="丁", 同场技术分="40;42")]
    out = rerun(env, rows)
    assert [r.status for r in out.rows] == ["new", "new", "new", "rejected", "rejected"]
    p = pairs(env)
    assert p["projects"]["TEST-A|"]["field_scores"] == [52.3, 48.0]
    own = [i for i in out.issues if i.category == "own_score_not_in_field"]
    assert [i.sample for i in own] == ["CASE-007"]
    # 库中已有同场技术分时，新行的不同值拒收
    make_pdf(env["inbox"] / "A戊.pdf", body_pages(77))
    out = rerun(env, rows[:3] + [row(技术标正文="A戊.pdf", 投标企业="戊", 同场技术分="1;2")])
    assert out.rows[3].status == "rejected"
    assert out.rows[3].issues[0].category == "field_conflict"


def test_project_inconsistencies_reported(env):
    rows = [row(), row(技术标正文="A乙.pdf", 投标企业="乙", 技术满分="70", 评分办法="其他评分办法")]
    out = rerun(env, rows)
    cats = {i.category for i in out.issues}
    assert {"max_inconsistent", "project_inconsistent"} <= cats
    assert all(r.status == "new" for r in out.rows)


def test_item_scores(env):
    rows = [row(技术分="52"), row(技术标正文="A乙.pdf", 投标企业="乙", 技术分="40")]
    write_sheet(env["sheet"], rows)
    items = env["tmp"] / "分项得分.csv"
    with items.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["样本编号或登记表行号", "评分项（原文或简称）", "得分", "满分", "评委"])
        w.writerows([["2", "施工方案", "30", "35", ""], ["2", "质量措施", "22", "25", ""],
                     ["2", "施工方案", "29", "35", "评委1"], ["3", "施工方案", "20", "35", ""],
                     ["3", "进度", "30", "25", ""], ["9", "施工方案", "1", "35", ""]])
    out = run(env["sheet"], env["inbox"], env["datasets"])   # 同目录的分项得分.csv 自动读取
    s = {x["sample_id"]: x for x in pairs(env)["samples"]}
    assert s["CASE-005"]["label"]["items"] == [
        {"item": "施工方案", "value": 30.0, "maximum": 35.0, "evaluator": None},
        {"item": "质量措施", "value": 22.0, "maximum": 25.0, "evaluator": None},
        {"item": "施工方案", "value": 29.0, "maximum": 35.0, "evaluator": "评委1"}]
    cats = [(i.category, i.row) for i in out.issues]
    assert ("items_invalid", 3) in cats and out.rows[1].issues[0].sample == "CASE-006" and ("items_invalid", None) in cats
    assert ("items_sum", 3) in cats and ("items_sum", 2) not in cats
    assert all(r.status == "new" for r in out.rows)


def test_sheet_header_and_encoding(env):
    bad = env["tmp"] / "bad.csv"
    bad.write_text("样本编号,项目名称\n", encoding="utf-8-sig")
    with pytest.raises(InputError):
        run(bad, env["inbox"], env["datasets"])
    write_sheet(env["sheet"], [row()], tail=False)          # 省略末尾两列的登记表
    gbk = env["tmp"] / "gbk.csv"
    gbk.write_bytes(env["sheet"].read_bytes().decode("utf-8-sig").encode("gb18030"))
    out = run(gbk, env["inbox"], env["datasets"], dry_run=True)
    assert out.rows[0].status == "new"


def test_naming_helpers():
    assert intake.region_short("安徽省广德市") == "广德"
    assert intake.region_short("安徽省合肥市") == "合肥"
    assert intake.bidder_short("安徽秉一市政工程有限公司") == "安徽秉一市政"
    name = "广德未来科创城污水管网改造及新建工程项目-科技路、昌盛路及周边污水管网改造及新建工程"
    assert intake.project_short(name, "安徽省广德市") == "未来科创城污水管网改造及新建"


def test_template_matches_columns():
    template = intake.DATASETS / "入库模板" / "样本登记表.csv"
    if not template.is_file():
        pytest.skip("数据集中没有入库模板")
    header, rows, encoding = intake.read_csv(template)
    assert encoding == "utf-8-sig" and header == list(COLUMNS + TAIL_COLUMNS) and rows


def test_duplicate_of_legacy_case(env):
    legacy = env["datasets"] / "CASE-001_广德_污水管网改造及新建" / "资料清单.json"
    sha = hashlib.sha256((env["inbox"] / "A甲.pdf").read_bytes()).hexdigest()
    legacy.write_text(json.dumps({"schema_version": "1.2", "case_id": "CASE-001",
                                  "file_records": [{"file_id": "FILE-002", "sha256": sha}]}), encoding="utf-8")
    rerun(env, [row()])
    sample = pairs(env)["samples"][0]
    assert sample["exclusion"] == "duplicate_body" and sample["duplicate_of"] == "CASE-001"
