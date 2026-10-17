"""模块边界契约；官方标签、诊断与预测互不替代。"""
from __future__ import annotations

import math
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class Evidence(Contract):
    file_id: str
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_path: str
    source_ids: list[str]
    pdf_page: int = Field(gt=0)
    start_char: int = Field(ge=0)
    end_char: int = Field(gt=0)
    quote: str = Field(min_length=1)
    kind: Literal["source_excerpt"]
    retrieval_score: float

    @model_validator(mode="after")
    def span(self):
        if self.end_char <= self.start_char:
            raise ValueError("引用范围必须非空")
        return self


class OfficialLabel(Contract):
    target: Literal["technical_total", "composite_total"]
    value: float = Field(ge=0)
    maximum: float = Field(gt=0)
    source_file_id: str
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_pdf_page: int = Field(gt=0)
    submission_id: str
    aggregation: str = Field(min_length=1)
    confirmed: bool
    known_at: datetime
    revision: str

    @model_validator(mode="after")
    def bounds(self):
        if self.value > self.maximum:
            raise ValueError("官方成绩超过满分")
        if self.known_at.tzinfo is None:
            raise ValueError("成绩时间须含时区")
        return self


class ReportedResult(Contract):
    """用户报送结果单独保存；不能自动升级为官方监督标签。"""
    result_id: str
    source_kind: Literal["user_statement"]
    received_date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    source_description: str = Field(min_length=1)
    verification_status: Literal["official_source_pending"]
    target: Literal["technical_total", "composite_total"]
    target_interpretation: str = Field(min_length=1)
    value: float = Field(ge=0)
    maximum: float = Field(gt=0)
    reported_shortlist_cutoff: float | None = Field(default=None, ge=0)
    cutoff_scope_status: str
    submission_file_id: str
    submission_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    official_training_eligible: Literal[False]

    @model_validator(mode="after")
    def bounds(self):
        datetime.strptime(self.received_date, "%Y-%m-%d")
        if self.value > self.maximum or (self.reported_shortlist_cutoff is not None and self.reported_shortlist_cutoff > self.maximum):
            raise ValueError("报送成绩或门槛超过满分")
        return self


class Handoff(BaseModel):
    """流水线交接记录：已知字段严格校验，上游追加的字段保留但不参与准入。"""
    model_config = ConfigDict(extra="allow", allow_inf_nan=False)


def decimal_string(value):
    """格式 B 数值一律十进制字符串；缺失为 null。"""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("数值须为十进制字符串")  # noqa: TRY004  pydantic 只把 ValueError 转为校验错误
    try:
        number = Decimal(value)
    except InvalidOperation as error:
        raise ValueError("不是十进制字符串") from error
    if not number.is_finite():
        raise ValueError("数值须有限")
    return float(number)


DecimalText = Annotated[float | None, BeforeValidator(decimal_string)]
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class PairFile(Handoff):
    file_id: str
    path: str
    sha256: Sha256
    pages: int | None = Field(default=None, ge=0)
    role: Literal["tender", "addendum"]


class PairBody(Handoff):
    file_id: str
    path: str
    sha256: Sha256
    pages: int | None = Field(default=None, ge=0)
    text_layer: bool


class LabelItem(Handoff):
    """官方分项得分；同一分项可有多位评委各一行。"""
    item: str = Field(min_length=1)
    value: float | None = Field(default=None, ge=0)
    maximum: float = Field(gt=0)
    evaluator: str | None = None


class PairLabel(Handoff):
    target: Literal["technical_total", "composite_total"]
    value: float | None = Field(ge=0)
    maximum: float = Field(gt=0)
    verification_status: Literal["verified", "pending"]
    source_kind: Literal["official_document", "company_ledger", "user_statement"]
    known_at: date | None = None
    stage_result: Literal["shortlisted", "not_shortlisted", "rejected", "awarded", "not_awarded", "unknown"] | None = None
    items: list[LabelItem] = Field(default_factory=list)


class PairSample(Handoff):
    """交接格式 A 的一份样本：某项目／标段下某企业的一次实际提交。"""
    sample_id: str = Field(min_length=1)
    project_key: str = Field(min_length=1)
    template_group: str = Field(min_length=1)
    region: str
    engineering_type: str
    scoring_scheme: str
    bidder: str
    own_company: bool
    submitted_at: date
    tender_files: list[PairFile]
    body_file: PairBody | None
    label: PairLabel | None
    eligible: bool
    exclusion: Literal["missing_label", "composite_not_technical", "no_text_layer", "duplicate_body",
                       "missing_files", "score_out_of_range"] | None

    @model_validator(mode="after")
    def exclusion_consistent(self):
        if self.eligible == (self.exclusion is not None):
            raise ValueError("eligible 与 exclusion 不一致")
        return self


class ProjectField(Handoff):
    """同场全部投标人的官方技术分（原量尺）与入围线；只用于评测，不作特征。

    没有同场成绩时 ``field_scores`` 为 null；公示里未公布的单个分数（写“/”）为 null 元素，计算时略去。
    """
    field_scores: list[float | None] | None = None
    shortlist_cutoff: float | None = Field(default=None, ge=0)


class PairSet(Handoff):
    """交接格式 A：评分配对集。样本逐条校验，单条不合格不使整份失效。"""
    schema_: Literal["score-pairs/1"] = Field(alias="schema")
    generated_at: str
    datasets_root: str
    code_sha256: str | dict
    samples: list[dict]
    projects: dict[str, ProjectField] = Field(default_factory=dict)


class PredictedTotal(Handoff):
    status: Literal["candidate", "abstained"]
    estimate: DecimalText = None
    low: DecimalText = None
    high: DecimalText = None
    maximum: DecimalText = None


class PredictedItem(Handoff):
    """分项预测；上游追加的数值键（如 grade_index、coverage）按分值加权聚合为特征。"""
    statement: str = ""
    max_points: DecimalText = None
    estimate: DecimalText = None
    low: DecimalText = None
    high: DecimalText = None
    failed: bool = False


class PredictionInputs(Handoff):
    tender_sha256: list[Sha256]
    body_sha256: Sha256


class PredictionExtra(Handoff):
    """产品后端计算并随预测保存的评分特征（feature_spec 如 pf-1）；值为数值或 null。"""
    input_method: str = "legacy-truncated"
    feature_spec: str | None = None
    features: dict[str, float | None] = Field(default_factory=dict)


class SystemPrediction(Handoff):
    """交接格式 B：系统对单份样本的批量预测；只由招标文件与正文产生。"""
    schema_: Literal["system-prediction/1"] = Field(alias="schema")
    sample_id: str
    status: Literal["succeeded", "failed"]
    error: str | None = None
    fingerprint: str | None = None
    model: str | None = None
    prompt_version: str | None = None
    code_sha256: str | None = None
    inputs: PredictionInputs
    technical_full_score: DecimalText = None
    prediction: PredictedTotal | None = None
    items: list[PredictedItem] = Field(default_factory=list)
    compliance: dict | None = None
    findings: dict | None = None
    body: dict | None = None
    artifact_extra: PredictionExtra | None = None

    @model_validator(mode="after")
    def candidate_values(self):
        p = self.prediction
        if self.status == "succeeded" and p and p.status == "candidate" and (
                p.estimate is None or not p.maximum or not 0 <= p.estimate <= p.maximum):
            raise ValueError("候选估计须处于预测满分量尺内")
        return self


METHODS = ("mean", "median", "scheme_mean", "raw", "linear", "isotonic", "anchor", "ridge", "ridge_anchor",
           "ridge_isotonic", "knn", "ridge_knn")
BASELINES = ("mean", "median", "scheme_mean")
# 以 pf-1 特征为唯一输入、可导出为 scoring-model/1 的方法；正文近邻类只评测不导出。
EXPORTABLE = ("raw", "linear", "isotonic", "anchor", "ridge", "ridge_anchor", "ridge_isotonic")


class EvalCV(Contract):
    folds: int = Field(default=5, ge=2)
    repeats: int = Field(default=3, ge=1)


class EvalHoldout(Contract):
    test_fraction: float = Field(default=0.2, gt=0, le=0.5)
    min_test_samples: int = Field(default=10, ge=1)


class EvalFeatures(Contract):
    include: list[str] = Field(min_length=1)
    exclude: list[str] = Field(default_factory=list)


class EvalKnn(Contract):
    k: int = Field(default=10, ge=1)
    ngrams: list[int] = Field(default=[2, 3], min_length=1)
    hash_bits: int = Field(default=18, ge=10, le=22)


class EvalMinimums(Contract):
    labeled_samples: int = Field(default=30, ge=1)
    independent_projects: int = Field(default=10, ge=1)
    slice_samples: int = Field(default=10, ge=1)
    ranking_projects: int = Field(default=5, ge=1)
    shortlist_samples: int = Field(default=20, ge=1)
    item_matches: int = Field(default=10, ge=1)


class EvalAcceptance(Contract):
    max_mae: float = Field(default=3.0, gt=0, le=100)
    max_abs_p90: float = Field(default=5.0, gt=0, le=100)
    within_points: float = Field(default=3.0, gt=0, le=100)
    min_within_ratio: float = Field(default=0.8, gt=0, le=1)
    max_abs_bias: float = Field(default=1.0, ge=0, le=100)
    max_repeat_range_p90: float = Field(default=2.0, ge=0, le=100)
    minimum_repeat_runs: int = Field(default=3, ge=2)


class EvalProtocol(Contract):
    """评测协议：在看测试标签前冻结；文件 SHA 写入报告。"""
    model_config = ConfigDict(extra="forbid", strict=False, allow_inf_nan=False, populate_by_name=True)
    schema_: Literal["score-eval-protocol/1"] = Field(alias="schema")
    frozen_at: date
    description: str = Field(min_length=1)
    target_mae: float = Field(gt=0, le=100)
    acceptance: EvalAcceptance | None = None
    selection: str = "cv_mae"
    methods: list[Literal[METHODS]] = Field(min_length=2)
    ridge_alphas: list[float] = Field(default=[0.1, 1.0, 10.0, 100.0], min_length=1)
    seed: int
    cv: EvalCV = EvalCV()
    holdout: EvalHoldout = EvalHoldout()
    bootstrap_resamples: int = Field(default=1000, ge=100)
    confidence: float = Field(default=0.95, gt=0.5, lt=1)
    interval_levels: list[float] = Field(default=[0.8, 0.9], min_length=1)
    features: EvalFeatures
    knn: EvalKnn = EvalKnn()
    slices: list[Literal["scoring_scheme", "engineering_type", "own_company", "region", "system_status"]] = Field(
        default=["scoring_scheme", "engineering_type", "own_company"])
    item_match_threshold: float = Field(default=0.5, gt=0, le=1)
    minimums: EvalMinimums = EvalMinimums()
    allow_repeat_runs: bool = False
    learning_curve_fractions: list[float] = Field(default=[0.25, 0.5, 0.75, 1.0])
    label_sources: list[Literal["official_document", "company_ledger", "user_statement"]] = Field(
        default=["official_document", "company_ledger"], min_length=1)

    @model_validator(mode="after")
    def coherent(self):
        if self.acceptance is not None and (self.target_mae > self.acceptance.max_mae or self.confidence != 0.95):
            raise ValueError("融合验收协议须采用不宽于 max_mae 的目标和 95% 组级区间")
        if not any(m in BASELINES for m in self.methods) or not any(m in EXPORTABLE for m in self.methods):
            raise ValueError("方法清单须同时含基线与可导出的系统方法")
        if len(set(self.methods)) != len(self.methods):
            raise ValueError("方法重复")
        if self.selection != "cv_mae" and (self.selection not in self.methods or self.selection not in EXPORTABLE):
            raise ValueError("selection 须为 cv_mae 或清单内的可导出方法")
        if any(not 0 < f <= 1 for f in self.learning_curve_fractions):
            raise ValueError("学习曲线比例须在 (0,1]")
        if any(a <= 0 for a in self.ridge_alphas) or any(not 0 < q < 1 for q in self.interval_levels):
            raise ValueError("alpha 须为正，区间水平须在 (0,1)")
        if any(n < 1 for n in self.knn.ngrams) or not all(math.isfinite(a) for a in self.ridge_alphas):
            raise ValueError("n-gram 长度须为正")
        return self

class Finding(Contract):
    finding_id: str
    check_id: str
    severity: Literal["review", "blocking"]
    status: Literal["needs_engineer_confirmation", "confirmed", "rejected"]
    statement: str
    evidence: list[Evidence] = Field(min_length=1)
    official_score_effect: None = None


class Prediction(Contract):
    target: Literal["technical_total"]
    status: Literal["candidate", "unavailable"]
    bundle_id: str
    input_snapshot: str
    estimated_value: float | None
    maximum: float
    validation_status: Literal["not_validated", "validated_in_scope"]
    applicability: str
    official_subscores: None = None
    win_probability: None = None

    @model_validator(mode="after")
    def candidate(self):
        if self.maximum <= 0:
            raise ValueError("满分必须为正")
        if self.status == "unavailable" and self.estimated_value is not None:
            raise ValueError("不可预测不能输出数值")
        if self.status == "candidate" and (self.estimated_value is None or not 0 <= self.estimated_value <= self.maximum):
            raise ValueError("候选值必须处于成绩量尺内")
        return self
