"""融合评分门槛：分开检查长尾、偏差、接近率、基线增益与完整复跑。"""
from types import SimpleNamespace

import numpy as np

from tender_research.contracts import EvalAcceptance
from tender_research.score_eval import acceptance_summary


def evaluate(errors, **changes):
    args = {"protocol": SimpleNamespace(acceptance=EvalAcceptance()), "errors": np.array(errors, dtype=float),
            "gain": {"low": 0.01}, "robustness": {"runs": 3, "complete_n": len(errors), "complete_range_p90": 2.0},
            "sample_count": len(errors)}
    args.update(changes)
    return acceptance_summary(**args)


def test_all_thresholds_and_strict_baseline_boundary():
    assert evaluate([0.0] * 10)["status"] == "passed"
    result = evaluate([0.0] * 10, gain={"low": 0.0})
    assert result["status"] == "not_passed"
    assert result["checks"]["gain_ci95_low"]["status"] == "not_passed"


def test_tail_error_and_bias_do_not_hide_behind_low_mae():
    tail = evaluate([6, -6] + [0] * 8)
    assert tail["checks"]["mae"]["status"] == "passed"
    assert tail["checks"]["abs_p90"]["status"] == "not_passed"
    assert evaluate([2.0] * 10)["checks"]["abs_bias"]["status"] == "not_passed"
    assert evaluate([3.1, -3.1] * 5)["checks"]["within_ratio"]["status"] == "not_passed"


def test_missing_partial_or_unstable_repeats_cannot_pass():
    assert evaluate([0] * 10, robustness={"runs": 1})["status"] == "not_evaluable"
    assert evaluate([0] * 10, robustness={"runs": 3, "complete_n": 9, "complete_range_p90": 1})["status"] == "not_evaluable"
    assert evaluate([0] * 10, robustness={"runs": 3, "complete_n": 10, "complete_range_p90": 2.01})["status"] == "not_passed"
