import math
from types import SimpleNamespace
from fait.core.utils import fuse_scores

def C(**kw):
    # tiny config shim (attrs, not dict) so _fcfg_get works either way
    return SimpleNamespace(**kw)

def test_and_rule_pass_and_fail():
    cfg = C(method="and", class_thresholds={"knife": 0.60}, gdino_only_default_tau=0.50)
    # both above class tau -> accept, fused=min(...)
    fused, tau, ok = fuse_scores(0.70, 0.80, "knife", cfg)
    assert ok and tau == 0.60 and abs(fused - 0.70) < 1e-6
    # one below -> reject
    fused, tau, ok = fuse_scores(0.59, 0.95, "knife", cfg)
    assert not ok and tau == 0.60

def test_weighted_linear_blend():
    cfg = C(method="weighted", alpha=0.6, tau_star=0.65)
    fused, tau, ok = fuse_scores(0.70, 0.50, "anything", cfg)  # 0.6*0.70 + 0.4*0.50 = 0.62
    assert math.isclose(fused, 0.62, rel_tol=1e-6) and tau == 0.65 and not ok
    fused, tau, ok = fuse_scores(0.85, 0.60, "anything", cfg)  # 0.6*0.85 + 0.4*0.60 = 0.75
    assert math.isclose(fused, 0.75, rel_tol=1e-6) and ok

def test_product_mode():
    cfg = C(method="product", tau_star=0.60)
    fused, tau, ok = fuse_scores(0.70, 0.90, "x", cfg)  # 0.63
    assert math.isclose(fused, 0.63, rel_tol=1e-6) and tau == 0.60 and ok
    fused, tau, ok = fuse_scores(0.50, 0.90, "x", cfg)  # 0.45
    assert not ok

def test_max_mode():
    cfg = C(method="max", tau_star=0.75)
    fused, tau, ok = fuse_scores(0.72, 0.80, "x", cfg)
    assert math.isclose(fused, 0.80, rel_tol=1e-6) and tau == 0.75 and ok

def test_sum_znorm_mode():
    # z = (x - mean) / std ; sum = 1 + (-1) = 0
    cfg = C(method="sum", tau_star=0.0,
            gdino_mean=0.50, gdino_std=0.10,
            det_mean=0.50,  det_std=0.10)
    fused, tau, ok = fuse_scores(0.60, 0.40, "x", cfg)
    assert math.isclose(fused, 0.0, rel_tol=1e-6) and ok
    # require positive sum
    cfg.tau_star = 0.1
    fused, tau, ok = fuse_scores(0.60, 0.40, "x", cfg)
    assert not ok

def test_logistic_mode_with_znorm():
    # sigmoid(w_g*z_g + w_c*z_c + b)
    cfg = C(method="logistic", tau_star=0.5,
            gdino_mean=0.5, gdino_std=0.1,
            det_mean=0.5,  det_std=0.1,
            w_g=1.0, w_c=1.0, b=0.0)
    # z_g=0, z_c=0 => sigmoid(0)=0.5 -> accept
    fused, tau, ok = fuse_scores(0.5, 0.5, "x", cfg)
    assert math.isclose(fused, 0.5, rel_tol=1e-6) and ok
    # negative sum => < 0.5 -> reject
    fused, tau, ok = fuse_scores(0.4, 0.4, "x", cfg)  # z=-1,-1 -> sigmoid(-2) ~ 0.119
    assert fused < 0.5 and not ok

def test_class_threshold_fallback():
    # AND mode: per-class threshold takes precedence; default used otherwise
    cfg = C(method="and",
            class_thresholds={"knife": 0.55},
            gdino_only_default_tau=0.50)
    # known class -> tau=0.55
    fused, tau, ok = fuse_scores(0.56, 0.56, "knife", cfg)
    assert ok and tau == 0.55
    # unknown class -> tau falls back to default
    fused, tau, ok = fuse_scores(0.51, 0.51, "spoon", cfg)
    assert ok and tau == 0.50
