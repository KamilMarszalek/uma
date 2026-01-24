def precision(tp: int, fp: int) -> float:
    return _safe_divide(tp, tp + fp)


def recall(tp: int, fn: int) -> float:
    return _safe_divide(tp, tp + fn)


def f1_score(tp: int, fp: int, fn: int) -> float:
    prec = precision(tp, fp)
    rec = recall(tp, fn)
    return _safe_divide(2.0 * prec * rec, prec + rec)


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)
