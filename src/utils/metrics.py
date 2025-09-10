import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    log_loss,
    brier_score_loss,
)


def ece_binary(p, y, n_bins=15):
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=int)
    yhat = (p >= 0.5).astype(int)
    conf_per_sample = np.where(
        yhat == 1, p, 1 - p
    )  # confidence = prob of predicted class

    bins = np.linspace(0, 1, n_bins + 1)
    inds = np.digitize(conf_per_sample, bins) - 1

    correct = (yhat == y).astype(int)
    N = len(p)
    ece = 0.0
    for b in range(n_bins):
        m = inds == b
        if not m.any():
            continue
        acc_bin = correct[m].mean()
        conf_bin = conf_per_sample[m].mean()
        ece += (m.sum() / N) * abs(acc_bin - conf_bin)
    return float(ece)


def sens_at_spec(p, y, target_spec=0.95):
    thr = np.unique(np.sort(p))
    best = 0.0
    for t in thr:
        yhat = (p >= t).astype(int)
        tn = ((yhat == 0) & (y == 0)).sum()
        fp = ((yhat == 1) & (y == 0)).sum()
        tp = ((yhat == 1) & (y == 1)).sum()
        fn = ((yhat == 0) & (y == 1)).sum()
        spec = tn / (tn + fp + 1e-9)
        if spec >= target_spec:
            sens = tp / (tp + fn + 1e-9)
            best = max(best, sens)
    return float(best)


def eval_binary_scores(p, y):
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    out = {}
    out["AUROC"] = float(roc_auc_score(y, p))
    out["AUPRC"] = float(average_precision_score(y, p))
    out["NLL"] = float(log_loss(y, np.c_[1 - p, p], labels=[0, 1]))
    out["Brier"] = float(brier_score_loss(y, p))
    out["ECE"] = ece_binary(p, y)
    out["Acc@0.5"] = float(((p >= 0.5).astype(int) == y).mean())
    out["Sens@95%Spec"] = sens_at_spec(p, y, 0.95)
    return out
