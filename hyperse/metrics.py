import numpy as np


def binary_metrics(y, score):
    """Global ROC-AUC and threshold-grouped AP, including score ties."""
    y,score = np.asarray(y,dtype=int),np.asarray(score,dtype=float)
    if not np.isfinite(score).all() or len(y) != len(score):
        raise ValueError('Invalid predictions')
    p,n = int(y.sum()),int((1-y).sum())
    if p == 0 or n == 0:
        raise ValueError('Both classes are required')
    order = np.argsort(-score,kind='stable')
    ys,ss = y[order],score[order]
    ends = np.r_[np.flatnonzero(np.diff(ss)!=0),len(y)-1]
    tp = np.r_[0,np.cumsum(ys)[ends]]
    fp = np.r_[0,np.cumsum(1-ys)[ends]]
    auc = np.trapz(tp/p,fp/n)
    ap = np.sum(np.diff(tp)/p * tp[1:]/(tp[1:]+fp[1:]))
    return dict(auc=float(auc),ap=float(ap),count=len(y),positives=p,negatives=n)
