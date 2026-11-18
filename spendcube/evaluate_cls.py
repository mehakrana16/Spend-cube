import numpy as np, pandas as pd
def accuracy(pred, truth, depth, weights=None):
    out = {}
    for d in range(1, depth + 1):
        ok = (pred[f"l{d}"].values == truth[f"l{d}"].values)
        out[f"L{d}"] = round(float(np.average(ok, weights=weights)), 4)
    return out
