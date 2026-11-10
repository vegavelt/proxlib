import matplotlib.pyplot as plt
import numpy as np
from .algorithms import metrics

def plot_convergence(elems, F_star = None, rates = None, log_x = True, log_y = True, ax = None, title = None, labels = None):
    if ax is None:
        _, ax = plt.subplots()
    axis_labels = {
        "x": "k",
        "y": r"$F(x_k) - F^\star$" if F_star is not None else "error",
    }
    if labels is not None:
        axis_labels.update(labels)
 
    anchor = None 
 
    for name, values in elems.items():
        y = np.asarray(values, dtype=float)
        if F_star is not None:
            y = y - F_star
        if log_y:
            y = np.where(y > 0, y, np.nan)
        k = np.arange(len(y))
        if log_x:
            k, y = k[1:], y[1:]
        ax.plot(k, y, label=name)
        if anchor is None:
            anchor = (k, y)
 
    if rates and anchor is not None:
        k, y = anchor
        mask = k >= 1                      
        k, y = k[mask], y[mask]
        valid = np.flatnonzero(np.isfinite(y))
        if valid.size > 0:
            k0, y0 = k[valid[0]], y[valid[0]]
            for p in rates:
                ref = y0 * (k0 / k) ** p   
                name = r"$O(1/k)$" if p == 1 else rf"$O(1/k^{{{p}}})$"
                ax.plot(k, ref, linestyle="--", linewidth=1, label=name)
 
    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
 
    ax.set_xlabel(axis_labels["x"])
    ax.set_ylabel(axis_labels["y"])
    if title is not None:
        ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
 
    return ax


