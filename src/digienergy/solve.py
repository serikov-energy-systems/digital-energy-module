"""Solve the PyPSA model with Clarabel, a direct interior point method.

HiGHS needs minutes even for a daily time step on this model, in which a few capacity variables enter every time step.
The model is built by PyPSA and linopy as usual; its matrices are handed to Clarabel in GW, GWh and million EUR, and
the solution and the prices are returned in the original units.
"""

from __future__ import annotations

import os
import time

import clarabel
import numpy as np
from scipy import sparse

POWER = 1e-3          # internal power and energy unit per MW or MWh
MONEY = 1e-6          # internal money unit per EUR


def split_dense_columns(a: sparse.csr_matrix, cost, lower, upper, limit: int = 40):
    """Replace each column with many entries by a chain of equal copies, each used in at most `limit` rows.

    Capacity variables appear in every time step; the copies y_k = y_(k-1) keep the factorisation sparse."""
    a = a.tocsc()
    n_rows, n_cols = a.shape
    dense = np.flatnonzero(np.diff(a.indptr) > limit)
    coo = a.tocoo()
    col = coo.col.copy()
    next_column = n_cols
    chain_rows, chain_cols, chain_vals = [], [], []
    n_chain = 0
    for j in dense:
        entries = np.arange(a.indptr[j], a.indptr[j + 1])
        entries = entries[np.argsort(a.indices[entries], kind="stable")]
        previous = j
        for start in range(limit, len(entries), limit):
            col[entries[start:start + limit]] = next_column
            chain_rows += [n_chain, n_chain]
            chain_cols += [next_column, previous]
            chain_vals += [1.0, -1.0]
            previous = next_column
            next_column += 1
            n_chain += 1
    extra = next_column - n_cols
    top = sparse.csr_matrix((coo.data, (coo.row, col)), shape=(n_rows, next_column))
    chain = sparse.csr_matrix((chain_vals, (chain_rows, chain_cols)), shape=(n_chain, next_column))
    return (top, chain, np.r_[cost, np.zeros(extra)], np.r_[lower, np.full(extra, -np.inf)],
            np.r_[upper, np.full(extra, np.inf)])


def solve(n, verbose: bool = False, tolerance: float = 1e-7, split: int = 40) -> dict:
    """n.model must exist. Returns the objective (EUR), primal values by variable name and duals by constraint
    name, each as xarray objects shaped like the model's."""
    model = n.model
    m = model.matrices
    a = m.A.tocsr()
    b, sense, cost, lower, upper = np.asarray(m.b, float), np.asarray(m.sense), np.asarray(m.c, float), m.lb, m.ub
    # variables x = y / POWER (all are powers or energies); rows with a power on the right-hand side scale likewise
    b = b * POWER
    cost = cost / POWER * MONEY
    lower, upper = lower * POWER, upper * POWER
    equal, below, above = sense == "=", sense == "<", sense == ">"
    n_rows, n_columns = a.shape
    chain = sparse.csr_matrix((0, n_columns))
    if split:
        a, chain, cost, lower, upper = split_dense_columns(a, cost, lower, upper, split)
    identity = sparse.identity(a.shape[1], format="csr")
    has_low, has_up = np.isfinite(lower), np.isfinite(upper)
    g = sparse.vstack([a[equal], chain, a[below], -a[above], identity[has_up], -identity[has_low]], format="csc")
    h = np.r_[b[equal], np.zeros(chain.shape[0]), b[below], -b[above], upper[has_up], -lower[has_low]]
    zero = int(equal.sum()) + chain.shape[0]
    cones = [clarabel.ZeroConeT(zero), clarabel.NonnegativeConeT(len(h) - zero)]
    settings = clarabel.DefaultSettings()
    settings.verbose = verbose
    settings.tol_gap_rel = tolerance
    settings.tol_gap_abs = tolerance
    settings.tol_feas = tolerance
    settings.max_iter = 300
    settings.direct_solve_method = os.environ.get("CLARABEL_LINEAR", "qdldl")
    p = sparse.csc_matrix((a.shape[1], a.shape[1]))
    started = time.time()
    result = clarabel.DefaultSolver(p, cost, g, h, cones, settings).solve()
    interior = time.time() - started
    status = str(result.status)
    if status not in ("Solved", "AlmostSolved"):
        raise RuntimeError(f"Clarabel: {status}")
    x = np.asarray(result.x)[:n_columns] / POWER
    z = np.asarray(result.z)
    # shadow price of each original row, with the sign convention of linopy (d objective / d right-hand side)
    dual = np.zeros(n_rows)
    n_eq, n_be, n_ab = int(equal.sum()), int(below.sum()), int(above.sum())
    dual[equal] = -z[:n_eq]
    dual[below] = -z[zero:zero + n_be]
    dual[above] = z[zero + n_be:zero + n_be + n_ab]
    dual *= POWER / MONEY
    return {"status": status, "objective": float(result.obj_val / MONEY), "iterations": result.iterations,
            "solver_seconds": interior,
            "x": x, "dual": dual, "vlabels": np.asarray(m.vlabels), "clabels": np.asarray(m.clabels)}


def lookup(labels, values: np.ndarray, index_labels: np.ndarray):
    """Values for an xarray of labels (-1 where the model has no entry)."""
    table = np.full(int(index_labels.max()) + 1, np.nan)
    table[index_labels] = values
    flat = labels.values
    out = np.where(flat >= 0, table[np.clip(flat, 0, None)], np.nan)
    return labels.copy(data=out)


def variable(n, solution: dict, name: str):
    """Primal values of one variable as a pandas object, e.g. "Link-p" -> time x name."""
    return lookup(n.model.variables[name].labels, solution["x"], solution["vlabels"]).to_pandas()


def dual_of(n, solution: dict, name: str):
    """Duals of one constraint block, e.g. "Bus-nodal_balance" -> name x time."""
    return lookup(n.model.constraints[name].labels, solution["dual"], solution["clabels"]).to_pandas()
