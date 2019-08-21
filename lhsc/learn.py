# noinspection PyUnresolvedReferences
from gurobipy import Model, GRB, quicksum

import numpy as np
from typing import List, Set, Tuple, Optional

from contextlib import contextmanager
import sys, os


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def learn_weighted_max_sat(
    m: int, data: np.ndarray, labels: np.ndarray, contexts: List[Set[int]]
) -> List[Tuple[Optional[float], Set[int]]]:
    """
    Learn a weighted MaxSAT model from examples. Contexts and clauses are set-encoded, i.e., they are represented by
    sets containing positive or negative integers in the range -n to n. If a set contains an positive integer i, the ith
     Boolean feature is set to True, if it contains a negative integer -i, the ith Boolean feature is set to False.
    :param m:
        The number of clauses in the MaxSAT model to learn
    :param data:
        A Boolean s x n (number examples x number Boolean variables) numpy array in which every row is an example and
        every column a Boolean feature (True = 1 or False = 0)
    :param labels:
        A Boolean numpy array (length s) where the kth entry contains the label (1 or 0) of the kth example
    :param contexts:
        A list of s set-encoded contexts.
    :return:
        A list of weights and clauses. Every entry of the list is a tuple containing as first element None for hard
        constraints (clauses) or a floating point number for soft constraints, and as second element a set-encoded clause.
    """

    w_max_value = 1
    s = data.shape[0]
    n = data.shape[1]
    big_m = 10000
    epsilon = 10 ** (-15)

    context_pool = dict()
    context_indices = []
    for context in contexts:
        key = frozenset(context)
        if key not in context_pool:
            context_pool[key] = len(context_pool)
        context_indices.append(context_pool[key])
    context_counts = len(context_pool)

    with suppress_stdout():
        m = Model("milp")

    m.setParam("OutputFlag", False)

    # Constraint decision variables
    c_j = [m.addVar(vtype=GRB.BINARY, name=f"c_{j})") for j in range(m)]
    a_jl = [
        [m.addConstr(vtype=GRB.BINARY, name="a_[{j}, {l}]") for l in range(2 * n)]
        for j in range(m)
    ]

    # Weights decision variables
    w_j = [
        m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=w_max_value, name=f"w_{j})")
        for j in range(m)
    ]

    # Auxiliary decision variabnles
    gamma_context = [
        m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=m + 1, name=f"gamma_{context})")
        for context in range(context_counts)
    ]

    # Coverage
    cov_jk = [
        [m.addVar(vtype=GRB.BINARY, name=f"cov_[{j}, {k}])") for k in range(s)]
        for j in range(m)
    ]
    covp_jk = [
        [m.addVar(vtype=GRB.BINARY, name=f"covp_[{j}, {k}])") for k in range(s)]
        for j in range(m)
    ]
    cov_k = [m.addVar(vtype=GRB.BINARY, name=f"cov_[{k}])") for k in range(s)]

    # Values
    opt_k = [m.addVar(vtype=GRB.BINARY, name=f"opt_[{k}])") for k in range(s)]
    w_jk = [
        [
            m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=w_max_value, name=f"w_[{j}, {k}])")
            for k in range(s)
        ]
        for j in range(m)
    ]

    m.setObjective(
        quicksum([gamma_context[context] for context in range(context_counts)]),
        GRB.MAXIMIZE,
    )

    # Constraints for weights
    for j in range(m):
        for k in range(s):
            m.addConstr(
                w_jk[j][k] <= big_m * cov_jk[j][k],
                name=f"w_[{j}, {k}] <= M * cov_[{j}, {k}]",
            )

    for j in range(m):
        for k in range(s):
            m.addConstr(
                w_jk[j][k] <= big_m * (1 - c_j[j]),
                name=f"w_[{j}, {k}] <= M * (1 - c_{j})",
            )

    for j in range(m):
        for k in range(s):
            m.addConstr(
                w_jk[j][k] <= w_j[j] + big_m * (1 - cov_jk[j][k]) + big_m * c_j[j],
                name=f"w_[{j}, {k}] <= w_{j} + M * (1 - cov_[{j}, {k}]) + M * c_{j}",
            )

    for j in range(m):
        for k in range(s):
            m.addConstr(
                w_jk[j][k] >= w_j[j] - big_m * (1 - cov_jk[j][k]) - big_m * c_j[j],
                name=f"w_[{j}, {k}] >= w_{j} - M * (1 - cov_[{j}, {k}]) - M * c_{j}",
            )

    # Constraints for gamma
    for k in range(s):
        m.addConstr(
            gamma_context[context_indices[k]]
            <= quicksum([w_jk[j][k] for j in range(m)]) + big_m * (1 - opt_k[k]),
            name=f"gamma_{context_indices[k]} <= SUM_j w_[j, {k}] + M * (1 - opt_{k})",
        )

    for k in range(s):
        m.addConstr(
            gamma_context[context_indices[k]]
            >= quicksum([w_jk[j][k] for j in range(m)]) + epsilon - big_m * opt_k[k],
            name=f"gamma_{context_indices[k]} >= SUM_j w_[j, {k}] + epsilon - M * opt_{k}",
        )

    # Constraints for coverage
    for j in range(m):
        for k in range(s):
            m.addConstr(
                covp_jk[j][k] >= 1 - c_j[j], name=f"covp_[{j}, {k}] >= 1 - c_{j}"
            )

    for j in range(m):
        for k in range(s):
            m.addConstr(
                covp_jk[j][k] >= cov_jk[j][k], name=f"covp_[{j}, {k}] >= cov_[{j}, {k}]"
            )

    for j in range(m):
        for k in range(s):
            m.addConstr(
                covp_jk[j][k] <= cov_jk[j][k] + (1 - c_j[j]),
                name=f"covp_[{j}, {k}] <= cov_[{j}, {k}] + (1 - c_{j})",
            )

    def x_kl(k, l):
        if l < n:
            return 1 if data[k][l] else 0
        else:
            return 0 if data[k][l - n] else 1

    for j in range(m):
        for k in range(s):
            for l in range(2 * n):
                m.addConstr(
                    cov_jk[j][k] >= a_jl[j][l] * x_kl(k, l),
                    name=f"cov_[{j}, {k}] >= a_[{j}, {l}] * {x_kl(k, l)}",
                )

    for j in range(m):
        for k in range(s):
            m.addConstr(
                cov_jk[j][k]
                <= quicksum([a_jl[j][l] * x_kl(k, l) for l in range(2 * n)]),
                name=f"cov_[{j}, {k}] <= SUM_l a_[{j}, l] * x_[{k}, l)",
            )

    for j in range(m):
        for l in range(n):
            m.addConstr(
                a_jl[j][l] + a_jl[j][n + l] <= 1,
                name=f"a_[{j}, {l}] + a_[{j}, {l + n}] <= 1",
            )

    for j in range(m):
        for k in range(s):
            m.addConstr(cov_k[k] <= covp_jk[j][k], name=f"cov_{k} <= covp_[{j}, {k}]")

    for k in range(s):
        m.addConstr(
            cov_k[k] >= quicksum([covp_jk[j][k] for j in range(m)]) - (m - 1),
            name=f"cov_{k} >= SUM_j covp_[j, {k}] - (m - 1)",
        )

    # Positive / negative constraints
    for k in range(s):
        if labels[k]:
            m.addConstr(cov_k[k] + opt_k[k] >= 2, name=f"cov_{k} + opt_{k} >= 2")
        else:
            m.addConstr(cov_k[k] + opt_k[k] <= 1, name=f"cov_{k} + opt_{k} <= 1")

    m.optimize()

    if m.status == GRB.Status.OPTIMAL:
        print("Found a solution")
        for j in range(m):
            clause = ", ".join([f"{a_jl[j][l]}" for l in range(2 * n)])
            print(f"{c_j[j].x}, {w_j[j].x}: " + clause)
    else:
        print("No solution found")


if __name__ == "__main__":
    learn_weighted_max_sat(2, np.array([]), np.array([]), [])
