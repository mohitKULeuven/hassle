# noinspection PyUnresolvedReferences
import os
import logging
from typing import List

import numpy as np

from .solve import solve_weighted_max_sat, get_value
from .type_def import MaxSatModel, Clause, suppress_stdout, Instance, Context
from gurobipy import Model, GRB, quicksum


logger = logging.getLogger(__name__)


def learn_weighted_max_sat(
    m: int, data: np.ndarray, labels: np.ndarray, contexts: List[Clause]
) -> MaxSatModel:
    """
    Learn a weighted MaxSAT model from examples. Contexts and clauses are set-encoded, i.e., they are represented by
    sets containing positive or negative integers in the range -n-1 to n+1. If a set contains an positive integer i, the i-1th
     Boolean feature is set to True, if it contains a negative integer -i, the i-1th Boolean feature is set to False.
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
    epsilon = 10 ** (-3)

    context_pool = dict()
    context_indices = []
    for context in contexts:
        key = frozenset(context)
        if key not in context_pool:
            context_pool[key] = len(context_pool)
        context_indices.append(context_pool[key])
    context_counts = len(context_pool)

    logger.debug("Learn wMaxSAT")
    logger.debug("w_max", w_max_value)
    logger.debug("s", s)
    logger.debug("n", n)
    logger.debug("m", m)

    with suppress_stdout():
        mod = Model("LearnMaxSat")

    mod.setParam("OutputFlag", False)

    # Constraint decision variables
    c_j = [mod.addVar(vtype=GRB.BINARY, name=f"c_{j})") for j in range(m)]
    a_jl = [
        [mod.addVar(vtype=GRB.BINARY, name=f"a_[{j}, {l}]") for l in range(2 * n)]
        for j in range(m)
    ]

    # Weights decision variables
    w_j = [
        mod.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=w_max_value, name=f"w_{j})")
        for j in range(m)
    ]

    # Auxiliary decision variabnles
    gamma_context = [
        mod.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=m + 1, name=f"gamma_{context})")
        for context in range(context_counts)
    ]

    # Coverage
    cov_jk = [
        [mod.addVar(vtype=GRB.BINARY, name=f"cov_[{j}, {k}])") for k in range(s)]
        for j in range(m)
    ]
    covp_jk = [
        [mod.addVar(vtype=GRB.BINARY, name=f"covp_[{j}, {k}])") for k in range(s)]
        for j in range(m)
    ]
    cov_k = [mod.addVar(vtype=GRB.BINARY, name=f"cov_[{k}])") for k in range(s)]

    # Values
    opt_k = [mod.addVar(vtype=GRB.BINARY, name=f"opt_[{k}])") for k in range(s)]
    w_jk = [
        [
            mod.addVar(
                vtype=GRB.CONTINUOUS, lb=0, ub=w_max_value, name=f"w_[{j}, {k}])"
            )
            for k in range(s)
        ]
        for j in range(m)
    ]

    mod.setObjective(
        quicksum([gamma_context[context] for context in range(context_counts)]),
        GRB.MAXIMIZE,
    )

    # Constraints for weights
    for j in range(m):
        for k in range(s):
            mod.addConstr(
                w_jk[j][k] <= big_m * cov_jk[j][k],
                name=f"w_[{j}, {k}] <= M * cov_[{j}, {k}]",
            )

    for j in range(m):
        for k in range(s):
            mod.addConstr(
                w_jk[j][k] <= big_m * (1 - c_j[j]),
                name=f"w_[{j}, {k}] <= M * (1 - c_{j})",
            )

    for j in range(m):
        for k in range(s):
            mod.addConstr(
                w_jk[j][k] <= w_j[j] + big_m * (1 - cov_jk[j][k]) + big_m * c_j[j],
                name=f"w_[{j}, {k}] <= w_{j} + M * (1 - cov_[{j}, {k}]) + M * c_{j}",
            )

    for j in range(m):
        for k in range(s):
            mod.addConstr(
                w_jk[j][k] >= w_j[j] - big_m * (1 - cov_jk[j][k]) - big_m * c_j[j],
                name=f"w_[{j}, {k}] >= w_{j} - M * (1 - cov_[{j}, {k}]) - M * c_{j}",
            )

    # Constraints for gamma
    for k in range(s):
        mod.addConstr(
            gamma_context[context_indices[k]]
            <= quicksum([w_jk[j][k] for j in range(m)]) + big_m * (1 - opt_k[k]),
            name=f"gamma_{context_indices[k]} <= SUM_j w_[j, {k}] + M * (1 - opt_{k})",
        )

    for k in range(s):
        mod.addConstr(
            gamma_context[context_indices[k]]
            >= quicksum([w_jk[j][k] for j in range(m)]) + epsilon - big_m * opt_k[k],
            name=f"gamma_{context_indices[k]} >= SUM_j w_[j, {k}] + epsilon - M * opt_{k}",
        )

    # Constraints for coverage
    for j in range(m):
        for k in range(s):
            mod.addConstr(
                covp_jk[j][k] >= 1 - c_j[j], name=f"covp_[{j}, {k}] >= 1 - c_{j}"
            )

    for j in range(m):
        for k in range(s):
            mod.addConstr(
                covp_jk[j][k] >= cov_jk[j][k], name=f"covp_[{j}, {k}] >= cov_[{j}, {k}]"
            )

    for j in range(m):
        for k in range(s):
            mod.addConstr(
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
                mod.addConstr(
                    cov_jk[j][k] >= a_jl[j][l] * x_kl(k, l),
                    name=f"cov_[{j}, {k}] >= a_[{j}, {l}] * {x_kl(k, l)}",
                )

    for j in range(m):
        for k in range(s):
            mod.addConstr(
                cov_jk[j][k]
                <= quicksum([a_jl[j][l] * x_kl(k, l) for l in range(2 * n)]),
                name=f"cov_[{j}, {k}] <= SUM_l a_[{j}, l] * x_[{k}, l)",
            )

    for j in range(m):
        for l in range(n):
            mod.addConstr(
                a_jl[j][l] + a_jl[j][n + l] <= 1,
                name=f"a_[{j}, {l}] + a_[{j}, {l + n}] <= 1",
            )

    for j in range(m):
        for k in range(s):
            mod.addConstr(cov_k[k] <= covp_jk[j][k], name=f"cov_{k} <= covp_[{j}, {k}]")

    for k in range(s):
        mod.addConstr(
            cov_k[k] >= quicksum([covp_jk[j][k] for j in range(m)]) - (m - 1),
            name=f"cov_{k} >= SUM_j covp_[j, {k}] - (m - 1)",
        )

    # mod.addConstr(c_j[0] >= 1, name="Forcing the first constraint to be hard")

    # Positive / negative constraints
    for k in range(s):
        if labels[k]:
            mod.addConstr(cov_k[k] + opt_k[k] >= 2, name=f"cov_{k} + opt_{k} >= 2")
        else:
            mod.addConstr(cov_k[k] + opt_k[k] <= 1, name=f"cov_{k} + opt_{k} <= 1")

    mod.optimize()

    if mod.status == GRB.Status.OPTIMAL:

        def char(_i):
            return (" " if _i < n else "!") + "abcdefghijklmnop"[
                abs(_i % n)
            ].capitalize()

        def char_feature(_i, val):
            return (" " if val else "!") + "abcdefghijklmnop"[abs(_i)].capitalize()

        # print("Found a solution")
        # for j in range(m):
        #     clause = " \\/ ".join([char(l) for l in range(2 * n) if a_jl[j][l].x])
        #     print(f"{'hard' if c_j[j].x else 'soft'}, {w_j[j].x}: " + clause)

        # print("Constraints")
        # for j in range(m):
        #     print(*[a_jl[j][l].x for l in range(2 * n)])

        logger.info("Learning results")
        for k in range(s):
            logger.info(
                " : ".join(
                    [
                        (" sat" if cov_k[k].x else "!sat"),
                        (" opt" if opt_k[k].x else "!opt"),
                        " + ".join([f"{w_jk[j][k].x}" for j in range(m)])
                        + f" = {sum(w_jk[j][k].x for j in range(m))}",
                        f"gamma_{context_indices[k]} {gamma_context[context_indices[k]].x}",
                        ("pos " if labels[k] else "neg ")
                        + ",".join(char_feature(i, data[k][i]) for i in range(n)),
                        ", ".join(
                            (" sat'" if covp_jk[j][k].x else "!sat'") for j in range(m)
                        ),
                    ]
                )
            )

        return [
            (
                None if c_j[j].x else w_j[j].x,
                {
                    l + 1 if l < n else -(l - n + 1)
                    for l in range(2 * n)
                    if a_jl[j][l].x
                },
            )
            for j in range(m)
        ]
    else:
        pass


def label_instance(model: MaxSatModel, instance: Instance, context: Context) -> bool:
    value = get_value(model, instance)
    if value is None:
        return False
    best_instance = solve_weighted_max_sat(len(instance), model, context)
    return value >= get_value(model, best_instance)


def check_learned_model(
    model: MaxSatModel, data: np.ndarray, labels: np.ndarray, contexts: List[Context]
):
    s = data.shape[0]
    assert s == len(labels) == len(contexts)
    for k in range(s):
        if labels[k] != label_instance(model, data[k, :], contexts[k]):
            return False
    return True


def example2():
    # Example
    #      A \/ B
    # 1.0: A
    #
    # --:  A, B   sat  opt (1)
    # --:  A,!B   sat  opt (1)
    # --: !A, B   sat !opt (0)
    # --: !A,!B  !sat !opt (0)
    #
    #  A:  A, B   sat  opt (1)
    #  A:  A,!B   sat  opt (1)
    #
    # !A: !A, B   sat  opt (0)
    # !A: !A,!B  !sat  opt (0)
    #
    #  B:  A, B   sat  opt (1)
    #  B: !A, B   sat !opt (0)
    #
    # !B:  A,!B   sat  opt (1)
    # !B: !A,!B  !sat !opt (0)

    data = np.array(
        [
            [True, True],
            [True, False],
            [False, True],
            [False, False],
            [True, True],
            [True, False],
            [False, True],
            [False, False],
            [True, True],
            [False, True],
            [True, False],
            [False, False],
        ]
    )

    labels = np.array(
        [True, True, False, False, True, True, True, False, True, False, True, False]
    )
    contexts = [set(), set(), set(), set(), {1}, {1}, {-1}, {-1}, {2}, {2}, {-2}, {-2}]
    learn_weighted_max_sat(2, data, labels, contexts)


def example3():
    # Example
    #      !A \/ !B \/ !C
    # 1.0: A
    # 0.5: B \/ !C
    #
    # pos  A, B,!C  A
    # neg  A,!B, C  A suboptimal
    # neg  A, B, C  A infeasible
    #
    # pos !A, B,!C !A
    # neg !A,!B, C !A suboptimal
    #
    # pos  A, B,!C  B
    # neg !A, B, C  B suboptimal
    # neg  A, B, C  B infeasible
    #
    # pos  A,!B,!C !B
    # neg !A,!B, C !B suboptimal
    #
    # pos  A,!B, C  C
    # neg !A, B, C  C suboptimal
    # neg  A, B, C  C infeasible
    #
    # pos  A, B,!C !C
    # neg !A,!B,!C !C suboptimal
    #
    # pos !A,!B,!C  !A,!B
    # pos  A,!B,!C  !B,!C
    # pos  !A,B,C  B,C

    data = np.array(
        [
            [True, True, False],
            [True, False, True],
            [True, True, True],
            [False, True, False],
            [False, False, True],
            [True, True, False],
            [False, True, True],
            [True, True, True],
            [True, False, False],
            [False, False, True],
            [True, False, True],
            [False, True, True],
            [True, True, True],
            [True, True, False],
            [False, False, False],
            [False, False, False],
            [True, False, False],
            [False, True, True],
        ]
    )

    labels = np.array(
        [
            True,
            False,
            False,
            True,
            False,
            True,
            False,
            False,
            True,
            False,
            True,
            False,
            False,
            True,
            False,
            True,
            True,
            True,
        ]
    )

    contexts = [
        {1},
        {1},
        {1},
        {-1},
        {-1},
        {2},
        {2},
        {2},
        {-2},
        {-2},
        {3},
        {3},
        {3},
        {-3},
        {-3},
        {-1, -2},
        {-2, -3},
        {2, 3},
    ]

    learn_weighted_max_sat(3, data, labels, contexts)


def solve_example2():
    # Example
    #      A \/ B
    # 1.0: A

    print("Global context")
    solve_weighted_max_sat(2, [(None, {1, 2}), (1.0, {1})], set())

    print("Context:  A")
    solve_weighted_max_sat(2, [(None, {1, 2}), (1.0, {1})], {1})

    print("Context: !A")
    solve_weighted_max_sat(2, [(None, {1, 2}), (1.0, {1})], {-1})

    print("Context:  B")
    solve_weighted_max_sat(2, [(None, {1, 2}), (1.0, {1})], {2})

    print("Context: !B")
    solve_weighted_max_sat(2, [(None, {1, 2}), (1.0, {1})], {-2})

    print("Context: !A, !B")
    solve_weighted_max_sat(2, [(None, {1, 2}), (1.0, {1})], {-1, -2})


if __name__ == "__main__":
    solve_example2()
