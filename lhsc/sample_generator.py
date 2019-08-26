#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 09:51:26 2019

@author: mohit
"""

from gurobipy import *
import random
import numpy as np
import itertools
from sklearn import preprocessing
import pickle

# checks for unsatisfiable cnfs
def is_feasible(lst, n):
    cnf = list(lst).copy()
    for l, val in enumerate(cnf):
        if val >= n:
            cnf[l] = val - n
    if len(cnf) != len(set(cnf)):
        return False
    return True


def generate_models(num_var, k, num_hard, num_soft, seed, file_name, num_models):

    random.seed(seed)
    N = list(range(num_var))
    cnfs = list(itertools.combinations(list(range(num_var * 2)), k))
    for cnf in cnfs:
        if not is_feasible(cnf, num_var):
            cnfs.remove(cnf)
    models = [
        {"num_var": num_var, "k": k, "num_hard": num_hard, "num_soft": num_soft}
        for i in range(num_models)
    ]
    constraint_indices = list(range(len(cnfs)))
    for i in range(num_models):
        hard_indices = random.sample(constraint_indices, num_hard)
        soft_indices = random.sample(
            [item for item in constraint_indices if item not in hard_indices], num_soft
        )
        weights = np.zeros(len(cnfs))
        for l in soft_indices:
            weights[l] = random.random()
        normalized_weights = preprocessing.normalize(weights.reshape(1, -1), norm="l2")[
            0
        ]
        models[i]["hard_indices"] = hard_indices
        models[i]["soft_indices"] = soft_indices
        models[i]["weights"] = normalized_weights
    pickle.dump(models, open(file_name, "wb"))


def generate_samples(file_name, num_sample, seed, percent_observed):

    models = pickle.load(open(file_name, "rb"))
    for model in models:
        num_var = model["num_var"]
        k = model["k"]
        hard_indices = model["hard_indices"]
        soft_indices = model["soft_indices"]
        normalized_weights = model["weights"]
        N = list(range(num_var))
        cnfs = list(itertools.combinations(list(range(num_var * 2)), k))
        for cnf in cnfs:
            if not is_feasible(cnf, num_var):
                cnfs.remove(cnf)
        constraint_indices = list(range(len(cnfs)))
        random.seed(seed)
        observed_indices = random.sample(N, int(num_var * percent_observed))
        print("####################################")
        print("Number of constraints: ", len(cnfs))  # nCk * 2^k
        print("hard constraints: ", hard_indices)
        print("soft constraints: ", soft_indices)
        print("weights: ", [w for w in normalized_weights if w > 0])

        try:
            m = Model("sample_generator")
            m.setParam(GRB.Param.OutputFlag, 0)

            x = m.addVars(N, vtype=GRB.BINARY, name="x")
            neg_x = m.addVars(N, vtype=GRB.BINARY, name="neg_x")
            satisfaction = m.addVars(constraint_indices, vtype=GRB.BINARY, name="sat")

            m.addConstrs(
                (x[i] == random.randint(0, 1) for i in observed_indices), "observed"
            )

            m.addConstrs((neg_x[n] == 1 - x[n] for n in N), "negations")
            for l, cnf in enumerate(cnfs):
                if l in soft_indices:
                    m.addConstr(
                        (
                            quicksum(x[i] for i in cnf if i < num_var)
                            + quicksum(neg_x[i - num_var] for i in cnf if i >= num_var)
                            - k
                            + 1
                            == satisfaction[l]
                        ),
                        "soft",
                    )
                if l in hard_indices:
                    m.addConstr(
                        (
                            quicksum(x[i] for i in cnf if i < num_var)
                            + quicksum(neg_x[i - num_var] for i in cnf if i >= num_var)
                            == k
                        ),
                        "hard",
                    )

            m.setObjective(
                quicksum(
                    satisfaction[i] * normalized_weights[i] * normalized_weights[i]
                    for i in soft_indices
                ),
                GRB.MAXIMIZE,
            )
            m.setParam(GRB.Param.PoolSearchMode, 2)
            m.setParam(GRB.Param.PoolSolutions, num_sample)
            m.optimize()
            nSolutions = m.SolCount
            print("Number of solutions found: " + str(nSolutions))
            if m.status == GRB.Status.INFEASIBLE:
                m.computeIIS()
                print("\nThe following constraint(s) cannot be satisfied:")
                for c in m.getConstrs():
                    if c.IISConstr:
                        print("%s" % c.constrName)
            if m.status == GRB.Status.OPTIMAL:
                m.write("m.sol")

            for i in range(nSolutions):
                m.setParam(GRB.Param.SolutionNumber, i)
                for v in m.getVars():
                    if "x" in v.varName and "neg" not in v.varName:
                        print("%s %g" % (v.varName, v.x))
                print("Obj: %g" % m.objVal)

        except GurobiError as e:
            print("Error code " + str(e.errno) + ": " + str(e))

        except AttributeError:
            print("Encountered an attribute error")
