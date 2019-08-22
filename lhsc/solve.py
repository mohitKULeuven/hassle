from .type_def import (
    MaxSatModel,
    Clause,
    Model,
    GRB,
    quicksum,
    suppress_stdout,
    Instance,
)


def solve_weighted_max_sat(n: int, model: MaxSatModel, context: Clause) -> Instance:
    with suppress_stdout():
        mod = Model("MaxSat")

    mod.setParam("OutputFlag", False)

    m = len(model)

    # Variables
    x_l = [mod.addVar(vtype=GRB.BINARY, name=f"x_{l})") for l in range(n)]
    cov_j = [mod.addVar(vtype=GRB.BINARY, name=f"cov_{j})") for j in range(m)]
    w_j = [
        mod.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name=f"w_{j}") for j in range(m)
    ]

    mod.setObjective(quicksum([w_j[j] for j in range(m)]), GRB.MAXIMIZE)

    for j, (weight, clause) in enumerate(model):
        # FOR-ALL_l: cov_j >= x_l * a_jl+ OR cov_j >= (1 - x_l) * a_jl-
        for i in clause:
            if i < 0:
                l = abs(i) - 1
                mod.addConstr(cov_j[j] >= (1 - x_l[l]), name=f"cov_{j} >= (1 - x_{l})")
            else:
                l = i - 1
                mod.addConstr(cov_j[j] >= x_l[l], name=f"cov_{j} >= x_{l}")

        # Force it to be false if no literal is satisfied
        mod.addConstr(
            cov_j[j]
            <= quicksum((1 - x_l[abs(i) - 1]) if i < 0 else x_l[i - 1] for i in clause),
            name=f"cov_{j} <= SUM_l clause_{j}",
        )

        if weight is None:
            # Constraint is hard, thus weight is 0
            mod.addConstr(w_j[j] == 0, name=f"w_{j} = 0")

            # Constraint is hard, thus must be covered
            mod.addConstr(cov_j[j] >= 1, name=f"cov_{j} >= 1")
        else:
            # Weight is 0 if clause is not covered, and weight otherwise
            mod.addConstr(w_j[j] == weight * cov_j[j], name=f"w_{j} <= {weight}")

    for i in context:
        # Fix values given by context
        if i < 0:
            l = abs(i) - 1
            mod.addConstr(x_l[l] == 0, name=f"x_{l} = 0")
        else:
            l = i - 1
            mod.addConstr(x_l[l] == 1, name=f"x_{l} = 1")

    mod.optimize()

    if mod.status == GRB.Status.OPTIMAL:
        print("Found a solution")
        print(
            f"{sum(w_j[j].x for j in range(m))}:",
            *[f"{x_l[l].x}" for l in range(n)],
            sep=" ",
        )
    else:
        print("No solution found")
        return None
