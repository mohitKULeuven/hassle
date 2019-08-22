import numpy as np

from typing import Tuple

from lhsc.solve import solve_weighted_max_sat
from lhsc.type_def import MaxSatModel


def example2_model() -> Tuple[int, MaxSatModel]:
    # Example
    #      A \/ B
    # 1.0: A
    return 2, [(None, {1, 2}), (1.0, {1})]


def test_example2():
    n, model = example2_model()

    # Global context
    contexts = [set(), {1}, {-1}, {2}, {-2}, {-1, -2}]
    solutions_list = [
        [[1, 1], [1, 0]],
        [[1, 1], [1, 0]],
        [[0, 1]],
        [[1, 1]],
        [[1, 0]],
        [],
    ]

    for context, solutions in zip(contexts, solutions_list):
        instance = solve_weighted_max_sat(n, model, context)
        if len(solutions) > 0:
            print(instance)
            print(*[np.array(s) for s in solutions], sep="\n")
            assert any(all(instance == np.array(s)) for s in solutions)
        else:
            assert instance is None
