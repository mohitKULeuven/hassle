import numpy as np

from typing import Tuple

import pytest

from lhsc.solve import solve_weighted_max_sat, get_value
from lhsc.type_def import MaxSatModel


DELTA = 10 * (-10)


def example2_model() -> Tuple[int, MaxSatModel]:
    # Example
    #      A \/ B
    # 1.0: A
    return 2, [(None, {1, 2}), (1.0, {1})]


def example_rand1_model() -> Tuple[int, MaxSatModel]:
    n = 6
    model = [
        (None, {2, -5, 4, -3}),
        (None, {3, -4, 5, -1}),
        (None, {5}),
        (29, {1, -5, -3, -2}),
        (15, {1, -4, 5}),
        (98, {2, 4, 5}),
    ]
    return n, model


def test_example_rand1_instance_value():
    n, model = example_rand1_model()
    instance = np.array([False, True, True, False, True, True])
    assert 113 == get_value(model, instance)


def test_solve_with_too_large_weight():
    n, model = 1, [(2, {1})]
    with pytest.raises(AttributeError):
        solve_weighted_max_sat(n, model, set())


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


def test_get_value():
    n, model = example2_model()
    assert get_value(model, np.array([True, True])) == pytest.approx(1.0, DELTA)
    assert get_value(model, np.array([True, False])) == pytest.approx(1.0, DELTA)
    assert get_value(model, np.array([False, True])) == pytest.approx(0.0, DELTA)
    assert get_value(model, np.array([False, False])) is None
