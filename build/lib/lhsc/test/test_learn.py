import numpy as np
from .test_solve import example2_model

from lhsc.learn import label_instance


def test_label_instance():
    n, model = example2_model()
    assert label_instance(model, np.array([True, True]), set())
    assert label_instance(model, np.array([True, False]), set())
    assert not label_instance(model, np.array([False, True]), set())
    assert not label_instance(model, np.array([False, False]), set())
