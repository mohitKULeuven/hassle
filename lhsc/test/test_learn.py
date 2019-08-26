import numpy as np

from .test_solve import example2_model

from lhsc.learn import label_instance, check_learned_model


def test_label_instance():
    n, model = example2_model()
    assert label_instance(model, np.array([True, True]), set())
    assert label_instance(model, np.array([True, False]), set())
    assert not label_instance(model, np.array([False, True]), set())
    assert not label_instance(model, np.array([False, False]), set())


def test_check_learned_model():
    # Example
    #      A \/ B
    # 1.0: A

    n, model = example2_model()
    data = np.array([[True, True], [True, False], [False, True], [False, False]])
    labels = np.array(
        [label_instance(model, data[k, :], set()) for k in range(data.shape[0])]
    )
    contexts = [set() for k in range(data.shape[0])]
    assert check_learned_model(model, data, labels, contexts)

    new_labels = labels.copy()
    new_labels[len(labels) - 1] = not labels[len(labels) - 1]
    assert not check_learned_model(model, data, new_labels, contexts)

    new_labels = labels.copy()
    new_labels[0] = not labels[0]
    assert not check_learned_model(model, data, new_labels, contexts)
