from typing import Optional

import numpy as np
import logging

from .sample_models import generate_models
from .solve import solve_weighted_max_sat, get_value
from .learn import label_instance
from .type_def import Instance
from .learn import learn_weighted_max_sat
from .type_def import MaxSatModel


logger = logging.getLogger(__name__)


def model_to_string(model: MaxSatModel):
    # noinspection SpellCheckingInspection
    letters = "abcdefghijklmnopqrstuvwxyz".upper()

    def char(_l):
        if _l < 0:
            return f"!{letters[-_l - 1]}"
        else:
            return letters[_l - 1]

    result = ""
    for weight, clause in model:
        clause = " \\/ ".join(
            char(l) for l in sorted(clause, key=lambda x: (abs(x), 0 if x > 0 else 1))
        )
        result += (
            f"{'hard' if weight is None else 'soft'}, "
            f"{0.0 if weight is None else weight}: {clause}\n"
        )
    return result


def get_instance(n, model, context, positive, rng) -> Optional[Instance]:
    if positive:
        return solve_weighted_max_sat(n, model, context)
    else:
        instance = rng.rand(n) > 0.5
        for i in context:
            instance[abs(i) - 1] = i > 0
        if not label_instance(model, instance, context):
            return instance
        else:
            return None


def eval_hard_constraints(n: int, true_model: MaxSatModel, learned_model: MaxSatModel):
    pass


def eval_regret(n: int, true_model: MaxSatModel, learned_model: MaxSatModel):
    context = set()
    true_best_solution = solve_weighted_max_sat(n, true_model, context)
    true_best_value = get_value(true_model, true_best_solution)

    learned_best_solution = solve_weighted_max_sat(n, learned_model, context)
    learned_best_value = get_value(true_model, learned_best_solution)

    return true_best_value - learned_best_value if learned_best_value else -1


def learn_from_random_model():
    for seed in [666, 777, 888, 999]:
        rng = np.random.RandomState(seed)
        num_hard = 3
        num_soft = 3
        n = 5
        m = num_hard + num_soft
        contexts_count = 10
        models_and_contexts = generate_models(
            None, 1, contexts_count, n, 4, num_hard, num_soft, False, rng
        )
        true_model = [
            (w / 100 if w else None, clause)
            for w, clause in models_and_contexts[0]["model"]
        ]
        contexts = models_and_contexts[0]["contexts"]
        print(true_model)
        print(contexts[0])

        data = []
        labels = []
        contexts_to_learn = []

        for context in contexts:
            add_negative = True
            for positive in [True, False]:
                instance = get_instance(n, true_model, context, positive, rng)
                if positive and instance is None:
                    add_negative = False

                if (positive or add_negative) and instance is not None:
                    data.append(instance)
                    labels.append(positive)
                    contexts_to_learn.append(context)
                    print(instance, positive, context)
        print()

        data = np.array(data)
        labels = np.array(labels)

        learned_model = learn_weighted_max_sat(m, data, labels, contexts_to_learn)
        print(f"True Model\n{model_to_string(true_model)}")
        print(f"Learned model\n{model_to_string(learned_model)}")

        for k in range(data.shape[0]):
            instance = data[k, :]
            label = labels[k]
            learned_label = label_instance(
                learned_model, instance, contexts_to_learn[k]
            )
            if label != learned_label:
                logger.warning(
                    f"{instance} labeled {learned_label} instead of {label} (learned / true)"
                )

        print("REGRET ", eval_regret(n, true_model, learned_model))


if __name__ == "__main__":
    # logging.basicConfig(level=logging.INFO)
    learn_from_random_model()