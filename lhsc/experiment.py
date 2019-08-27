from typing import Optional, List

import numpy as np
import logging

from .sample_models import generate_model, generate_contexts
from .solve import solve_weighted_max_sat, get_value
from .learn import label_instance
from .type_def import Instance, MaxSatModel, Context
from .learn import learn_weighted_max_sat


logger = logging.getLogger(__name__)


def model_to_string(model: MaxSatModel):
    # noinspection SpellCheckingInspection
    letters = "abcdefghijklmnopqrstuvwxyz".upper()

    if model is None:
        return "No model."

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
        max_tries = 100
        for _ in range(max_tries):
            instance = rng.rand(n) > 0.5
            for i in context:
                instance[abs(i) - 1] = i > 0
            if not label_instance(model, instance, context):
                return instance
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


def get_data_set(
    n: int, model: MaxSatModel, contexts: List[Context], rng, allow_negative=False
):
    data = []
    labels = []
    contexts_to_learn = []
    for context in contexts:
        # print("Context:", context)
        add_negative = True
        for positive in [True, False]:
            instance = get_instance(n, model, context, positive, rng)
            if positive and instance is None:
                add_negative = False or allow_negative

            if (positive or add_negative) and instance is not None:
                data.append(instance)
                labels.append(positive)
                contexts_to_learn.append(context)
                # print(instance, positive, context)

    data = np.array(data)
    labels = np.array(labels)
    return data, labels, contexts_to_learn


def learn_from_random_model():
    model_seeds = [111, 222, 333, 444, 555, 666, 777, 888, 999]
    context_seed_start = [101010]
    for model_seed in model_seeds:
        rng = np.random.RandomState(model_seed)
        num_hard = 3
        num_soft = 3
        n = 10
        m = num_hard + num_soft
        clause_length = int(4 / 5 * n)

        true_model = generate_model(n, clause_length, num_hard, num_soft, rng)
        true_model = [(w / 100 if w else None, clause) for w, clause in true_model]
        print(f"True Model\n{model_to_string(true_model)}")

        contexts_count = 50
        rng = np.random.RandomState(context_seed_start)
        contexts = generate_contexts(true_model, contexts_count, 1, n, rng)

        data, labels, learning_contexts = get_data_set(
            n, true_model, contexts, rng, allow_negative=True
        )

        learned_model = learn_weighted_max_sat(m, data, labels, learning_contexts)
        print(f"Learned model\n{model_to_string(learned_model)}")

        for k in range(data.shape[0]):
            instance = data[k, :]
            label = labels[k]
            learned_label = label_instance(
                learned_model, instance, learning_contexts[k]
            )
            if label != learned_label:
                logger.warning(
                    f"{instance} labeled {learned_label} instead of {label} (learned / true)"
                )

        print("REGRET ", eval_regret(n, true_model, learned_model))


def learn_hard_constraints():
    logging.basicConfig(level=logging.INFO)
    rng = np.random.RandomState(4)
    n = 3
    true_model = [(None, {-1, -2}), (1.0, {3})]
    contexts = [set(), {1}, {-1}, {2}, {-2}, {3}, {-3}, {1, 2}]
    data, labels, learning_contexts = get_data_set(
        n, true_model, contexts, rng, allow_negative=True
    )
    for k in range(len(labels)):
        print(data[k, :], labels[k], learning_contexts[k])
    learned_model = learn_weighted_max_sat(
        len(true_model), data, labels, learning_contexts
    )
    print("True model")
    print(model_to_string(true_model))
    print("Learned model")
    print(model_to_string(learned_model))

    if learned_model:
        for k in range(data.shape[0]):
            instance = data[k, :]
            label = labels[k]
            learned_label = label_instance(
                learned_model, instance, learning_contexts[k]
            )
            if label != learned_label:
                logger.warning(
                    f"{instance} labeled {learned_label} instead of {label} (learned / true)"
                )

        print(f"REGRET {eval_regret(n, true_model, learned_model)}")


if __name__ == "__main__":
    # logging.basicConfig(level=logging.INFO)
    learn_from_random_model()
    # learn_hard_constraints()
