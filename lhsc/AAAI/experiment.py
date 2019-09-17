#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 20:16:19 2019

@author: mohit
"""

from typing import Optional, List

import numpy as np
import logging
import pickle
import csv
import time
import argparse
import os

from .sample_models import generate_model
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


def context_to_string(context: Context):
    # noinspection SpellCheckingInspection
    letters = "abcdefghijklmnopqrstuvwxyz".upper()

    def char(_l):
        if _l < 0:
            return f"!{letters[-_l - 1]}"
        else:
            return letters[_l - 1]

    return " /\\ ".join(
        char(l) for l in sorted(context, key=lambda x: (abs(x), 0 if x > 0 else 1))
    )


def get_instance(n, model, context, positive, rng) -> Optional[Instance]:
    if positive:
        return solve_weighted_max_sat(n, model, context, 1)
    else:
        max_tries = 100
        for l in range(max_tries):
            instance = rng.rand(n) > 0.5

            for i in context:
                instance[abs(i) - 1] = i > 0
            if not label_instance(model, instance, context):
                return instance
        return None


def add_instances(n, model, context, positive, num_instances, prev_instances, rng):
    output = []
    if positive:
        for instance in solve_weighted_max_sat(n, model, context, num_instances):
            if not instance.tolist() in [a.tolist() for a in prev_instances]:
                output.append(instance)
            if len(output) >= num_instances - len(prev_instances):
                break
        output = np.array(output).astype(np.int)
        return output
    else:
        max_tries = 100 * (num_instances - len(prev_instances))
        for l in range(max_tries):
            instance = rng.rand(n) > 0.5
            for i in context:
                instance[abs(i) - 1] = i > 0
            if not label_instance(model, instance, context):
                if not instance.tolist() in [a.tolist() for a in prev_instances]:
                    output.append(instance)
                if len(output) >= num_instances - len(prev_instances):
                    break
        return output


def eval_hard_constraints(n: int, true_model: MaxSatModel, learned_model: MaxSatModel):
    pass



def eval_instance_regret(n: int, model: MaxSatModel, instance, context):
    #    context = set()
    true_best_solution = solve_weighted_max_sat(n, model, context, 1)
    true_best_value = get_value(model, true_best_solution)
    learned_best_value = get_value(model, instance)
    #    print(true_best_value, learned_best_value)
    return learned_best_value / true_best_value if true_best_value else 1


def get_data_set(
    n: int, model: MaxSatModel, contexts: List[Context], rng, allow_negative=False
):
    data = []
    labels = []
    contexts_to_learn = []
    for context in contexts:
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

    data = np.array(data).astype(np.int)
    labels = np.array(labels)
    return data, labels, contexts_to_learn


def random_contexts(
    n,
    num_contexts,
    prev_contexts,
    min_num_literals,
    max_num_literals,
    rng,
    max_tries=10,
):
    #    print(frozenset(c) for c in prev_contexts)
    result = set(frozenset(c) for c in prev_contexts)
    output = set()
    #    print("before:",result)
    for _ in range(max_tries * num_contexts):
        choices = list(range(n))
        num_literals = rng.randint(min_num_literals, max_num_literals + 1)
        selected = rng.choice(np.array(choices), size=num_literals, replace=False) + 1
        selected *= np.sign(rng.rand(len(selected)) - 0.5).astype(np.int)
        before = len(result)
        result.add(frozenset(set(selected)))
        if len(result) > before:
            output.add(frozenset(set(selected)))
        if len(result) >= num_contexts:
            break
    #    print("after:",result)
    return [set(c) for c in output]


def eval_precision_and_regret(
    target_model, learned_model, n, sample_size, context_ind, num_literal=None, rng=None
):
    correct_instances = []
    correct = 0
    num_sam = 0

    if context_ind == 0:
        context_less_instances = solve_weighted_max_sat(
            n, learned_model, set(), sample_size
        )
        num_sam = len(context_less_instances)
        if context_less_instances:
            for instance in context_less_instances:
                if get_value(target_model, instance) is not None:
                    correct_instances.append({"instance": instance, "context": set()})
                    correct += 1
    else:
        contexts = random_contexts(n, sample_size, [], 1, num_literal, rng)
        for context in contexts:
            instance = get_instance(n, learned_model, context, True, rng)
            if instance is not None:
                num_sam += 1
                if get_value(target_model, instance) is not None:
                    correct_instances.append({"instance": instance, "context": context})
                    correct += 1

    precision = correct / num_sam
    regret = 0
    for row in correct_instances:
        regret += eval_instance_regret(n, target_model, row["instance"], row["context"])
    regret = regret / len(correct_instances) if len(correct_instances) > 0 else ""

    return precision, regret, num_sam


def eval_recall(
    target_model, learned_model, n, sample_size, num_literal, prev_contexts, rng
):
    contexts = random_contexts(
        n, sample_size + len(prev_contexts), prev_contexts, 1, num_literal, rng
    )
    num_sam = 0
    correct = 0
    #    print(contexts)
    for context in contexts:
        instance = get_instance(n, target_model, context, True, rng)
        #        print(instance)
        if instance is not None:
            num_sam += 1
            if get_value(learned_model, instance) is not None:
                correct += 1
    return correct / num_sam



def generate_models(
    n,
    clause_length,
    num_hard,
    num_soft,
    context_counts,
    num_literal,
    model_seed,
    rng,
    model_seeds,
    num_vars,
):
    logging.basicConfig(level=logging.WARNING)

    m = num_hard + num_soft
    pickle_var = {}

    for context_count in context_counts:
        param = f"_model_seed_{model_seed}_num_hard_{num_hard}_num_soft_{num_soft}_n_{n}__clause_length_{clause_length}_context_count_{context_count}_num_literal_{num_literal}"
        print(param)
        if os.path.exists("pickles/" + param + ".pickle"):
            print("found it")
            pickle_old = pickle.load(open("pickles_v3/" + param + ".pickle", "rb"))
            true_model = pickle_old["true_model"]

        else:
            true_model = generate_model(n, clause_length, num_hard, num_soft, rng)
            true_model = [(w / 100 if w else None, clause) for w, clause in true_model]
        #    print(f"True Model\n{model_to_string(true_model)}")
        pickle_var["true_model"] = true_model

        prev_context = []
        data = []
        labels = []
        learning_contexts = []

        num_infeasible_context = 0
        if len(prev_context) < context_count:
            contexts = random_contexts(
                n, context_count, prev_context, 1, num_literal, rng
            )
            tmp_data, tmp_labels, tmp_learning_contexts = get_data_set(
                n, true_model, contexts, rng, allow_negative=True
            )
            if len(data) > 0:
                data = np.append(data, tmp_data, axis=0)
                labels = np.append(labels, tmp_labels, axis=0)
                learning_contexts.extend(tmp_learning_contexts)
            else:
                data = tmp_data
                labels = tmp_labels
                learning_contexts = tmp_learning_contexts
            prev_context.extend(contexts)
            contexts = prev_context
        else:
            contexts = random_contexts(n, context_count, 2, num_literal, rng)
            prev_context = contexts

        num_instances = 5
        for i, context in enumerate(learning_contexts):
            prev_instances = [
                data[j] for j, x in enumerate(learning_contexts) if x == context
            ]
            for positive in [True, False]:
                instances = add_instances(
                    n, true_model, context, positive, num_instances, prev_instances, rng
                )
                for instance in instances:
                    data = np.append(data, [instance], axis=0)
                    labels = np.append(labels, [positive], axis=0)
                    learning_contexts.append(context)

        start = time.time()
        learned_model = learn_weighted_max_sat(m, data, labels, learning_contexts)
        end = time.time()
        #        print(f"Learned model\n{model_to_string(learned_model)}")
        pickle_var["learned_model"] = learned_model
        sample_size = 100 * (5 ** int(n / 10))

        if learned_model:
            param = f"_model_seed_{model_seed}_num_hard_{num_hard}_num_soft_{num_soft}_n_{n}__clause_length_{clause_length}_context_count_{context_count}_num_literal_{num_literal}_num_instances_{num_instances}"
            pickle.dump(pickle_var, open("pickles/" + param + ".pickle", "wb"))
            recall = eval_recall(
                true_model,
                learned_model,
                n,
                sample_size,
                num_literal,
                learning_contexts,
                rng,
            )
            precision, regret, num_sam = eval_precision_and_regret(
                true_model, learned_model, n, sample_size, 0, num_literal=num_literal
            )
            precision_c, regret_c, num_sam_c = eval_precision_and_regret(
                true_model,
                learned_model,
                n,
                sample_size,
                1,
                num_literal=num_literal,
                rng=rng,
            )
            csvfile = open(
                "results/evaluations_"
                + "".join(map(str, model_seeds))
                + "_"
                + "".join(map(str, num_vars))
                + ".csv",
                "a",
            )
            filewriter = csv.writer(csvfile, delimiter=",")
            row = [
                model_seed,
                num_hard,
                num_soft,
                n,
                clause_length,
                context_count,
                num_literal,
                end - start,
                data.shape[0],
                num_infeasible_context,
                recall,
                precision,
                regret,
                precision_c,
                regret_c,
            ]
            filewriter.writerow(row)
            csvfile.close()


def exp_generate_models(model_seeds, num_vars):
    #    logging.basicConfig(level=logging.INFO)
    context_counts = [2, 5, 10]

    csvfile = open(
        "results/evaluations_"
        + "".join(map(str, model_seeds))
        + "_"
        + "".join(map(str, num_vars))
        + ".csv",
        "w",
    )
    filewriter = csv.writer(csvfile, delimiter=",")
    filewriter.writerow(
        [
            "model_seed",
            "num_hard",
            "num_soft",
            "n",
            "clause_length",
            "context_count",
            "num_literal",
            "time",
            "num_example",
            "num_infeasible_context",
            "recall",
            "precision",
            "regret",
            "precision_c",
            "regret_c",
        ]
    )
    csvfile.close()
    for model_seed in model_seeds:
        rng = np.random.RandomState(model_seed)
        for n in num_vars:
            num_literal = 2
            for clause_length in [2, 5]:
                for num_hard in [0, 5, 10]:
                    for num_soft in [5, 10]:
                        #                            for context_count in context_counts:
                        try:
                            generate_models(
                                n,
                                clause_length,
                                num_hard,
                                num_soft,
                                context_counts,
                                num_literal,
                                model_seed,
                                rng,
                                model_seeds,
                                num_vars,
                            )
                        except AssertionError as error:
                            print(
                                f"_model_seed_{model_seed}_num_hard_{num_hard}_num_soft_{num_soft}_n_{n}__clause_length_{clause_length}_num_literal_{num_literal}"
                            )
                            continue


if __name__ == "__main__":

    CLI = argparse.ArgumentParser()
    CLI.add_argument(
        "--model_seeds",  # name on the CLI - drop the `--` for positional/required parameters
        nargs="*",  # 0 or more values expected => creates a list
        type=int,
        default=[111, 222, 333, 444, 555],  # default if nothing is provided
    )
    CLI.add_argument(
        "--num_vars",  # name on the CLI - drop the `--` for positional/required parameters
        nargs="*",  # 0 or more values expected => creates a list
        type=int,
        default=[5, 10, 15],  # default if nothing is provided
    )
    args = CLI.parse_args()

    exp_generate_models(args.model_seeds, args.num_vars)
