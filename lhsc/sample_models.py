# -*- coding: utf-8 -*-

import numpy as np
import itertools as it
import logging

from pysat.examples.fm import FM
from pysat.formula import WCNF
from scipy.special import binom
from typing import List

from type_def import MaxSatModel


logger = logging.getLogger(__name__)

# XXX some solvers support integer weights only, let's do that
_MIN_WEIGHT, _MAX_WEIGHT = 1, 100


def is_entailed(wcnf, clause):
    wcnf_new = wcnf.copy()
    for literal in clause:
        wcnf_new.append((-literal,))
    fm = FM(wcnf_new, verbose=0)
    #    print(wcnf_new.hard,fm.compute())
    return not fm.compute()


def _generate_all_clauses_up_to_length(num_vars, length):
    flip_or_dont = lambda v: -(v - num_vars) if v > num_vars else v

    lits = range(1, 2 * num_vars + 1)
    clauses = set(
        [
            tuple(set(map(flip_or_dont, clause)))
            for clause in it.combinations_with_replacement(lits, length)
        ]
    )

    # This makes sure that all symmetries are accounted for...
    must_be = sum(binom(2 * num_vars, l) for l in range(1, length + 1))
    assert len(clauses) == must_be

    # check entailment property of the added constraints

    # ... except for impossible clauses like 'x and not x', let's delete them
    def possible(clause):
        for i in range(len(clause)):
            for j in range(i + 1, len(clause)):
                if clause[i] == -clause[j]:
                    return False
        return True

    return list(sorted(filter(possible, clauses)))


def generate_wcnfs(path, models_and_contexts):

    for i, elem in enumerate(models_and_contexts):
        model = elem["model"]
        contexts = elem["contexts"]

        wcnf = WCNF()
        for weight_clause in model:
            if weight_clause[0] == None:
                wcnf.append(tuple(weight_clause[1]))
            else:
                wcnf.append(tuple(weight_clause[1]), weight=weight_clause[0])
        wcnf.to_file(path + f"_{i}.wcnf")

        for j, context in enumerate(contexts):
            wcnf_context = wcnf.copy()
            #            print(context)
            for literals in context:
                wcnf_context.append((literals,))
            wcnf_context.to_file(path + f"_{i}_context_{j}.wcnf")


def get_random_clauses(wcnf, rng, clauses, n):
    #    wcnf = WCNF()
    selected_indices = []
    checked_indices = []
    while n > 0:
        indices = [ind for ind in range(len(clauses)) if ind not in checked_indices]
        i = rng.choice(indices)
        checked_indices.append(i)
        if not is_entailed(wcnf, clauses[i]):
            wcnf.append(clauses[i])
            selected_indices.append(i)
            n = n - 1
        if len(checked_indices) == len(clauses):
            break

    return selected_indices


def generate_contexts(model: MaxSatModel, num_context, num_constraints, num_vars, rng):
    wcnf = WCNF()
    for clauses in model:
        #        print(clauses[1])
        wcnf.append(tuple(clauses[1]))
    contexts = []
    #    n=0
    for n in range(num_context):
        if num_constraints == 0:
            num_constraints = rng.randint(1, 2 * num_vars)
        literals = []
        for i in range(1, 1 + num_vars):
            literals.append({i})
            literals.append({-i})
        #        print(literals)
        indices = get_random_clauses(wcnf, rng, literals, num_constraints)
        contexts.append([literals[j] for j in indices])
    #    print(num_vars, contexts)
    #    exit()
    return contexts


def generate_model(num_vars, clause_length, num_hard, num_soft, rng):
    return list(generate_models(1, num_vars, clause_length, num_hard, num_soft, rng))[0]


def generate_models(
    num_models, num_vars, clause_length, num_hard, num_soft, rng
) -> List[MaxSatModel]:
    clauses = _generate_all_clauses_up_to_length(num_vars, clause_length)

    if logger.isEnabledFor(logging.DEBUG):
        # Print the clauses and quit
        from pprint import pprint

        pprint(clauses)

    num_clauses = len(clauses)
    total = num_hard + num_soft
    assert total > 0

    logger.info(f"{num_clauses} clauses total - {num_hard} hard and {num_soft} soft")

    for m in range(num_models):
        logger.info(f"generating model {m + 1} of {num_models}")
        model = []
        wcnf = WCNF()
        indices = get_random_clauses(wcnf, rng, clauses, total)
        #        indices = list(sorted(rng.permutation(num_clauses)[:total]))
        hard_indices = list(sorted(rng.permutation(indices)[:num_hard]))
        soft_indices = list(sorted(set(indices) - set(hard_indices)))
        assert len(soft_indices) == num_soft

        weights = rng.randint(_MIN_WEIGHT, _MAX_WEIGHT, size=num_soft)
        for i in hard_indices:
            model.append((None, set(clauses[i])))
        for i, weight in zip(soft_indices, weights):
            model.append((weight, set(clauses[i])))
        yield model


def generate_models_and_contexts(
    path,
    num_models,
    num_context,
    num_context_constraints,
    num_vars,
    clause_length,
    num_hard,
    num_soft,
    rng,
):

    models_and_contexts = []
    for i, model in enumerate(
        generate_models(num_models, num_vars, clause_length, num_hard, num_soft, rng)
    ):

        contexts = generate_contexts(
            model, num_context, num_context_constraints, num_vars, rng
        )
        models_and_contexts.append(
            {"model": model, "contexts": contexts, "n": num_vars}
        )

    generate_wcnfs(path, models_and_contexts)
    #    print(models_and_contexts)
    return models_and_contexts


def main():
    import argparse

    fmt_class = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=fmt_class)
    parser.add_argument(
        "-o", "--output", type=str, default="model", help="Path to the output file"
    )
    parser.add_argument(
        "--num_models", type=int, default=2, help="Number of models to be generated"
    )
    parser.add_argument(
        "--num_context", type=int, default=1, help="Number of models to be generated"
    )
    parser.add_argument(
        "--num_context_constraints",
        type=int,
        default=2,
        help="Number of constraints in each context",
    )
    parser.add_argument("-n", type=int, default=3, help="number of variables")
    parser.add_argument("-k", type=int, default=3, help="length of clauses")
    parser.add_argument(
        "--num_hard", type=int, default=1, help="Number of hard constraints"
    )
    parser.add_argument(
        "--num_soft", type=int, default=1, help="Number of soft constraints"
    )
    #    parser.add_argument(
    #        "--perc-context",
    #        type=float,
    #        default=0.1,
    #        help="Number of constraints in the context",
    #    )
    parser.add_argument("-q", action="store_true", help="Print the clauses and quit")
    parser.add_argument("-s", "--seed", type=int, default=0, help="RNG seed")
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)
    for model in generate_models(
        args.num_models, args.n, args.k, args.num_hard, args.num_soft, rng
    ):
        generate_contexts(model, args.num_context, 2, args.n, rng)


if __name__ == "__main__":
    main()
