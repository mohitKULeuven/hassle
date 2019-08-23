# -*- coding: utf-8 -*-

import pickle
import numpy as np
import itertools as it

from pysat.formula import WCNF
from scipy.special import binom


# XXX some solvers support integer weights only, let's do that
_MIN_WEIGHT, _MAX_WEIGHT = 1, 100


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
    
    #check entailment property of the added constraints
    
    # ... except for impossible clauses like 'x and not x', let's delete them
    def possible(clause):
        for i in range(len(clause)):
            for j in range(i + 1, len(clause)):
                if clause[i] == -clause[j]:
                    return False
        return True

    return list(sorted(filter(possible, clauses)))


def add_context(rng, num_clauses, hard_indices, soft_indices, perc_context):

    num_context = int(np.ceil(num_clauses * perc_context))
    indices = list(sorted(rng.permutation(num_clauses)))
    not_hard_indices = list(sorted(set(indices) - set(hard_indices)))
    context_indices = list(sorted(rng.permutation(not_hard_indices)[:num_context]))
    return (
        set(hard_indices).union(set(context_indices)),
        set(soft_indices).difference(set(context_indices)),
    )


def generate_models(
    path,
    num_models,
    num_context,
    num_vars,
    clause_length,
    perc_hard,
    perc_soft,
    perc_context,
    pcaq,
    rng,
):

    clauses = _generate_all_clauses_up_to_length(num_vars, clause_length)

    if pcaq:
        # Print the clauses and quit
        from pprint import pprint

        pprint(clauses)
        quit()

    num_clauses = len(clauses)
    num_hard = int(np.ceil(num_clauses * perc_hard))
    num_soft = int(np.ceil(num_clauses * perc_soft))
    total = num_hard + num_soft
    assert total > 0

    print(f"{num_clauses} clauses total - {num_hard} hard and {num_soft} soft")

    #    models = []
    for m in range(num_models):
        print(f"generating model {m + 1} of {num_models}")

        indices = list(sorted(rng.permutation(num_clauses)[:total]))
        hard_indices = list(sorted(rng.permutation(indices)[:num_hard]))
        soft_indices = list(sorted(set(indices) - set(hard_indices)))
        assert len(soft_indices) == num_soft

        weights = rng.randint(_MIN_WEIGHT, _MAX_WEIGHT, size=num_soft)
        wcnf = WCNF()
        for i in hard_indices:
            wcnf.append(clauses[i])
        for i, weight in zip(soft_indices, weights):
            wcnf.append(clauses[i], weight=weight)
        wcnf.to_file(path + f"_{m}.wcnf")
        #        print(soft_indices,weights)
        for n in range(num_context):
            hard_indices, soft_indices_context = add_context(
                rng, num_clauses, hard_indices, soft_indices, perc_context
            )
            num_soft = len(soft_indices)

            context_weights = rng.randint(_MIN_WEIGHT, _MAX_WEIGHT, size=num_soft)

            for i, ind in enumerate(soft_indices_context):
                if ind in soft_indices:
                    context_weights[i] = weights[soft_indices.index(ind)]

            #            print(soft_indices,weights)
            wcnf = WCNF()
            for i in hard_indices:
                wcnf.append(clauses[i])
            for i, weight in zip(soft_indices_context, context_weights):
                wcnf.append(clauses[i], weight=weight)
            wcnf.to_file(path + f"_{m}_context_{n}.wcnf")


def main():
    import argparse

    fmt_class = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=fmt_class)
    parser.add_argument(
        "-o", "--output", type=str, default="model", help="Path to the output file"
    )
    parser.add_argument(
        "--num-models", type=int, default=2, help="Number of models to be generated"
    )
    parser.add_argument(
        "--num-context", type=int, default=0, help="Number of models to be generated"
    )
    parser.add_argument("-n", type=int, default=3, help="number of variables")
    parser.add_argument("-k", type=int, default=3, help="length of clauses")
    parser.add_argument(
        "--perc-hard", type=float, default=0.0, help="Number of hard constraints"
    )
    parser.add_argument(
        "--perc-soft", type=float, default=0.1, help="Number of soft constraints"
    )
    parser.add_argument(
        "--perc-context",
        type=float,
        default=0.1,
        help="Number of constraints in the context",
    )
    parser.add_argument("-q", action="store_true", help="Print the clauses and quit")
    parser.add_argument("-s", "--seed", type=int, default=0, help="RNG seed")
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)
    generate_models(
        args.output,
        args.num_models,
        args.num_context,
        args.n,
        args.k,
        args.perc_hard,
        args.perc_soft,
        args.perc_context,
        args.q,
        rng,
    )


if __name__ == "__main__":
    main()
