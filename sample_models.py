#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
import numpy as np
import itertools as it
from scipy.special import binom


_EPSILON = 0.001


def _generate_all_clauses_up_to_length(num_vars, length, pcaq):
    flip_or_dont = lambda v: -(v - num_vars) if v > num_vars else v

    lits = range(1, 2 * num_vars + 1)
    clauses = set([tuple(set(map(flip_or_dont, clause))) for clause
                   in it.combinations_with_replacement(lits, length)])

    # This makes sure that all symmetries are accounted for...
    must_be = sum(binom(2 * num_vars, l) for l in range(1, length + 1))
    assert len(clauses) == must_be

    # ... except for impossible clauses like 'x and not x', let's delete them
    def possible(clause):
        for i in range(len(clause)):
            for j in range(i + 1, len(clause)):
                if clause[i] == -clause[j]:
                    return False
        return True

    clauses = list(sorted(filter(possible, clauses)))

    if pcaq:
        # Print the clauses and quit
        from pprint import pprint
        pprint(clauses)
        quit()

    return clauses


def _to_clause_line(clause, weight=None):
    line_weight = '' if weight is None else f'{weight:5.3f} '
    return line_weight + ' '.join(list(map(str, clause + (0,))))


def _write_wcnf(path, clauses, num_vars, hard_indices, soft_indices, c, w):
    his, sis = set(hard_indices), set(soft_indices)
    num_active_clauses = len(his | sis)

    with open(path, 'wt') as fp:
        fp.write(f'p wcnf {num_vars} {num_active_clauses}\n')

        # XXX hard and soft indices might overlap;  hard wins
        # XXX some solvers only support integer weights

        for i in hard_indices:
            fp.write(_to_clause_line(clauses[i]) + '\n')

        for i in sorted(sis - his):
            fp.write(_to_clause_line(clauses[i], weight=w[i]) + '\n')


def generate_models(path,
                    num_models,
                    num_vars,
                    clause_length,
                    perc_hard,
                    perc_soft,
                    pcaq,
                    rng):

    clauses = _generate_all_clauses_up_to_length(num_vars, clause_length, pcaq)

    num_clauses = len(clauses)
    num_hard = int(np.ceil(num_clauses * perc_hard))
    num_soft = int(np.ceil(num_clauses * perc_soft))
    assert num_hard + num_soft > 0

    print(f'{num_clauses} clauses total - {num_hard} hard and {num_soft} soft')

    models = []
    for m in range(num_models):
        print(f'generating model {m + 1} of {num_models}')

        hard_indices = list(sorted(rng.permutation(num_clauses)[:num_hard]))
        c = np.zeros(num_clauses, dtype=np.int)
        c[hard_indices] = 1

        soft_indices = list(sorted(rng.permutation(num_clauses)[:num_soft]))
        w = np.zeros(num_clauses, dtype=np.float32)
        temp = rng.uniform(_EPSILON, 1, size=num_soft)
        w[soft_indices] = temp / np.linalg.norm(temp)

        _write_wcnf(path + f'_{m}.wcnf',
                    clauses,
                    num_vars,
                    hard_indices,
                    soft_indices,
                    c, w)


def main():
    import argparse

    fmt_class = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=fmt_class)
    parser.add_argument('-o', '--output', type=str, default='model',
                        help='Path to the output file')
    parser.add_argument('--num-models', type=int, default=2,
                        help='Number of models to be generated')
    parser.add_argument('-n', type=int, default=3,
                        help='number of variables')
    parser.add_argument('-k', type=int, default=3,
                        help='length of clauses')
    parser.add_argument('--perc-hard', type=float, default=0.0,
                        help='Number of hard constraints')
    parser.add_argument('--perc-soft', type=float, default=0.1,
                        help='Number of soft constraints')
    parser.add_argument('-q', action='store_true',
                        help='Print the clauses and quit')
    parser.add_argument('-s', '--seed', type=int, default=0,
                        help='RNG seed')
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)
    generate_models(args.output,
                    args.num_models,
                    args.n,
                    args.k,
                    args.perc_hard,
                    args.perc_soft,
                    args.q,
                    rng)


if __name__ == '__main__':
    main()
