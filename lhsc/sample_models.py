# -*- coding: utf-8 -*-

import pickle
import numpy as np
import itertools as it

from pysat.examples.fm import FM
from pysat.formula import WCNF
from scipy.special import binom


# XXX some solvers support integer weights only, let's do that
_MIN_WEIGHT, _MAX_WEIGHT = 1, 100


def is_entailed(wcnf,clause):
    wcnf_new=wcnf.copy()
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
    
    #check entailment property of the added constraints
    
    # ... except for impossible clauses like 'x and not x', let's delete them
    def possible(clause):
        for i in range(len(clause)):
            for j in range(i + 1, len(clause)):
                if clause[i] == -clause[j]:
                    return False
        return True

    return list(sorted(filter(possible, clauses)))


def generate_wcnfs(path,models_and_contexts):
    
    for i,elem in enumerate(models_and_contexts):
        model=elem['model']
        contexts=elem['contexts']
        
        wcnf = WCNF()
        for weight_clause in model:
            if weight_clause[0]==None:
                wcnf.append(tuple(weight_clause[1]))
            else:
                wcnf.append(tuple(weight_clause[1]), weight=weight_clause[0])
        wcnf.to_file(path + f"_{i}.wcnf")
        
        for j,context in enumerate(contexts):
            wcnf_context=wcnf.copy()
            for literals in context:
                wcnf_context.append((literals,))
            wcnf_context.to_file(path + f"_{i}_context_{j}.wcnf")
            
        
def get_random_clauses(rng,clauses,n):
    wcnf = WCNF()
    selected_indices=[]
    while n>0:
        indices=[ind for ind in range(len(clauses)) if ind not in selected_indices]
        i=rng.choice(indices)
        if not is_entailed(wcnf,clauses[i]):
            wcnf.append(clauses[i])
            selected_indices.append(i)
            n=n-1
    return selected_indices
    


def generate_models(
    path,
    num_models,
    num_context,
    num_vars,
    clause_length,
    num_hard,
    num_soft,
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
    
    total = num_hard + num_soft
    assert total > 0

    print(f"{num_clauses} clauses total - {num_hard} hard and {num_soft} soft")

    models_and_contexts = []
    for m in range(num_models):
        print(f"generating model {m + 1} of {num_models}")
        MaxSatModel=[]
        indices=get_random_clauses(rng,clauses,total)
#        indices = list(sorted(rng.permutation(num_clauses)[:total]))
        hard_indices = list(sorted(rng.permutation(indices)[:num_hard]))
        soft_indices = list(sorted(set(indices) - set(hard_indices)))
        assert len(soft_indices) == num_soft

        weights = rng.randint(_MIN_WEIGHT, _MAX_WEIGHT, size=num_soft)
        for i in hard_indices:
            MaxSatModel.append((None,set(clauses[i])))
        for i, weight in zip(soft_indices, weights):
            MaxSatModel.append((weight,set(clauses[i])))

        contexts=[]
        for n in range(num_context):
            indices = list(sorted(rng.permutation(num_clauses)))
            not_hard_indices = list(sorted(set(indices) - set(hard_indices)))
            context = set(clauses[rng.permutation(not_hard_indices)[0]])
            contexts.append(context)
        models_and_contexts.append({'model':MaxSatModel,'contexts':contexts})
        
    generate_wcnfs(path,models_and_contexts)
    return models_and_contexts

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
        "--num-context", type=int, default=2, help="Number of models to be generated"
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
    return generate_models(
        args.output,
        args.num_models,
        args.num_context,
        args.n,
        args.k,
        args.num_hard,
        args.num_soft,
        args.q,
        rng,
    )


if __name__ == "__main__":
    main()
