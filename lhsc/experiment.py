from typing import Optional, List

import numpy as np
import logging
import pickle
from tabulate import tabulate
import csv
import time
import os

from .sample_models import generate_model, generate_contexts
from .solve import solve_weighted_max_sat, get_value
from .learn import label_instance
from .type_def import Instance, MaxSatModel, Context
from .learn import learn_weighted_max_sat


logger = logging.getLogger(__name__)

def represent_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def cnf_to_model(cnf_file) -> Optional[MaxSatModel]:
    model=[]
    with open(cnf_file) as fp:
        line = fp.readline()
        p_ind=0
        while line:
            line_as_list=line.strip().split()
            if len(line_as_list)>0 and p_ind==1 and represent_int(line_as_list[0]):
                model.append((None,set(map(int,line_as_list))))
            if p_ind==0 and line_as_list[0]=='p':
                p_ind=1
            line = fp.readline()
    if model:
        return model
    return None

#def model_to_cnf(model,param,path):
#    cnf_file = open(path+'param'+'.cnf', 'w')
#    filewriter = csv.writer(csvfile, delimiter=' ')
#    filewriter.writerow(['model_seed','num_hard', 'num_soft', 'n',
#                         'clause_length','context_count','num_literal',
#                         'time','num_example','training_error'])
#    csvfile.close()


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

    data = np.array(data).astype(np.int)
    labels = np.array(labels)
    return data, labels, contexts_to_learn


def random_contexts(
    n, num_contexts, min_num_literals, max_num_literals, rng, max_tries=10
):
    result = set()
    for _ in range(max_tries * num_contexts):
        choices = list(range(n))
        num_literals = rng.randint(min_num_literals, max_num_literals + 1)
        selected = rng.choice(np.array(choices), size=num_literals, replace=False) + 1
        selected *= np.sign(rng.rand(len(selected)) - 0.5).astype(np.int)
        result.add(frozenset(list(selected)))
        if len(result) >= num_contexts:
            break
    return [set(c) for c in result]


def learn_from_random_model(n,clause_length,num_hard,num_soft,
                            contexts_count,num_literal,model_seed):
    logging.basicConfig(level=logging.WARNING)
    
#    model_seeds = [444]
    context_seed_start = [101010]
    
    rng = np.random.RandomState(model_seed)
    m = num_hard + num_soft
    
    param=f"_model_seed_{model_seed}_num_hard_{num_hard}_num_soft_{num_soft}_n_{n}__clause_length_{clause_length}_context_count_{context_count}_num_literal_{num_literal}"
    row=[model_seed,num_hard,num_soft,n,clause_length,context_count,num_literal]
    pickle_var={}
#   pickle_vars=pickle.load( open( 'pickles/'+param+'.pickle', "rb" ) )
#   print(pickle_vars['data'].shape[0])
#   continue
    
    true_model = generate_model(n, clause_length, num_hard, num_soft, rng)
    true_model = [(w / 100 if w else None, clause) for w, clause in true_model]
    print(f"True Model\n{model_to_string(true_model)}")
    pickle_var["true_model"]=true_model
    rng = np.random.RandomState(context_seed_start)
    contexts = random_contexts(n, contexts_count, 1, num_literal, rng)

    for context in contexts:
#        print(f"Context {context_to_string(context)}")
        solution = solve_weighted_max_sat(n, true_model, context)
#        print(f"Solution {solution}")

        if solution is None:
            print(f"INFEASIBLE CONTEXT {context_to_string(context)}")

    data, labels, learning_contexts = get_data_set(
        n, true_model, contexts, rng, allow_negative=True
    )
    pickle_var["data"]=data
    pickle_var["labels"]=labels
    pickle_var["learning_contexts"]=learning_contexts

    if logger.isEnabledFor(logging.INFO):
        overview = [
            [
                f"{context_to_string(learning_contexts[k])}",
                f"{data[k, :]}",
                f"{labels[k]}",
            ]
            for k in range(len(labels))
        ]
        print(tabulate(overview))
#        print(data, labels, learning_contexts)
    
    start=time.time()
    learned_model = learn_weighted_max_sat(m, data, labels, learning_contexts)
    end=time.time()
    print(f"Learned model\n{model_to_string(learned_model)}")
    pickle_var["learned_model"]=learned_model
    
    
    training_error=0
    if learned_model:
        for k in range(data.shape[0]):
            instance = data[k, :]
            label = labels[k]
            learned_label = label_instance(
                learned_model, instance, learning_contexts[k]
            )
            if label != learned_label:
                training_error+=1
#                if label:
#                    
#                    incorrect_best_solution=solve_weighted_max_sat(n, learned_model, learning_contexts[k]).astype(np.int)
#                    logger.warning(
#                        f"{instance} labeled {learned_label} instead of {label} because of {incorrect_best_solution}"
#                    )
                logger.warning(
                    f"{instance} labeled {learned_label} instead of {label} (learned / true)"
                )
#        regret=eval_regret(n, true_model, learned_model)
#        print("REGRET ", regret)
        pickle.dump( pickle_var, open( 'pickles/'+param+'.pickle', "wb" ) )
        csvfile = open('results/results_big'+'.csv', 'a')
        filewriter = csv.writer(csvfile, delimiter=',')
        row.extend([end-start,data.shape[0],training_error])
        filewriter.writerow(row)
        csvfile.close()


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
#    cnf_dir="cnfs/"
#    for cnf_file in os.listdir(cnf_dir):
#        print(cnf_to_model(cnf_dir+cnf_file))
#    exit()
    model_seeds = [111, 222, 333, 444, 555]
    num_hard_lst=[5,10,15,20]
    num_soft_lst=[5,10,15,20]
    num_vars=[5,10,15]
    context_counts=[5,10,15,20]
    csvfile = open('results/results_big'+'.csv', 'w')
    filewriter = csv.writer(csvfile, delimiter=',')
    filewriter.writerow(['model_seed','num_hard', 'num_soft', 'n',
                         'clause_length','context_count','num_literal',
                         'time','num_example','training_error'])
    csvfile.close()
    for n in num_vars:
        for clause_length in range(3,max(3, int(2 * n / 5))):
            for num_hard in num_hard_lst:
                for num_soft in num_soft_lst:
                    for context_count in context_counts:
                        for num_literal in range(1,max(2, int(2 * n / 5))):
                            for model_seed in model_seeds:
                                try:
                                    learn_from_random_model(n,clause_length,
                                                            num_hard,num_soft,
                                                            context_count,
                                                            num_literal,
                                                            model_seed)
                                except AssertionError as error:
                                    continue
    # learn_hard_constraints()
