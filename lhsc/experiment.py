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
    model = []
    with open(cnf_file) as fp:
        line = fp.readline()
        p_ind = 0
        while line:
            line_as_list = line.strip().split()
            if len(line_as_list) > 0 and p_ind == 1 and represent_int(line_as_list[0]):
                model.append((None, set(map(int, line_as_list))))
            if p_ind == 0 and line_as_list[0] == "p":
                p_ind = 1
            line = fp.readline()
    if model:
        return model
    return None


# def model_to_cnf(model,param,path):
#    param_str=f"_model_seed_{param['model_seed']}_num_hard_{param['num_hard']}_num_soft_{param['num_soft']}_n_{param['n']}__clause_length_{param['clause_length']}_context_count_{param['context_count']}_num_literal_{param['num_literal']}"
#    cnf_file = open(path+param_str+'.cnf', 'w')
#    filewriter = csv.writer(cnf_file, delimiter=' ')
#    filewriter.writerow(['p','cnf',param['n'],param['num_hard']+param['num_soft']])
#    for row in model:
#        filewriter.writerow(list(row[1]))

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
        return solve_weighted_max_sat(n, model, context,1)
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
    output=[]
    if positive:
        for instance in  solve_weighted_max_sat(n, model, context,num_instances):
            if not instance.tolist() in [a.tolist() for a in prev_instances]:
                output.append(instance)
            if len(output) >= num_instances-len(prev_instances):
                break
        output = np.array(output).astype(np.int)
        return output
    else:
        max_tries = 100*(num_instances-len(prev_instances))
        for l in range(max_tries):
            instance = rng.rand(n) > 0.5
            for i in context:
                instance[abs(i) - 1] = i > 0
            if not label_instance(model, instance, context):
                if not instance.tolist() in [a.tolist() for a in prev_instances]:
                    output.append(instance)
                if len(output) >= num_instances-len(prev_instances):
                    break
        return output

def eval_hard_constraints(n: int, true_model: MaxSatModel, learned_model: MaxSatModel):
    pass


def eval_regret(n: int, true_model: MaxSatModel, learned_model: MaxSatModel):
    context = set()
    true_best_solution = solve_weighted_max_sat(n, true_model, context,1)
    true_best_value = get_value(true_model, true_best_solution)

    learned_best_solution = solve_weighted_max_sat(n, learned_model, context,1)
    learned_best_value = get_value(true_model, learned_best_solution)

    return true_best_value - learned_best_value if learned_best_value else -1

def eval_instance_regret(n: int, model: MaxSatModel, instance,context):
#    context = set()
    true_best_solution = solve_weighted_max_sat(n, model, context,1)
    true_best_value = get_value(model, true_best_solution)
    learned_best_value = get_value(model, instance)
#    print(true_best_value, learned_best_value)
    return learned_best_value/true_best_value if true_best_value else 1


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
    n, num_contexts,prev_contexts, min_num_literals, max_num_literals, rng, max_tries=10
):
#    print(frozenset(c) for c in prev_contexts)
    result = set(frozenset(c) for c in prev_contexts)
    output=set()
#    print("before:",result)
    for _ in range(max_tries * num_contexts):
        choices = list(range(n))
        num_literals = rng.randint(min_num_literals, max_num_literals + 1)
        selected = rng.choice(np.array(choices), size=num_literals, replace=False) + 1
        selected *= np.sign(rng.rand(len(selected)) - 0.5).astype(np.int)
        before=len(result)
        result.add(frozenset(set(selected)))
        if len(result)>before:
            output.add(frozenset(set(selected)))
        if len(result) >= num_contexts:
            break
#    print("after:",result)
    return [set(c) for c in output]


def eval_precision_and_regret(target_model,learned_model,n,sample_size,
                              context_ind,num_literal=None,rng=None):
    correct_instances=[]
    correct=0
    num_sam=0
    
    if context_ind==0:
        context_less_instances = solve_weighted_max_sat(n, learned_model, set(),sample_size)
        num_sam=len(context_less_instances)
        if context_less_instances:
            for instance in context_less_instances:
                if get_value(target_model,instance) is not None:
                    correct_instances.append({'instance':instance,'context':set()})
                    correct+=1
    else:
        contexts = random_contexts(n, sample_size,[], 1, num_literal, rng)
        for context in contexts:
            instance=get_instance(n, learned_model, context, True, rng)
            if instance is not None:
                num_sam+=1
                if get_value(target_model,instance) is not None:
                    correct_instances.append({'instance':instance,'context':context})
                    correct+=1
    
    precision=correct/num_sam
    regret=0
    for row in correct_instances:
        regret +=eval_instance_regret(n, target_model, row['instance'],row['context'])
    regret=regret/len(correct_instances) if len(correct_instances)>0 else ""
    
    return precision,regret,num_sam


def eval_recall(target_model,learned_model,n,sample_size,num_literal,prev_contexts):
    contexts = random_contexts(n, sample_size+len(prev_contexts),prev_contexts, 1, num_literal, rng)
    num_sam=0
    correct=0
#    print(contexts)
    for context in contexts:
        instance=get_instance(n, target_model, context, True, rng)
#        print(instance)
        if instance is not None:
            num_sam+=1
            if get_value(learned_model,instance) is not None:
                correct+=1
    return correct/num_sam

def eval_increasing_examples(param,m,n,sample_size,rng):
    print(param)
    num_literal=int(n/2)
    pickle_var=pickle.load( open( 'pickles/'+param+'.pickle', "rb" ) )
    target_model=pickle_var["true_model"]
    learned_model=pickle_var["learned_model"]
    prev_contexts=pickle_var["contexts"]
    
    data=pickle_var["data"]
    labels=pickle_var["labels"]
    learning_contexts=pickle_var["learning_contexts"]
    
    print("True model")
    print(model_to_string(target_model))
    
    for num_instances in [1,20,50]:
        if num_instances>1:
            for i,context in enumerate(learning_contexts):
                prev_instances=[data[j] for j, x in enumerate(learning_contexts) if x == context]
                for positive in [True,False]:
                    instances=add_instances(n, target_model, context, positive, 
                                                  num_instances, prev_instances, rng)
                    for instance in instances:
                        data=np.append(data,[instance],axis=0)
                        labels=np.append(labels,[positive],axis=0)
                        learning_contexts.append(context)
        for k in range(len(data)):
            learned_label = label_instance(target_model, data[k], learning_contexts[k])
            if learned_label!=labels[k]:
                print(f"data:{data[k]},labels:{labels[k]},learning_contexts:{learning_contexts[k]}")
        start = time.time()
        learned_model = learn_weighted_max_sat(m, data, labels, learning_contexts)
        end = time.time()
        
        pickle_var["data"]=data
        pickle_var["labels"]=labels
        pickle_var["learning_contexts"]=learning_contexts
        pickle_var["learned_model"]=learned_model
        pickle.dump(pickle_var, open("pickles/" + param+f"_num_instances_{num_instances}" + ".pickle", "wb"))

        
        print("Learned model")
        print(model_to_string(learned_model))
        recall=eval_recall(target_model,learned_model,n,sample_size,num_literal,prev_contexts)
        precision,regret,num_sam=eval_precision_and_regret(target_model,learned_model,n,
                                                   sample_size,0,
                                                   num_literal=num_literal)
        precision_c,regret_c,num_sam_c=eval_precision_and_regret(target_model,learned_model,n,
                                                   sample_size,1,
                                                   num_literal=num_literal,rng=rng)
        csvfile = open("results/increasing_examples" + ".csv", "a")
        filewriter = csv.writer(csvfile, delimiter=",")
        row = [
            model_seed,num_hard,num_soft,n,clause_length,context_count,num_literal,
            len(data),recall,precision,regret,num_sam,precision_c,regret_c,num_sam_c,end-start
        ]
        filewriter.writerow(row)
        csvfile.close()

def evaluation(param,n,sample_size,rng):
    print(param)
    num_literal=int(n/2)
    pickle_var=pickle.load( open( 'pickles/'+param+'.pickle', "rb" ) )
    target_model=pickle_var["true_model"]
    learned_model=pickle_var["learned_model"]
    prev_contexts=pickle_var["contexts"]
#    for num_literal in 4:
#    print(model_to_string(target_model))
#    print(prev_contexts)
    recall=eval_recall(target_model,learned_model,n,sample_size,num_literal,prev_contexts)
    precision,regret,num_sam=eval_precision_and_regret(target_model,learned_model,n,
                                               sample_size,0,
                                               num_literal=num_literal)
    precision_c,regret_c,num_sam_c=eval_precision_and_regret(target_model,learned_model,n,
                                               sample_size,1,
                                               num_literal=num_literal,rng=rng)
    
    csvfile = open("results/evaluation_stats_2" + ".csv", "a")
    filewriter = csv.writer(csvfile, delimiter=",")
    row = [
        model_seed,num_hard,num_soft,n,clause_length,context_count,num_literal,
        recall,precision,regret,num_sam,precision_c,regret_c,num_sam_c
    ]
    filewriter.writerow(row)
    csvfile.close()
 
if __name__ == "__main__":
    model_seeds = [111, 222, 333, 444, 555]
    num_hard_lst = [1, 5, 10, 20]
    num_soft_lst = [1, 5, 10, 20]
    num_vars = [5, 10, 20]
    context_counts = [5, 10, 20]
    csvfile = open("results/evaluation_stats_2" + ".csv", "w")
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
            "recall",
            "precision",
            "regret",
            "num_sam",
            "precision_c",
            "regret_c",
            "num_sam_c"
        ]
    )
    csvfile.close()
    for model_seed in model_seeds:
        rng = np.random.RandomState(model_seed)
        for n in num_vars:
            num_sam=100*(5**int(n/10))
            for num_literal in range(2, max(4, int(2 * n / 5)),2):
                for clause_length in range(2, max(3, int(2 * n / 5)),2):
                    for num_hard in num_hard_lst:
                        for num_soft in num_soft_lst:
                            for context_count in context_counts:
                                param = f"_model_seed_{model_seed}_num_hard_{num_hard}_num_soft_{num_soft}_n_{n}__clause_length_{clause_length}_context_count_{context_count}_num_literal_{num_literal}"
#                                pickle_var=pickle.load( open( 'pickles/'+param+'.pickle', "rb" ) )
#                                print(model_to_string(pickle_var["true_model"]))
#                                get_data_set(n, pickle_var["true_model"], [set()], rng)
                                try:
                                    evaluation(param,n,num_sam,rng)
                                except FileNotFoundError as error:
                                    continue                   

#if __name__ == "__main__":
##    exit()
#    model_seeds = [111,222,333]
#    num_hard_lst = [1,5,10,20]
#    num_soft_lst = [1,5,10,20]
#    num_vars = [5,10]
#    context_counts = [5,10]
#    csvfile = open("results/increasing_examples" + ".csv", "w")
#    filewriter = csv.writer(csvfile, delimiter=",")
#    filewriter.writerow(
#        [
#            "model_seed",
#            "num_hard",
#            "num_soft",
#            "n",
#            "clause_length",
#            "context_count",
#            "num_literal",
#            "num_instances",
#            "recall",
#            "precision",
#            "regret",
#            "num_sam",
#            "precision_c",
#            "regret_c",
#            "num_sam_c",
#            "time_taken"
#        ]
#    )
#    csvfile.close()
#    for model_seed in model_seeds:
#        rng = np.random.RandomState(model_seed)
#        for n in num_vars:
#            num_sam=100*(5**int(n/10))
#            for num_literal in range(2, max(4, int(2 * n / 5)),2):
#                for clause_length in range(2, max(3, int(2 * n / 5)),2):
#                    for num_hard in num_hard_lst:
#                        for num_soft in num_soft_lst:
#                            for context_count in context_counts:
#                                param = f"_model_seed_{model_seed}_num_hard_{num_hard}_num_soft_{num_soft}_n_{n}__clause_length_{clause_length}_context_count_{context_count}_num_literal_{num_literal}"
##                                pickle_var=pickle.load( open( 'pickles/'+param+'.pickle', "rb" ) )
##                                print(model_to_string(pickle_var["true_model"]))
##                                get_data_set(n, pickle_var["true_model"], [set()], rng)
#                                try:
##                                    evaluation(param,n,num_sam,rng)
#                                    eval_increasing_examples(param,num_hard+num_soft,n,num_sam,rng)
#                                except FileNotFoundError as error:
#                                    continue
##                                exit()



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


def learn_from_random_model(
    n, clause_length, num_hard, num_soft, context_counts, num_literal, 
    model_seed,rng
):
    logging.basicConfig(level=logging.WARNING)
    
    
    m = num_hard + num_soft
    pickle_var = {}
    #   pickle_vars=pickle.load( open( 'pickles/'+param+'.pickle', "rb" ) )
    #   print(pickle_vars['data'].shape[0])
    #   continue

    true_start = time.time()
    true_model = generate_model(n, clause_length, num_hard, num_soft, rng)
    true_end = time.time()
    true_model = [(w / 100 if w else None, clause) for w, clause in true_model]
    print(f"True Model\n{model_to_string(true_model)}")
    pickle_var["true_model"] = true_model
    
    prev_context=[]
    data=[]
    labels=[]
    learning_contexts=[]
    for context_count in context_counts:
        num_infeasible_context=0
        if len(prev_context)<context_count:
            contexts = random_contexts(n, context_count,prev_context, 1, num_literal, rng)
            tmp_data, tmp_labels, tmp_learning_contexts = get_data_set(
                n, true_model, contexts, rng, allow_negative=True
            )
            if len(data)>0:
                data=np.append(data,tmp_data,axis=0)
                labels=np.append(labels,tmp_labels,axis=0)
                learning_contexts.extend(tmp_learning_contexts)
            else:
                data=tmp_data
                labels=tmp_labels
                learning_contexts=tmp_learning_contexts
            prev_context.extend(contexts)
            contexts=prev_context
        else:
            contexts = random_contexts(n, context_count, 1, num_literal, rng)
            prev_context=contexts
        for context in contexts:
            #        print(f"Context {context_to_string(context)}")
            solution = solve_weighted_max_sat(n, true_model, context,1)
            #        print(f"Solution {solution}")
        
            if solution is None:
                num_infeasible_context+=1
                print(f"INFEASIBLE CONTEXT {context_to_string(context)}")

        pickle_var["contexts"] = contexts
        pickle_var["data"] = data
        pickle_var["labels"] = labels
        pickle_var["learning_contexts"] = learning_contexts
        
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
        
        start = time.time()
        learned_model = learn_weighted_max_sat(m, data, labels, learning_contexts)
        end = time.time()
        print(f"Learned model\n{model_to_string(learned_model)}")
        pickle_var["learned_model"] = learned_model
        
        training_error = 0
        if learned_model:
            for k in range(data.shape[0]):
                instance = data[k, :]
                label = labels[k]
                learned_label = label_instance(
                    learned_model, instance, learning_contexts[k]
                )
                if label != learned_label:
                    training_error += 1
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
            param = f"_model_seed_{model_seed}_num_hard_{num_hard}_num_soft_{num_soft}_n_{n}__clause_length_{clause_length}_context_count_{context_count}_num_literal_{num_literal}"
            pickle.dump(pickle_var, open("pickles/" + param + ".pickle", "wb"))
            csvfile = open("results/results" + ".csv", "a")
            filewriter = csv.writer(csvfile, delimiter=",")
            row = [
                model_seed,
                num_hard,
                num_soft,
                n,
                clause_length,
                context_count,
                num_literal,
                true_end - true_start,
                end - start,
                data.shape[0], 
                num_infeasible_context,
                training_error
            ]
            filewriter.writerow(row)
            csvfile.close()
            print(param)

#if __name__ == "__main__":
#    learn_hard_constraints()
#    exit()
#    
#    # logging.basicConfig(level=logging.INFO)
#    #    cnf_dir="cnfs/"
#    #    for cnf_file in os.listdir(cnf_dir):
#    #        print(cnf_to_model(cnf_dir+cnf_file))
#    #    exit()
#    model_seeds = [111, 222, 333, 444, 555]
#    num_hard_lst = [1, 5, 10, 20]
#    num_soft_lst = [1, 5, 10, 20]
#    num_vars = [5, 10, 20]
#    context_counts = [5, 10, 20]
#    
#    csvfile = open("results/results" + ".csv", "w")
#    filewriter = csv.writer(csvfile, delimiter=",")
#    filewriter.writerow(
#        [
#            "model_seed",
#            "num_hard",
#            "num_soft",
#            "n",
#            "clause_length",
#            "context_count",
#            "num_literal",
#            "true_model_time",
#            "time",
#            "num_example",
#            "num_infeasible_context",
#            "training_error",
#        ]
#    )
#    csvfile.close()
#    for model_seed in model_seeds:
#        rng = np.random.RandomState(model_seed)
#        for n in num_vars:
#            for num_literal in range(2, max(4, int(2 * n / 5)),2):
#                for clause_length in range(2, max(3, int(2 * n / 5)),2):
#                    for num_hard in num_hard_lst:
#                        for num_soft in num_soft_lst:
##                            for context_count in context_counts:
#                            try:
#                                learn_from_random_model(
#                                    n,
#                                    clause_length,
#                                    num_hard,
#                                    num_soft,
#                                    context_counts,
#                                    num_literal,
#                                    model_seed,rng
#                                )
#                            except AssertionError as error:
#                                continue
#                        exit()
