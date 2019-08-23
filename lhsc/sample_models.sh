#!/usr/bin/env bash

NUM_MODELS=100

for n in 5 10 15; do
    for k in 3 5; do
        for nh in 0 3; do
            for ns in 1 5; do
                output="random_n=${n}_k=${k}_ph=${nh}_ps=${ns}"
                python sample_models.py \
                    -o $output \
                    --num-models $NUM_MODELS \
                    -n $n -k $k \
                    --num_hard $nh \
                    --num_soft $ns \
                    -s 0
            done
        done
    done
done

