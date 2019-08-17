#!/usr/bin/env bash

NUM_MODELS=100

for n in 5 10 15; do
    for k in 3 5; do
        for ph in 0 0.1 0.25; do
            for ps in 0.1 0.25; do
                output="random_n=${n}_k=${k}_ph=${ph}_ps=${ps}"
                python sample_models.py \
                    -o $output \
                    --num-models $NUM_MODELS \
                    -n $n -k $k \
                    --perc-hard $ph \
                    --perc-soft $ps \
                    -s 0
            done
        done
    done
done

