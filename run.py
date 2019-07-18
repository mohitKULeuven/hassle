#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 11:09:55 2019

@author: mohit
"""

import numpy as np
import argparse
import sample_generator


def main():

    fmt_class = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=fmt_class)
    parser.add_argument('-n', '--num_var', type=int, default=4,
                        help='number of binary variables')
    parser.add_argument('--k', type=int, default=2,
                        help='size of cnfs')
    parser.add_argument('--num_hard', type=int, default=0,
                        help='Number of hard constraints')
    parser.add_argument('--num_soft', type=int, default=2,
                        help='Number of soft constraints')
    parser.add_argument('-s','--seed', type=int, default=0,
                        help='RNG seed')
    parser.add_argument('--file_name', type=str, default='tmp',
                        help='Name of the file where models will be saved')
    parser.add_argument('--num_models', type=int, default=2,
                        help='Number of models to be generated')
    parser.add_argument('--num_samples', type=int, default=1,
                        help='Number of models to be generated')
    args = parser.parse_args()

    np.seterr(all='raise')
#    np.random.seed(args.seed)
#
#    rng = np.random.RandomState(args.seed)

    sample_generator.generate_models(
            args.num_var,args.k,args.num_hard,args.num_soft,args.seed,args.file_name,args.num_models)
    sample_generator.generate_samples(args.file_name,args.num_samples)


if __name__ == '__main__':
    main()