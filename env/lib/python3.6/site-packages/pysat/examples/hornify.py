#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## hornify.py
##
##  Created on: Mar 21, 2017
##      Author: Alexey S. Ignatiev
##      E-mail: aignatiev@ciencias.ulisboa.pt
##

#
#==============================================================================
from __future__ import print_function
import getopt
import os
from pysat.formula import CNF, WCNF
from six.moves import range
import sys


#
#==============================================================================
class Hornify(WCNF, object):
    """
        Dual-rail encoder creating Horn MaxSAT formulas given a CNF.
    """

    def __init__(self, cnf, nop=False):
        """
            Constructor.
        """

        # initializing WCNF's internal parameters
        super(Hornify, self).__init__()

        self.nv = 2 * cnf.nv
        self.soft = [[l] for l in range(1, self.nv + 1)]
        self.wght = [1 for cl in self.soft]
        self.topw = self.nv + 1

        if not nop:
            for v in range(1, cnf.nv + 1):
                self.hard.append([-v, -v - cnf.nv])

        for cl in cnf.clauses:
            self.hard.append(map(lambda l: l if l < 0 else -l - cnf.nv, cl))


#
#==============================================================================
def parse_options():
    """
        Parses command-line options:
    """

    try:
        opts, args = getopt.getopt(sys.argv[1:],
                                   'hp',
                                   ['help',
                                    'nop'])
    except getopt.GetoptError as err:
        sys.stderr.write(str(err).capitalize())
        usage()
        sys.exit(1)

    nop = False

    for opt, arg in opts:
        if opt in ('-h', '--help'):
            usage()
            sys.exit(0)
        elif opt in ('-p', '--nop'):
            nop = True
        else:
            assert False, 'Unhandled option: {0} {1}'.format(opt, arg)

    return nop, args


#
#==============================================================================
def usage():
    """
        Prints usage message.
    """

    print('Usage:', os.path.basename(sys.argv[0]), '[options] dimacs-file')
    print('Options:')
    print('        -h, --help')
    print('        -p, --nop     Discard the P clauses')


#
#==============================================================================
if __name__ == '__main__':
    nop, files = parse_options()
    cnf = CNF()  # input CNF formula

    if files:
        cnf.from_file(files[0])
    else:
        cnf.from_fp(sys.stdin)

    wcnf = Hornify(cnf, nop)
    wcnf.to_fp(sys.stdout)
