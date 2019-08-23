#!/usr/bin/env python3
#-*- coding:utf-8 -*-
##
## sudoku.py
##
##  Created on: Jul 10, 2018
##      Author: Alexey S. Ignatiev
##      E-mail: aignatiev@ciencias.ulisboa.pt
##

#
#==============================================================================
from __future__ import print_function
import getopt
import itertools
import os
from pysat.formula import CNF, IDPool
from pysat.solvers import Solver
import random
import sys

# GTK3 GUI
#==============================================================================
import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk

# Sudoku cell styles
#==============================================================================
css_data = """
    .generate-button {
        font-size: 150%;
    }

    .normal-button {
        font-size: 200%;
        font-weight: 600;
    }

    .bordertop-cell {
        border-top: 1px solid black;
    }

    .borderbottom-cell {
        border-bottom: 1px solid black;
    }

    .borderleft-cell {
        border-left: 1px solid black;
    }

    .borderright-cell {
        border-right: 1px solid black;
    }

    .red-background {
        background-image: none;
        background-color: #ffcccb;
    }

    .green-background {
        background-image: none;
        background-color: #cbf4b0
    }

    .blue-background {
        background-image: none;
        background-color: #d0dfff;
    }
"""


#
#==============================================================================
class SudokuEncoding(CNF, object):
    """
        Sudoku grid encoder.
    """

    def __init__(self):
        """
            Constructor.
        """

        # initializing CNF's internal parameters
        super(SudokuEncoding, self).__init__()

        self.vpool = IDPool()

        # at least one value in each cell
        for i, j in itertools.product(range(9), range(9)):
            self.append([self.var(i, j, val) for val in range(9)])

        # at most one value in each row
        for i in range(9):
            for val in range(9):
                for j1, j2 in itertools.combinations(range(9), 2):
                    self.append([-self.var(i, j1, val), -self.var(i, j2, val)])

        # at most one value in each column
        for j in range(9):
            for val in range(9):
                for i1, i2 in itertools.combinations(range(9), 2):
                    self.append([-self.var(i1, j, val), -self.var(i2, j, val)])

        # at most one value in each square
        for val in range(9):
            for i in range(3):
                for j in range(3):
                    subgrid = itertools.product(range(3 * i, 3 * i + 3), range(3 * j, 3 * j + 3))
                    for c in itertools.combinations(subgrid, 2):
                        self.append([-self.var(c[0][0], c[0][1], val), -self.var(c[1][0], c[1][1], val)])

    def var(self, i, j, v):
        """
            Return Boolean variable corresponding to a given tuple.
        """

        return self.vpool.id(tuple([i + 1, j + 1, v + 1]))

    def cell(self, var):
        """
            Return a tuple for a given Boolean variable.
        """

        return self.vpool.obj(var)


#
#==============================================================================
class Puzzle(Gtk.Window):
    """
        PyGTK window for Sudoku.
    """

    def __init__(self, grid_encoding, nof_clues=20, get_muses=False,
            solver='m22'):
        """
            Constructor.
        """

        Gtk.Window.__init__(self, title='Sudoku Puzzle with SAT')

        self.provider = Gtk.CssProvider()
        self.provider.load_from_data(css_data.encode())
        Gtk.StyleContext.add_provider_for_screen(Gdk.Screen.get_default(),
            self.provider, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)

        self.init_popover()
        self.init_grid()

        self.oracle = Solver(name=solver, bootstrap_with=grid_encoding.clauses)
        self.muses = get_muses
        self.core = None

        self.vals = {}
        self.clues = set([])
        self.nof_clues = nof_clues

        self.enc = grid_encoding
        self.assumps = []

        random.seed()

    def init_grid(self):
        """
            Initialize the grid.
        """

        self.grid = Gtk.Grid()
        self.add(self.grid)

        self.cells = [[] for i in range(9)]
        for i in range(9):
            for j in range(9):
                self.cells[i].append(Gtk.Button(label=' '))
                self.grid.attach(self.cells[i][j], j, i, 1, 1)
                self.cells[i][j].set_size_request(50, 50)
                self.cells[i][j].set_sensitive(False)

                self.cells[i][j].i = i
                self.cells[i][j].j = j
                self.cells[i][j].v = None
                self.cells[i][j].connect('clicked', self.on_cell_clicked)
                self.cells[i][j].get_style_context().add_class('normal-button')

                if i in (3, 6):
                    self.cells[i][j].get_style_context().add_class('bordertop-cell')
                elif i in (2, 5):
                    self.cells[i][j].get_style_context().add_class('borderdown-cell')

                if j in (3, 6):
                    self.cells[i][j].get_style_context().add_class('borderleft-cell')
                elif j in (2, 5):
                    self.cells[i][j].get_style_context().add_class('borderright-cell')

        self.generator = Gtk.Button(label='Generate Puzzle')
        self.generator.connect('clicked', self.on_generator_clicked)
        self.generator.get_style_context().add_class('generate-button')
        self.generator.set_size_request(9 * 50, 50)
        self.grid.attach(self.generator, 0, 9, 9, 1)

    def init_popover(self):
        """
            Initialize the popover.
        """

        self.popover = Gtk.Popover()
        vgrid = Gtk.Grid()

        for i in range(3):
            for j in range(3):
                value = Gtk.ModelButton(label=str(3 * i + j + 1))
                value.connect('clicked', self.on_val_clicked)
                vgrid.attach(value, j, i, 1, 1)

        # special value button
        help_button = Gtk.ModelButton(label='Help!')
        help_button.connect('clicked', self.on_help_clicked)
        vgrid.attach(help_button, 0, 3, 2, 1)

        # erase button
        erase_button = Gtk.ModelButton(label='<')
        erase_button.connect('clicked', self.on_erase_clicked)
        vgrid.attach(erase_button, 2, 3, 1, 1)

        self.popover.add(vgrid)
        self.popover.set_position(Gtk.PositionType.BOTTOM)

    def __del__(self):
        """
            Destructor.
        """

        if self.oracle:
            self.oracle.delete()
            self.oracle = None

        Gtk.main_quit()

    def on_generator_clicked(self, widget):
        """
            Generate a new puzzle.
        """

        # delete everything
        self.reset_puzzle()

        cells = set(itertools.product(range(9), range(9)))

        while len(self.vals) < self.nof_clues:
            cell = random.choice(list(cells))
            cells.remove(cell)

            while True:
                val = random.choice(range(9))
                var = self.enc.var(cell[0], cell[1], val)

                if self.is_sat(extra_assumps=[var]):
                    self.vals[cell] = val
                    self.clues.add(var)
                    break

        for cell in self.vals:
            self.cells[cell[0]][cell[1]].set_label(str(self.vals[cell] + 1))
            self.cells[cell[0]][cell[1]].set_sensitive(False)
            self.cells[cell[0]][cell[1]].v = self.vals[cell]

        for cell in cells:
            self.cells[cell[0]][cell[1]].set_sensitive(True)

    def reset_puzzle(self):
        """
            Start from scratch.
        """

        for i in range(9):
            for j in range(9):
                self.cells[i][j].set_label(' ')
                self.cells[i][j].set_sensitive(True)
                self.cells[i][j].v = None
                self.cells[i][j].get_style_context().remove_class('red-background')
                self.cells[i][j].get_style_context().remove_class('green-background')
                self.cells[i][j].get_style_context().remove_class('blue-background')

        self.vals = {}
        self.clues = set([])
        self.core = None

    def on_cell_clicked(self, widget):
        """
            Show a matrix.
        """

        self.popover.set_relative_to(widget)
        self.popover.set_sensitive(True)
        self.popover.show_all()
        self.popover.popup()

    def on_val_clicked(self, widget):
        """
            Save clicked value.
        """

        cell = self.popover.get_relative_to()

        if self.core:
            val_ = cell.get_label()

            if val_ != ' ' and self.enc.var(cell.i, cell.j, int(val_) - 1) in self.core:
                for l in self.core:
                    i, j, v = self.enc.cell(l)

                    self.cells[i - 1][j - 1].get_style_context().remove_class('red-background')

                self.core = None
            else:
                return

        val_ = widget.get_label()

        cell.v = int(val_) - 1
        cell.set_label(val_)

        self.vals[(cell.i, cell.j)] = cell.v

        if not self.is_sat():
            self.core = self.oracle.get_core()

            # minimize a core to an MUS
            if self.muses:
                self.minimize_core()

            for l in self.core:
                i, j, v = self.enc.cell(l)
                self.cells[i - 1][j - 1].get_style_context().add_class('red-background')
        else:
            if len(self.vals) == 81:
                for i in range(9):
                    for j in range(9):
                        self.cells[i][j].set_sensitive(False)
                        self.cells[i][j].get_style_context().add_class('green-background')

    def is_sat(self, extra_assumps=[]):
        """
            Make a SAT call a return the result.
        """

        # map cell values into Boolean variables
        values = list(map(lambda k: self.enc.var(k[0], k[1], self.vals[k]), self.vals))

        return self.oracle.solve(assumptions=values + extra_assumps)

    def minimize_core(self):
        """
            Minimize and unsatisfiable core and get an MUS.
        """

        i = 0

        while i < len(self.core):
            to_test = self.core[:i] + self.core[(i + 1):]

            if self.oracle.solve(assumptions=to_test):
                i += 1
            else:
                self.core = to_test

    def on_help_clicked(self, widget):
        """
            Save clicked value.
        """

        cell = self.popover.get_relative_to()

        if self.core:
            val_ = cell.get_label()

            if val_ != ' ' and self.enc.var(cell.i, cell.j, int(val_) - 1) in self.core:
                cell.v = None
                cell.set_label(' ')
                del(self.vals[(cell.i, cell.j)])

                for l in self.core:
                    i, j, v = self.enc.cell(l)

                    self.cells[i - 1][j - 1].get_style_context().remove_class('red-background')

                self.core = None
            else:
                # the previous core was not resolved => do nothing
                return

        self.is_sat()  # making a SAT call, which should return True
        model = [None] + self.oracle.get_model()

        for val in range(9):
            if model[self.enc.var(cell.i, cell.j, val)] > 0:
                cell.v = val
                cell.set_label(str(val + 1))
                self.vals[(cell.i, cell.j)] = cell.v
                self.cells[cell.i][cell.j].get_style_context().add_class('blue-background')

                break

        if len(self.vals) == 81:
            for i in range(9):
                for j in range(9):
                    self.cells[i][j].set_sensitive(False)
                    self.cells[i][j].get_style_context().add_class('green-background')

    def on_erase_clicked(self, widget):
        """
            Save clicked value.
        """

        cell = self.popover.get_relative_to()
        val_ = cell.get_label()

        if val_ == ' ':
            return

        if self.core:
            if val_ != ' ' and self.enc.var(cell.i, cell.j, int(val_) - 1) in self.core:
                for l in self.core:
                    i, j, v = self.enc.cell(l)

                    self.cells[i - 1][j - 1].get_style_context().remove_class('red-background')

                self.core = None

        for i in range(9):
            for j in range(9):
                self.cells[i][j].set_label(' ')
                self.cells[i][j].set_sensitive(True)
                self.cells[i][j].v = None
                self.cells[i][j].get_style_context().remove_class('green-background')

        cell.v = None
        cell.set_label(' ')
        del(self.vals[(cell.i, cell.j)])


#
#==============================================================================
def parse_options():
    """
        Parses command-line options:
    """

    try:
        opts, args = getopt.getopt(sys.argv[1:],
                                   'c:hms:', ['clues=', 'help', 'muses', 'solver='])
    except getopt.GetoptError as err:
        sys.stderr.write(str(err).capitalize())
        usage()
        sys.exit(1)

    get_muses = False
    nof_clues = 20
    solver = 'm22'

    for opt, arg in opts:
        if opt in ('-c', '--clues'):
            nof_clues = int(arg)
        elif opt in ('-h', '--help'):
            usage()
            sys.exit(0)
        elif opt in ('-m', '--muses'):
            get_muses = True
        elif opt in ('-s', '--solver'):
            solver = str(arg)
        else:
            assert False, 'Unhandled option: {0} {1}'.format(opt, arg)

    return nof_clues, get_muses, solver


#
#==============================================================================
def usage():
    """
        Prints usage message.
    """

    print('Usage:', os.path.basename(sys.argv[0]), '[options] dimacs-file')
    print('Options:')
    print('        -c, --clues=<int>    The number of clues to generate')
    print('                             Available values: [1 .. 81] (default: 20)')
    print('        -h, --help')
    print('        -m, --muses          Get MUSes instead of cores')
    print('        -s, --solver         SAT solver to use')
    print('                             Available values: g3, g4, lgl, mcb, mcm, mpl, m22, mc, mgh (default: m22)')


#
#==============================================================================
if __name__ == '__main__':
    nof_clues, get_muses, solver = parse_options()
    puzzle = Puzzle(SudokuEncoding(), nof_clues=nof_clues, get_muses=get_muses,
            solver=solver)

    puzzle.connect('destroy', Gtk.main_quit)
    puzzle.show_all()

    Gtk.main()
