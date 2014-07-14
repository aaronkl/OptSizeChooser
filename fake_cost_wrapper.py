'''
Created on 10.04.2014


@author: Simon Bartels

Chooser that fakes costs.

'''
import entropy_search_chooser

def init(expt_dir, arg_string):
    return FakeCostWrapper(expt_dir, arg_string)

class FakeCostWrapper(object):
    def __init__(self, expt_dir, arg_string):
        self._wrapped_chooser = entropy_search_chooser.init(expt_dir, arg_string)

    def next(self, grid, values, durations, candidates, pending, complete):
        comp = grid[complete, 0]
        durations[complete] = (20 * comp + 5) ** 3
        return self._wrapped_chooser.next(grid, values, durations, candidates, pending, complete)
