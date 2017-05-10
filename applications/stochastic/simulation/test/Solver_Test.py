"""Tests the Solver class."""

import sys
if __name__ == '__main__':
    sys.path.insert(1, '..')
else:
    sys.path.insert(1, 'simulation')
    
from unittest import TestCase, main

from Solver import Solver
from Model import Model

class SolverTest(TestCase):
    def test(self):
        m = Model(0)
        s = Solver(m, 1000)
        s.initialize()
        self.assertEqual(s.stepCount, 0)
        s.incrementStepCount()
        self.assertEqual(s.stepCount, 1)

if __name__ == '__main__':
    main()
