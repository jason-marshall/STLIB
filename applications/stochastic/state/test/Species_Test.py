"""Tests the Species class."""

import sys
if __name__ == '__main__':
    sys.path.insert(1, '..')
    sys.path.insert(1, '../..')
else:
    sys.path.insert(1, 'state')

import math

from unittest import TestCase, main

from Species import Species
from Value import Value
from ParameterEvaluation import evaluateInitialAmounts

class SpeciesTest(TestCase):
    def test(self):
        x = Species('C1', 'species 1', '0')
        assert not x.hasErrors(['C1'])
        assert x.hasErrors([])
        assert not evaluateInitialAmounts({'s1':x}, {})
        assert x.initialAmountValue == 0

        x = Species('C1', 'species 1', '')
        assert x.hasErrors(['C1'])
        error = evaluateInitialAmounts({'s1':x}, {})
        assert error

        x = Species('C1', 'species 2', '1')
        assert not x.hasErrors(['C1'])
        assert not evaluateInitialAmounts({'s2':x}, {})
        assert x.initialAmountValue == 1

        x = Species('C1', 'species 2', '-1')
        # I don't check the initial amount expression with hasErrors().
        assert not x.hasErrors(['C1'])
        error = evaluateInitialAmounts({'s2':x}, {})
        assert error

        x = Species('C1', 'species 2', 'p1')
        assert not x.hasErrors(['C1'])
        error = evaluateInitialAmounts({'s1':x}, {})
        assert error

        x = Species('C1', 'species 2', 'p1')
        p = Value('', '5.0')
        p.value = 5.0
        assert not x.hasErrors(['C1'])
        assert not evaluateInitialAmounts({'s1':x}, {'p1':p})
        assert x.initialAmountValue == 5

        x = Species('C1', 'species 2', 'sqrt(p1)')
        assert not x.hasErrors(['C1'])
        assert not evaluateInitialAmounts({'s1':x}, {'p1':p})
        assert x.initialAmountValue == math.sqrt(5)

        x = Species('C1', 'species 2', 'p2')
        assert not x.hasErrors(['C1'])
        error = evaluateInitialAmounts({'s1':x}, {'p1':p})
        assert error

if __name__ == '__main__':
    main()
