"""Tests the SimulationOutput class."""

import sys
if __name__ == '__main__':
    sys.path.insert(1, '..')
else:
    sys.path.insert(1, 'state')

from unittest import TestCase, main

from SimulationOutput import SimulationOutput

class SimulationOutputTest(TestCase):
    def test(self):
        x = SimulationOutput([0, 1], [0, 1])
        assert not x.hasErrors()

if __name__ == '__main__':
    main()
