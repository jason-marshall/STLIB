"""Tests the Mt19937 class."""

import sys
if __name__ == '__main__':
    sys.path.insert(1, '..')
else:
    sys.path.insert(1, 'state')

from unittest import TestCase, main

from Mt19937 import generateState

class Mt19937Test(TestCase):
    def test(self):
        state, seed = generateState(0)
        self.assertEqual(len(state), 625)
        for x in state:
            assert 0 <= x and x < 0xffffffff
        assert 0 <= seed and seed < 0xffffffff

if __name__ == '__main__':
    main()
