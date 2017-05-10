"""Tests the State class."""

import sys
if __name__ == '__main__':
    sys.path.insert(1, '..')
    sys.path.insert(1, '../..')
else:
    sys.path.insert(1, 'state')

from StringIO import StringIO

from unittest import TestCase, main

from State import State

class SpeciesReferenceTest(TestCase):
    def test(self):
        state = State()
        state.read('../../examples/cain/DecayingDimerizing.xml')
        state.write(StringIO())

if __name__ == '__main__':
    main()
