"""Tests the Model class."""

import sys
if __name__ == '__main__':
    sys.path.insert(1, '..')
else:
    sys.path.insert(1, 'simulation')
    
from unittest import TestCase, main

from Model import Model
from Parameter import Parameter

class EventTest(TestCase):
    def test(self):
        m = Model(0)
        m.parameters['p1'] = Parameter(1.)
        m.initialize()
        self.assertEqual(m.decorate('p1'), "m.parameters['p1'].value")
        self.assertEqual(m.decorate('p2'), 'p2')

if __name__ == '__main__':
    main()
