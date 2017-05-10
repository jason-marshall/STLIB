"""Tests the TimeEvent class."""

import sys
if __name__ == '__main__':
    sys.path.insert(1, '..')
else:
    sys.path.insert(1, 'simulation')
    
from unittest import TestCase, main

from TimeEvent import TimeEvent
from Model import Model
from Parameter import Parameter

class TimeEventTest(TestCase):
    def setUp(self):
        self.model = Model(0)
        self.model.parameters['p1'] = Parameter(1.)
        self.model.initialize()

    def tearDown(self):
        self.model = None

    def testAssignment(self):
        e = TimeEvent(self.model, 'p1=2', [])
        e.initialize()
        self.assertEqual(e.count, 0)
        e.fire()
        self.assertEqual(e.count, 1)
        self.assertEqual(self.model.parameters['p1'].value, 2)

    def testFormula(self):
        e = TimeEvent(self.model, 'p1=2*p1', [])
        e.initialize()
        e.fire()
        self.assertEqual(self.model.parameters['p1'].value, 2)

if __name__ == '__main__':
    main()
