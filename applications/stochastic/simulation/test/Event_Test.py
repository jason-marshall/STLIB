"""Tests the Event class."""

import sys
if __name__ == '__main__':
    sys.path.insert(1, '..')
else:
    sys.path.insert(1, 'simulation')
    
from unittest import TestCase, main

from Event import Event, makeTriggerTimeEvent
from Model import Model
from Parameter import Parameter

class EventTest(TestCase):
    def setUp(self):
        self.model = Model(0)
        self.model.parameters['p1'] = Parameter(1.)
        self.model.initialize()

    def tearDown(self):
        self.model = None

    def testNoAssignments(self):
        # No assignments.
        e = Event(self.model)
        e.initialize()
        self.assertEqual(e.count, 0)
        e.fire()
        self.assertEqual(e.count, 1)

    def testSimpleAssignment(self):
        e = Event(self.model, 'p1=2')
        e.initialize()
        self.assertEqual(e.count, 0)
        e.fire()
        self.assertEqual(e.count, 1)
        self.assertEqual(self.model.parameters['p1'].value, 2)

    def testAssignmentFormula(self):
        e = Event(self.model, 'p1=2*p1')
        e.initialize()
        e.fire()
        self.assertEqual(self.model.parameters['p1'].value, 2)

    def testTimeDependent(self):
        e = Event(self.model, 'p1=t')
        e.initialize()
        f = makeTriggerTimeEvent(e)
        self.assertEqual(self.model.time, 0)
        self.model.time = 1.
        f.fire()
        self.assertEqual(self.model.parameters['p1'].value, 0)

if __name__ == '__main__':
    main()
