"""Tests the TriggerEvent class."""

import sys
if __name__ == '__main__':
    sys.path.insert(1, '..')
else:
    sys.path.insert(1, 'simulation')
    
from unittest import TestCase, main

from TriggerEvent import TriggerEvent
from Model import Model
from Parameter import Parameter

class TriggerEventTest(TestCase):
    def setUp(self):
        self.model = Model(0)
        self.model.parameters['p1'] = Parameter(0.)
        self.model.initialize()

    def tearDown(self):
        self.model = None

    def testParameter(self):
        e = TriggerEvent(self.model, 'p1=0', 'p1>0', 0, False)
        e.initialize()
        assert e.count == 0
        assert not e.evaluate(self.model.time)
        self.model.parameters['p1'].value = 1
        assert e.evaluate(self.model.time)
        e.fire()
        assert e.count == 1
        assert self.model.parameters['p1'].value == 0
        assert not e.evaluate(self.model.time)

    def testTimeDependent(self):
        e = TriggerEvent(self.model, 'p1=0', 't>1', 0, False)
        e.initialize()
        assert e.count == 0
        assert not e.evaluate(self.model.time)
        self.model.time = 2
        assert e.evaluate(self.model.time)
        assert not e.evaluate(self.model.time)

if __name__ == '__main__':
    main()
