"""Tests the FirstReaction class."""

import sys
if __name__ == '__main__':
    sys.path.insert(1, '..')
else:
    sys.path.insert(1, 'simulation')
    
from unittest import TestCase, main

from FirstReaction import FirstReaction

from Model import Model
from Species import Species
from SpeciesReference import SpeciesReference
from Reaction import Reaction
from TimeEvent import TimeEvent
from TriggerEvent import TriggerEvent

class FirstReactionTest(TestCase):
    def testImmigration(self):
        m = Model(0)
        m.species['s1'] = Species(0)
        m.reactions['r1'] = Reaction(m, [], [SpeciesReference('s1', 1)], '1')
        solver = FirstReaction(m, 1000)
        solver.initialize()
        solver.simulate(10.)
        assert m.species['s1'].amount >= 0

    def testIncrementEvent(self):
        m = Model(0)
        m.species['s1'] = Species(0)
        m.timeEvents['ti1'] = TimeEvent(m, 's1=s1+1', range(10))
        solver = FirstReaction(m, 1000)
        solver.initialize()
        solver.simulate(10.)
        self.assertEqual(m.species['s1'].amount, 10)

    def testImmigrationAndIncrement(self):
        m = Model(0)
        m.species['s1'] = Species(0)
        m.reactions['r1'] = Reaction(m, [], [SpeciesReference('s1', 1)], '1')
        m.timeEvents['ti1'] = TimeEvent(m, 's1=s1+1', range(10))
        solver = FirstReaction(m, 1000)
        solver.initialize()
        solver.simulate(10.)
        assert m.species['s1'].amount >= 0

    def testImmigrationIncrementAndDecimate(self):
        m = Model(0)
        m.species['s1'] = Species(0)
        m.reactions['r1'] = Reaction(m, [], [SpeciesReference('s1', 1)], '1')
        m.timeEvents['ti1'] = TimeEvent(m, 's1=s1+1', range(10))
        m.triggerEvents['tr1'] = TriggerEvent(m, 's1=s1//2', 's1>10', 0, False)
        solver = FirstReaction(m, 1000)
        solver.initialize()
        solver.simulate(10.)
        assert m.species['s1'].amount >= 0

if __name__ == '__main__':
    main()
