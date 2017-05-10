"""Tests the Reaction class."""

import sys
if __name__ == '__main__':
    sys.path.insert(1, '..')
else:
    sys.path.insert(1, 'simulation')
    
from unittest import TestCase, main

from Reaction import Reaction
from Model import Model
from Parameter import Parameter
from Species import Species
from SpeciesReference import SpeciesReference

class ReactionTest(TestCase):
    def setUp(self):
        self.model = Model(0)
        self.model.parameters['p1'] = Parameter(2.)
        self.model.species['s1'] = Species(3.)
        self.model.species['s2'] = Species(5.)
        self.model.initialize()

    def tearDown(self):
        self.model = None

    def testTrivialReaction(self):
        r = Reaction(self.model, [], [], 'p1')
        r.initialize()
        self.assertEqual(r.propensity(), 2.)
        self.assertEqual(r.count, 0)

    def testFirstOrder(self):
        r = Reaction(self.model, [SpeciesReference('s1', 1)],
                     [SpeciesReference('s2', 1)], 'p1*s1')
        r.initialize()
        self.assertEqual(r.propensity(), 6.)
        self.assertEqual(r.count, 0)
        r.fire()
        self.assertEqual(self.model.species['s1'].amount, 2.)
        self.assertEqual(self.model.species['s2'].amount, 6.)
        self.assertEqual(r.count, 1)
    
    def testTimeDependent(self):
        r = Reaction(self.model, [], [], 't')
        r.initialize()
        self.assertEqual(r.propensity(), self.model.time)
        self.assertEqual(r.count, 0)
        self.model.time = 1
        self.assertEqual(r.propensity(), self.model.time)

if __name__ == '__main__':
    main()
