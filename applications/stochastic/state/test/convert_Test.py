"""Tests the convert module."""

import sys
if __name__ == '__main__':
    sys.path.insert(1, '..')
    sys.path.insert(1, '../..')
else:
    sys.path.insert(1, 'state')

from unittest import TestCase, main

from convert import makeModel, makeTimeSeriesUniform

from Model import Model as StateModel
from Species import Species as StateSpecies
from Method import Method
from Reaction import Reaction as StateReaction
from TimeEvent import TimeEvent as StateTimeEvent
from TriggerEvent import TriggerEvent as StateTriggerEvent
from SpeciesReference import SpeciesReference as StateSpeciesReference
from TimeSeriesFrames import TimeSeriesFrames
from simulation.FirstReaction import FirstReaction

class ConvertTest(TestCase):
    def testTimeHomogeneous(self):
        model = StateModel()
        model.id = 'model'
        model.speciesIdentifiers.append('s1')
        model.species['s1'] = StateSpecies('C1', 'species 1', '13')
        model.speciesIdentifiers.append('s2')
        model.species['s2'] = StateSpecies('C1', 'species 2', '17')
        model.reactions.append(
            StateReaction('r1', 'reaction 1', [StateSpeciesReference('s1')], 
                          [StateSpeciesReference('s2')], True, '1.5'))
        model.reactions.append(
            StateReaction('r2', 'reaction 2', 
                     [StateSpeciesReference('s1'), StateSpeciesReference('s2')], 
                     [StateSpeciesReference('s1', 2)], True, '2.5'))
        error = model.evaluate()
        assert not error
        method = Method()
        output = TimeSeriesFrames()
        solver = FirstReaction(makeModel(model, method), 2**64)
        simulator = makeTimeSeriesUniform(solver, model, output)

    def testTimeInhomogeneous(self):
        model = StateModel()
        model.id = 'model'
        model.speciesIdentifiers.append('s1')
        model.species['s1'] = StateSpecies('', 'species 1', '13')
        model.speciesIdentifiers.append('s2')
        model.species['s2'] = StateSpecies('', 'species 2', '17')
        model.reactions.append(
            StateReaction('r1', 'reaction 1', [StateSpeciesReference('s1')], 
                     [StateSpeciesReference('s2')], True, '2+sin(t)'))
        model.reactions.append(
            StateReaction('r2', 'reaction 2', 
                     [StateSpeciesReference('s1'), StateSpeciesReference('s2')], 
                     [StateSpeciesReference('s1', 2)], False, '1+exp(-t)'))
        error = model.evaluateInhomogeneous()
        assert not error
        method = Method()
        output = TimeSeriesFrames()
        solver = FirstReaction(makeModel(model, method), 2**64)
        simulator = makeTimeSeriesUniform(solver, model, output)

if __name__ == '__main__':
    main()
