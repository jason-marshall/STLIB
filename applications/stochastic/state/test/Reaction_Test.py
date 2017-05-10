"""Tests the Reaction class."""

import sys
if __name__ == '__main__':
    sys.path.insert(1, '..')
    sys.path.insert(1, '../..')
else:
    sys.path.insert(1, 'state')

from unittest import TestCase, main

from Reaction import Reaction
from SpeciesReference import SpeciesReference
from ParameterEvaluation import evaluatePropensityFactors

class ReactionTest(TestCase):
    def test(self):
        assert Reaction('r1', '', [], [], False, '').hasErrors([], True)

        # Mass action kinetics.

        identifiers = ['s1', 's2']
        a = SpeciesReference('s1', 1)
        x = Reaction('r1', '1', [a], [], True, '0')
        evaluatePropensityFactors([x], {})
        assert not x.hasErrors(identifiers, True)
        assert x.hasErrors([], True)
        assert x.influence('s1') == -1
        assert x.influence('s2') == 0

        p = SpeciesReference('s2', 1)
        x = Reaction('r1', 'reaction', [a], [p], True, '1')
        evaluatePropensityFactors([x], {})
        assert not x.hasErrors(identifiers, True)
        assert x.hasErrors([], True)
        assert x.influence('s1') == -1
        assert x.influence('s2') == 1

        b = SpeciesReference('s2', 2)
        x = Reaction('r1', 'name', [a, b], [p], True, '3')
        evaluatePropensityFactors([x], {})
        assert not x.hasErrors(identifiers, True)
        assert x.hasErrors([], True)
        assert x.influence('s1') == -1
        assert x.influence('s2') == -1

        print('Time inhomogeneous')
        p = SpeciesReference('s2', 1)
        x = Reaction('r1', 'reaction', [a], [p], True, '2+sin(t)')
        assert not x.hasErrors(identifiers, True)
        assert x.hasErrors([], True)
        assert x.influence('s1') == -1
        assert x.influence('s2') == 1

        p = SpeciesReference('s2', 1)
        x = Reaction('r1', 'name', [a, b], [p], True, '2+sin(t)')
        assert not x.hasErrors(identifiers, True)
        assert x.hasErrors([], True)
        assert x.influence('s1') == -1
        assert x.influence('s2') == -1

        # Custom kinetics.

        x.massAction = False
        x.propensity = '5*s1*s2'
        assert not x.hasErrors(identifiers, True)

        x.propensity = '5'
        assert not x.hasErrors(identifiers, True)

        x.propensity = 's10'
        assert not x.hasErrors(identifiers, True)

        identifiers.append('s3')
        x.propensity = 's3'
        assert x.hasErrors(identifiers, True)
        assert not x.hasErrors(identifiers, False)

        # Time inhomogeneous.
        x.propensity = '2+sin(t)'
        assert not x.hasErrors(identifiers, True)

        # Convert custom to mass action.

        species = ['s1']
        parameters = []
        s1 = SpeciesReference('s1', 1)

        # Zeroth order.
        x = Reaction('r1', 'reaction 1', [], [s1], False, '0')
        x.convertCustomToMassAction(species, parameters)
        assert x.propensity == '0.0'
        assert x.massAction

        x = Reaction('r1', 'reaction 1', [], [s1], False, '1')
        x.convertCustomToMassAction(species, parameters)
        assert x.propensity == '1.0'
        assert x.massAction

        x = Reaction('r1', 'reaction 1', [], [s1], False, 's2')
        x.convertCustomToMassAction(species, parameters)
        assert x.propensity == 's2'
        assert not x.massAction

        x = Reaction('r1', 'reaction 1', [], [s1], False, 's1')
        x.convertCustomToMassAction(species, parameters)
        assert x.propensity == 's1'
        assert not x.massAction

        # First order.
        x = Reaction('r1', 'reaction 1', [s1], [], False, '0')
        x.convertCustomToMassAction(species, parameters)
        assert x.propensity == '0'
        assert x.massAction

        x = Reaction('r1', 'reaction 1', [s1], [], False, '1')
        x.convertCustomToMassAction(species, parameters)
        assert x.propensity == '1'
        assert not x.massAction

        x = Reaction('r1', 'reaction 1', [s1], [], False, 's1')
        x.convertCustomToMassAction(species, parameters)
        assert x.propensity == '1'
        assert x.massAction

        x = Reaction('r1', 'reaction 1', [s1], [], False, 's1*s1')
        x.convertCustomToMassAction(species, parameters)
        assert x.propensity == 's1*s1'
        assert not x.massAction

        x = Reaction('r1', 'reaction 1', [s1], [], False, 's2')
        x.convertCustomToMassAction(species, parameters)
        assert x.propensity == 's2'
        assert not x.massAction

        # Second order.
        s1 = SpeciesReference('s1', 2)

        x = Reaction('r1', 'reaction 1', [s1], [], False, '0')
        x.convertCustomToMassAction(species, parameters)
        assert x.propensity == '0'
        assert x.massAction

        x = Reaction('r1', 'reaction 1', [s1], [], False, '1')
        x.convertCustomToMassAction(species, parameters)
        assert x.propensity == '1'
        assert not x.massAction

        x = Reaction('r1', 'reaction 1', [s1], [], False, 's1')
        x.convertCustomToMassAction(species, parameters)
        assert x.propensity == 's1'
        assert not x.massAction

        x = Reaction('r1', 'reaction 1', [s1], [], False, 's1*(s1-1)/2')
        x.convertCustomToMassAction(species, parameters)
        assert x.propensity == '1'
        assert x.massAction

        x = Reaction('r1', 'reaction 1', [s1], [], False, 's2')
        x.convertCustomToMassAction(species, parameters)
        assert x.propensity == 's2'
        assert not x.massAction

if __name__ == '__main__':
    main()
