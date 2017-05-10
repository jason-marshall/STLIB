"""Convert the state classes to simulation classes."""

# If we are running the unit tests.
if __name__ == '__main__':
    import sys
    sys.path.insert(1, '..')

from simulation.Model import Model
from simulation.Species import Species
from simulation.SpeciesReference import SpeciesReference
from simulation.Reaction import Reaction
from simulation.TimeEvent import TimeEvent
from simulation.TriggerEvent import TriggerEvent
from simulation.Parameter import Parameter
from simulation.TimeSeriesUniform import TimeSeriesUniform

# CONTINUE: Check for errors.
def makeModel(model, method):
    """Make a simulation.Model from a state.Model and a state.Method."""
    m = Model(method.startTime)
    for key in model.species:
        m.species[key] = Species(model.species[key].initialAmountValue)
    for r in model.reactions:
        reactants = [SpeciesReference(x.species, x.stoichiometry) for x in
                     r.reactants]
        products = [SpeciesReference(x.species, x.stoichiometry) for x in
                     r.products]
        if r.massAction:
            propensity = '(' + r.propensity + ')'
            denominator = 1.0
            for x in reactants:
                propensity += '*%s' % x.species
                for i in range(1, x.stoichiometry):
                    propensity += '*(%s-%d)' % (x.species, i)
                for i in range(1, x.stoichiometry + 1):
                    denominator *= i
            if denominator != 1:
                propensity += '*' + repr(1./denominator)
        else:
            propensity = r.propensity
        m.reactions[r.id] = Reaction(m, reactants, products, propensity)
    for x in model.timeEvents:
        m.timeEvents[x.id] = TimeEvent(m, x.assignments, eval(x.times))
    for x in model.triggerEvents:
        m.triggerEvents[x.id] = TriggerEvent(m, x.assignments, x.trigger,
                                             x.delay,
                                             x.useValuesFromTriggerTime)
    # Parameters and compartments are grouped together.
    for key in model.compartments:
        initialValue = model.compartments[key].value
        if not initialValue:
            initialValue = 1.
        m.parameters[key] = Parameter(initialValue)
    for key in model.parameters:
        m.parameters[key] = Parameter(model.parameters[key].value)
    return m

def makeTimeSeriesUniform(solver, model, output):
    """Make a simulation.TimeSeriesUniform from a solver, a state.Model, and a
    state.TimeSeriesFrames."""
    recordedSpecies = [model.speciesIdentifiers[index] for index in
                       output.recordedSpecies]
    recordedReactions = [model.reactions[index].id for index in
                         output.recordedReactions]
    return TimeSeriesUniform(solver, recordedSpecies, recordedReactions,
                             output.frameTimes)
