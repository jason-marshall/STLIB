"""XML content handler."""
# CONTINUE: store attributes.keys() in keys.

if __name__ == '__main__':
    import sys
    sys.path.insert(1, '..')

from state.Model import Model, writeModelXml
from state.Method import Method
from state.TimeSeriesFrames import TimeSeriesFrames
from state.TimeSeriesAllReactions import TimeSeriesAllReactions
from state.HistogramFrames import HistogramFrames
from state.HistogramAverage import HistogramAverage
from state.StatisticsFrames import StatisticsFrames
from state.StatisticsAverage import StatisticsAverage
from state.Value import Value
from state.Species import Species
from state.Reaction import Reaction
from state.TimeEvent import TimeEvent
from state.TriggerEvent import TriggerEvent
from state.SpeciesReference import SpeciesReference

import xml.sax.handler

class ContentHandler(xml.sax.ContentHandler):
    """An XML content handler.

    Inherit from the XML SAX content handler.

    Error messages are collected in the string self.errors. After recording an
    error there is a return statement if the element cannot be further 
    processed."""

    def __init__(self):
        xml.sax.ContentHandler.__init__(self)
        self.models = {}
        self.methods = {}
        self.output = {}
        self.seed = None
        self.listOfMt19937States = []
        # The stack of the enclosing elements.
        self.elements = []
        self.errors = ''
        self.content = ''

    def startElement(self, name, attributes):
        self.elements.append(name)
        #
        # The top level element.
        if name == 'cain':
            if 'version' in attributes.keys():
                self.version = float(attributes['version'])
            else:
                self.version = 1.6
        elif name == 'sbml':
            # Fatal error.
            raise Exception('This is an SBML file. One may only directly open Cain files. Use File->Import SBML instead.')
        #
        # Elements for the model.
        #
        elif name == 'listOfModels':
            pass
        elif name == 'model':
            # Start a new model.
            self.model = Model()
            if 'id' in attributes.keys():
                self.model.id = attributes['id']
            if 'name' in attributes.keys():
                self.model.name = attributes['name']
        elif name == 'listOfParameters':
            # Start the dictionary of parameters.
            self.model.parameters = {}
        elif name == 'listOfCompartments':
            # Start the dictionary of compartments.
            self.model.compartments = {}
        elif name == 'listOfSpecies':
            # Start the dictionary of species and list of species identifiers.
            self.model.species = {}
            self.model.speciesIdentifiers = []
        elif name == 'listOfReactions':
            # Start the list of reactions.
            self.model.reactions = []
        elif name == 'listOfTimeEvents':
            # Start the list of time events.
            self.model.timeEvents = []
        elif name == 'listOfTriggerEvents':
            # Start the list of trigger events.
            self.model.triggerEvents = []
        elif name == 'parameter':
            # Append a parameter.
            if not 'id' in attributes.keys():
                self.errors += 'Missing id attribute in parameter.\n'
                return
            x = Value('', '')
            error = x.readXml(attributes)
            if error:
                self.errors += error
                return
            self.model.parameters[attributes['id']] = x
        elif name == 'compartment':
            # Append a compartment.
            if not 'id' in attributes.keys():
                self.errors += 'Missing id attribute in compartment.\n'
                return
            if not attributes['id']:
                self.errors += 'Compartment identifier is empty.\n'
                return
            compartment = Value('', '')
            compartment.readXml(attributes)
            self.model.compartments[attributes['id']] = compartment
        elif name == 'species':
            # Append a species.
            if not 'id' in attributes.keys():
                self.errors += 'Missing id attribute in species.\n'
                return
            x = Species(None, None, None)
            x.readXml(attributes)
            self.model.species[attributes['id']] = x
            self.model.speciesIdentifiers.append(attributes['id'])
        elif name == 'reaction':
            # Append a reaction.
            x = Reaction('', '', [], [], True, '')
            error = x.readXml(attributes)
            if error:
                self.errors += error
                return
            self.model.reactions.append(x)
        elif name == 'timeEvent':
            # Append a time event.
            x = TimeEvent('', '', '', '')
            error = x.readXml(attributes)
            if error:
                self.errors += error
                return
            self.model.timeEvents.append(x)
        elif name == 'triggerEvent':
            # Append a trigger event.
            x = TriggerEvent('', '', '', '', 0., False)
            error = x.readXml(attributes)
            if error:
                self.errors += error
                return
            self.model.triggerEvents.append(x)
        elif name == 'listOfReactants':
            if not self.model.reactions:
                self.errors += 'Badly placed listOfReactants tag.\n'
                return
        elif name == 'listOfProducts':
            if not self.model.reactions:
                self.errors += 'Badly placed listOfProducts tag.\n'
                return
        elif name == 'listOfModifiers':
            if not self.model.reactions:
                self.errors += 'Badly placed listOfModifiers tag.\n'
                return
        elif name == 'speciesReference':
            # Add the reactant or product to the current reaction.
            if not self.model.reactions:
                self.errors += 'Badly placed speciesReference tag.\n'
                return
            if not 'species' in attributes.keys():
                self.errors +=\
                    'Missing species attribute in speciesReference.\n'
                return
            if 'stoichiometry' in attributes.keys():
                stoichiometry = int(attributes['stoichiometry'])
            else:
                stoichiometry = 1
            # No need to record if the stoichiometry is zero.
            if stoichiometry != 0:
                sr = SpeciesReference(attributes['species'], stoichiometry)
                if self.elements[-2] == 'listOfReactants':
                    self.model.reactions[-1].reactants.append(sr)
                elif self.elements[-2] == 'listOfProducts':
                    self.model.reactions[-1].products.append(sr)
                else:
                    self.errors += 'Badly placed speciesReference tag.\n'
                    return
        elif name == 'modifierSpeciesReference':
            # Add to the reactants and products of the current reaction.
            if not self.model.reactions:
                self.errors += 'Badly placed modifierSpeciesReference tag.\n'
                return
            if not 'species' in attributes.keys():
                self.errors +=\
                    'Missing species attribute in modifierSpeciesReference.\n'
                return
            if self.elements[-2] != 'listOfModifiers':
                self.errors += 'Badly placed modifierSpeciesReference tag.\n'
                return
            sr = SpeciesReference(attributes['species'])
            self.model.reactions[-1].reactants.append(sr)
            self.model.reactions[-1].products.append(sr)
        #
        # Elements for the simulation parameters.
        #
        elif name == 'listOfMethods':
            pass
        elif name == 'method':
            m = Method()
            error = m.readXml(attributes)
            if error:
                self.errors += error
                return
            self.methods[m.id] = m
        #
        # Elements for the simulation output.
        #
        elif name == 'listOfOutput':
            pass
        # CONTINUE: trajectoryFrames is deprecated.
        elif name in ('timeSeriesFrames', 'trajectoryFrames'):
            if not 'model' in attributes.keys():
                self.errors +=\
                    'Missing model attribute in timeSeriesFrames.\n'
                return
            if not 'method' in attributes.keys():
                self.errors += 'Missing method attribute in timeSeriesFrames.\n'
                return
            key = (attributes['model'], attributes['method'])
            # An ensemble of trajectories should not be listed twice.
            if key in self.output:
                self.errors += 'Simulation output (' + key[0] + ', ' +\
                    key[1] + ') listed twice.\n'
            else:
                # Start a new ensemble.
                self.output[key] = TimeSeriesFrames()
                self.currentOutput = self.output[key]
        # CONTINUE: trajectoryAllReactions is deprecated.
        elif name in ('timeSeriesAllReactions', 'trajectoryAllReactions'):
            if not 'model' in attributes.keys():
                self.errors +=\
                    'Missing model attribute in timeSeriesAllReactions.\n'
                return
            if not 'method' in attributes.keys():
                self.errors += 'Missing method attribute in timeSeriesAllReactions.\n'
                return
            key = (attributes['model'], attributes['method'])
            if self.version >= 1:
                if not 'initialTime' in attributes.keys():
                    self.errors +=\
                        'Missing initialTime attribute in timeSeriesAllReactions.\n'
                    return
                if not 'finalTime' in attributes.keys():
                    self.errors +=\
                        'Missing finalTime attribute in timeSeriesAllReactions.\n'
                    return
            else:
                if not 'endTime' in attributes.keys():
                    self.errors +=\
                        'Missing endTime attribute in timeSeriesAllReactions.\n'
                    return
            # An ensemble of trajectories should not be listed twice.
            if key in self.output:
                self.errors += 'Simulation output (' + key[0] + ', ' +\
                    key[1] + ') listed twice.\n'
                return
            # Check that the model has been defined.
            if not attributes['model'] in self.models:
                self.errors += 'timeSeriesAllReactions uses an undefined model.\n'
                return
            # Start a new ensemble. By definition all of the species and
            # reactions are recorded.
            model = self.models[attributes['model']]
            if self.version >= 1:
                self.output[key] =\
                    TimeSeriesAllReactions(range(len(model.speciesIdentifiers)),
                                           range(len(model.reactions)),
                                           float(attributes['initialTime']),
                                           float(attributes['finalTime']))
            else:
                # CONTINUE: Deprecated.
                self.output[key] =\
                    TimeSeriesAllReactions(range(len(model.speciesIdentifiers)),
                                           range(len(model.reactions)),
                                           0.,
                                           float(attributes['endTime']))
            self.currentOutput = self.output[key]
        elif name == 'histogramFrames':
            for key in ('model', 'method', 'numberOfTrajectories'):
                if not key in attributes.keys():
                    self.errors +=\
                        'Missing ' + key + ' attribute in histogramFrames.\n'
                    return
            # CONTINUE: Make multiplicity mandatory.
            if 'multiplicity' in attributes.keys():
                multiplicity = int(attributes['multiplicity'])
            else:
                multiplicity = 2
            key = (attributes['model'], attributes['method'])
            # An ensemble of trajectories should not be listed twice.
            if key in self.output:
                self.errors += 'Simulation output (' + key[0] + ', ' +\
                    key[1] + ') listed twice.\n'
            else:
                # Check that the method has been defined.
                if not attributes['method'] in self.methods:
                    self.errors += 'HistogramFrames uses an undefined method.\n'
                    return
                # Start a new ensemble.
                method = self.methods[attributes['method']]
                self.output[key] = HistogramFrames(method.numberOfBins,
                                                   multiplicity)
                self.currentOutput = self.output[key]
                self.currentOutput.numberOfTrajectories =\
                    float(attributes['numberOfTrajectories'])
        elif name == 'histogramAverage':
            for key in ('model', 'method', 'numberOfTrajectories'):
                if not key in attributes.keys():
                    self.errors +=\
                        'Missing ' + key + ' attribute in histogramAverage.\n'
                    return
            # CONTINUE: Make multiplicity mandatory.
            if 'multiplicity' in attributes.keys():
                multiplicity = int(attributes['multiplicity'])
            else:
                multiplicity = 2
            key = (attributes['model'], attributes['method'])
            # An ensemble of trajectories should not be listed twice.
            if key in self.output:
                self.errors += 'Simulation output (' + key[0] + ', ' +\
                    key[1] + ') listed twice.\n'
            else:
                # Check that the method has been defined.
                if not attributes['method'] in self.methods:
                    self.errors += 'HistogramAverage uses an undefined method.\n'
                    return
                # Start a new ensemble.
                method = self.methods[attributes['method']]
                self.output[key] =\
                    HistogramAverage(method.numberOfBins, multiplicity)
                self.currentOutput = self.output[key]
                self.currentOutput.numberOfTrajectories =\
                    float(attributes['numberOfTrajectories'])
        elif name == 'statisticsFrames':
            for key in ('model', 'method'):
                if not key in attributes.keys():
                    self.errors +=\
                        'Missing ' + key + ' attribute in statisticsFrames.\n'
                    return
            key = (attributes['model'], attributes['method'])
            # An ensemble of trajectories should not be listed twice.
            if key in self.output:
                self.errors += 'Simulation output (' + key[0] + ', ' +\
                    key[1] + ') listed twice.\n'
            else:
                # Check that the method has been defined.
                if not attributes['method'] in self.methods:
                    self.errors += 'StatisticsFrames uses an undefined method.\n'
                    return
                # Start a new ensemble.
                method = self.methods[attributes['method']]
                self.output[key] = StatisticsFrames()
                self.currentOutput = self.output[key]
        elif name == 'statisticsAverage':
            for key in ('model', 'method'):
                if not key in attributes.keys():
                    self.errors +=\
                        'Missing ' + key + ' attribute in statisticsAverage.\n'
                    return
            key = (attributes['model'], attributes['method'])
            # An ensemble of trajectories should not be listed twice.
            if key in self.output:
                self.errors += 'Simulation output (' + key[0] + ', ' +\
                    key[1] + ') listed twice.\n'
            else:
                # Check that the method has been defined.
                if not attributes['method'] in self.methods:
                    self.errors += 'StatisticsAverage uses an undefined method.\n'
                    return
                # Start a new ensemble.
                method = self.methods[attributes['method']]
                self.output[key] = StatisticsAverage()
                self.currentOutput = self.output[key]
        elif name == 'frameTimes':
            self.frameTimes = []
            # The content should be empty.
            if self.content:
                self.errors += 'Mishandled content in frameTimes tag.\n'
        elif name == 'recordedSpecies':
            self.recordedSpecies = []
            # The content should be empty.
            if self.content:
                self.errors += 'Mishandled content in recordedSpecies tag.\n'
        elif name == 'recordedReactions':
            self.recordedReactions = []
            # The content should be empty.
            if self.content:
                self.errors += 'Mishandled content in recordedReactions tag.\n'
        elif name == 'populations':
            self.populations = []
            # The content should be empty.
            if self.content:
                self.errors += 'Mishandled content in population tag.\n'
        elif name == 'reactionCounts':
            self.reactionCounts = []
            # The content should be empty.
            if self.content:
                self.errors += 'Mishandled content in reactionCounts tag.\n'
        elif name == 'initialPopulations':
            self.initialPopulations = []
            # The content should be empty.
            if self.content:
                self.errors += 'Mishandled content in initialPopulations tag.\n'
        elif name == 'indices':
            self.indices = []
            # The content should be empty.
            if self.content:
                self.errors += 'Mishandled content in indices tag.\n'
        elif name == 'times':
            self.times = []
            # The content should be empty.
            if self.content:
                self.errors += 'Mishandled content in times tag.\n'
        elif name == 'histogram':
            # CONTINUE: Add 'cardinality', 'mean', 'summedSecondCenteredMoment',
            # and 'sumOfWeights' to the required attributes.
            for key in ('lowerBound', 'width', 'species'):
                if not key in attributes.keys():
                    self.errors +=\
                        'Missing ' + key + ' attribute in histogram.\n'
                    return
            species = int(attributes['species'])
            if self.currentOutput.__class__.__name__ == 'HistogramFrames':
                if not 'frame' in attributes.keys():
                    self.errors +=\
                        'Missing frame attribute in histogram.\n'
                    return
                frame = int(attributes['frame'])
                self.currentHistogram =\
                    self.currentOutput.histograms[frame][species]
            elif self.currentOutput.__class__.__name__ == 'HistogramAverage':
                self.currentHistogram =\
                    self.currentOutput.histograms[species]
            else:
                self.errors += 'Unkown histogram type.\n'
                return
            self.currentHistogramIndex = 0
            # CONTINUE Make these attributes required.
            if 'cardinality' in attributes:
                self.currentHistogram.cardinality =\
                    float(attributes['cardinality'])
            if 'mean' in attributes:
                self.currentHistogram.mean = float(attributes['mean'])
            if 'summedSecondCenteredMoment' in attributes:
                self.currentHistogram.summedSecondCenteredMoment =\
                    float(attributes['summedSecondCenteredMoment'])
            if 'sumOfWeights' in attributes:
                self.currentHistogram.sumOfWeights =\
                    float(attributes['sumOfWeights'])
            self.currentHistogram.lowerBound = float(attributes['lowerBound'])
            self.currentHistogram.setWidth(float(attributes['width']))
        elif name == 'statistics':
            self.statistics = []
            # The content should be empty.
            if self.content:
                self.errors += 'Mishandled content in statistics tag.\n'
        # CONTINUE: These are deprecated.
        elif name == 'firstHistogram':
            # The content should be empty.
            if self.content:
                self.errors += 'Mishandled content in firstHistogram tag.\n'
        elif name == 'secondHistogram':
            # The content should be empty.
            if self.content:
                self.errors += 'Mishandled content in secondHistogram tag.\n'
        elif name == 'histogramElement':
            # The content should be empty.
            if self.content:
                self.errors += 'Mishandled content in histogramElement tag.\n'
        #
        # Elements for the random number generator state.
        #
        elif name == 'random':
            if 'seed' in attributes.keys():
                self.seed = int(attributes['seed'])
        elif name == 'stateMT19937':
            self.listOfMt19937States.append([])
            # The content should be empty.
            if self.content:
                self.errors += 'Mishandled content in stateMT19937 tag.\n'
        #
        # Unknown tag.
        #
        else:
            self.errors += 'Unknown tag: ' + name + '\n'
            
    def endElement(self, name):
        del self.elements[-1]
        if name == 'model':
            # Add the current model.
            self.models[self.model.id] = self.model
        elif name == 'listOfParameters':
            # The list of parameters may be empty.
            pass
        elif name == 'listOfCompartments':
            # The list of compartments may be empty.
            pass
        elif name == 'listOfSpecies':
            if not (self.model.species and self.model.speciesIdentifiers):
                self.errors += 'No species were defined.\n'
        elif name == 'frameTimes':
            self.currentOutput.setFrameTimes\
                (map(float, self.content.split()))
            self.content = ''
        elif name == 'recordedSpecies':
            self.currentOutput.setRecordedSpecies\
                (map(int, self.content.split()))
            self.content = ''
        elif name == 'recordedReactions':
            self.currentOutput.recordedReactions =\
                map(int, self.content.split())
            self.content = ''
        elif name == 'histogram':
            self.currentHistogram = None
            self.currentHistogramIndex = None
        # CONTINUE: These are deprecated.
        elif name == 'firstHistogram':
            self.currentHistogram.set(0, map(float, self.content.split()))
            self.content = ''
        elif name == 'secondHistogram':
            self.currentHistogram.set(1, map(float, self.content.split()))
            self.content = ''
        elif name == 'histogramElement':
            self.currentHistogram.set(self.currentHistogramIndex,
                                      map(float, self.content.split()))
            self.currentHistogramIndex += 1
            self.content = ''
        elif name == 'populations':
            # Add the populations to the current set of trajectories.
            self.currentOutput.appendPopulations\
                (map(float, self.content.split()))
            self.content = ''
        elif name == 'reactionCounts':
            # Add the reaction counts the current set of trajectories.
            self.currentOutput.appendReactionCounts\
                (map(float, self.content.split()))
            self.content = ''
        elif name == 'initialPopulations':
            # Add the initial populations to the current set of trajectories.
            self.currentOutput.appendInitialPopulations\
                (map(float, self.content.split()))
            self.content = ''
        elif name == 'indices':
            # Add the reaction indices to the current set of trajectories.
            self.currentOutput.appendIndices(map(int, self.content.split()))
            self.content = ''
        elif name == 'times':
            # Add the reaction times to the current set of trajectories.
            self.currentOutput.appendTimes\
                (map(float, self.content.split()))
            self.content = ''
        elif name == 'statistics':
            self.currentOutput.setStatistics\
                (map(float, self.content.split()))
            self.content = ''
        elif name == 'stateMT19937':
            # Add the elements to the state.
            self.listOfMt19937States[-1].extend(map(int, self.content.split()))
            self.content = ''
            # 624 for the array, 1 for the position.
            if len(self.listOfMt19937States[-1]) != 625:
                self.errors += 'Bad Mersenne Twister state.\n'
        # CONTINUE: Should I check the populations and reaction counts?
                
    def characters(self, content):
        # CONTINUE: firstHistogram and secondHistogram are deprecated.
        if self.elements[-1] in\
                ('firstHistogram', 'secondHistogram', 'histogramElement',
                 'frameTimes', 'recordedSpecies', 'recordedReactions',
                 'populations', 'reactionCounts', 'initialPopulations',
                 'indices', 'times', 'statistics', 'stateMT19937'):
            self.content += content

def main():
    from xml.sax import parse
    from glob import glob

    for name in glob('../examples/cain/*.xml'):
        handler = ContentHandler()
        parse(open(name, 'r'), handler)
        assert not handler.errors

if __name__ == '__main__':
    main()
