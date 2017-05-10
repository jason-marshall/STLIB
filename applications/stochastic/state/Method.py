"""Implements the Method class."""

# If we are running the unit tests.
if __name__ == '__main__':
    import sys
    sys.path.insert(1, '..')

from io.XmlWriter import XmlWriter
import simulationMethods

class Method:
    """Simulation method."""
    
    def __init__(self):
        """Make a simulation. Some data is invalid and must be set later.

        - id: A unique identifier for the method.
        - timeDependence: The index of the time dependence category.
        - category: The index of the simulation category.
        - method: The index of the simulation method.
        - options: The index of the simulation options.
        - startTime: The start time for the simulation.
        - equilibrationTime: The length of time to simulate before recording.
        - recordingTime: The length of time to simulate and record.
        - maximumSteps: The maximum number of allowed steps in generating a
        trajectory.
        - numberOfFrames: If there is only one frame, it is taken at the end 
        time.
        - numberOfBins: The number bins in the histograms.
        - multiplicity: The histogram multiplicity.
        - solverParameter: A solver parameter.
        - seed: The seed used to generate the Mersenne twister state vector."""
        self.id = ''
        self.timeDependence = 0
        self.category = 0
        self.method = 0
        self.options = 0
        self.startTime = 0
        self.equilibrationTime = 0
        self.recordingTime = 1
        self.maximumSteps = None
        self.numberOfFrames = 11
        self.numberOfBins = 32
        self.multiplicity = 4
        self.solverParameter = None
        self.seed = 0

    def hasErrors(self):
        """Return None if the simulation parameters are valid. Otherwise 
        return an error message."""
        # Check the identifier.
        if not self.id:
            return 'Bad identifier.'
        # Check the category.
        if not self.timeDependence in\
                range(simulationMethods.numberOfTimeDependence):
            return 'Bad time dependence category.'
        # Check the category.
        if not self.category in range(simulationMethods.numberOfCategories\
                                          [self.timeDependence]):
            return 'Bad category.'
        # Check the method.
        if not self.method in \
                range(simulationMethods.numberOfMethods[self.timeDependence]\
                          [self.category]):
            return 'Bad method.'
        # Check the options.
        if not self.options in \
                range(simulationMethods.numberOfOptions[self.timeDependence]\
                          [self.category][self.method]):
            return 'Bad options.'
        # Check the times.
        if float(self.startTime) != self.startTime:
            return 'Bad start time.'
        if float(self.equilibrationTime) != self.equilibrationTime:
            return 'Bad equilibration time.'
        if float(self.recordingTime) != self.recordingTime:
            return 'Bad recording time.'
        if not 0 <= self.equilibrationTime:
            return 'Bad equilibration time.'
        if not 0 < self.recordingTime:
            return 'Bad recording time.'
        # Check the maximum steps.
        if self.maximumSteps != None:
            if float(self.maximumSteps) != self.maximumSteps or \
                    self.maximumSteps <= 0:
                return 'Bad value for the maximum steps.'
        # Check the number of frames.
        if int(self.numberOfFrames) != self.numberOfFrames:
            return 'The number of frames is not an integer.'
        if not self.numberOfFrames > 0:
            return 'Bad number of frames.'
        # Check the number of bins.
        if int(self.numberOfBins) != self.numberOfBins:
            return 'The number of bins is not an integer.'
        if not self.numberOfBins > 0:
            return 'Bad number of bins.'
        # Check the multiplicity.
        if int(self.multiplicity) != self.multiplicity:
            return 'The histogram multiplicity is not an integer.'
        if not self.multiplicity > 0:
            return 'Bad histogram multiplicity.'
        # Check the parameter.
        parameterName = simulationMethods.parameterNames1[self.timeDependence]\
            [self.category][self.method][self.options]
        if simulationMethods.parameterValues1[self.timeDependence]\
                [self.category][self.method][self.options] and\
                self.solverParameter is None:
            return 'The solver parameter ' + parameterName + ' is undefined.'
        if self.solverParameter is not None and float(self.solverParameter) != self.solverParameter:
            return 'The method parameter ' + parameterName +\
                ' must be a floating point number.'
        return None

    def isDiscrete(self):
        """Return True if the simulation method is discrete. (That is, the 
        species amounts are integers.)"""
        return simulationMethods.isDiscrete(self.timeDependence, self.category)

    def writeXml(self, writer):
        assert not self.hasErrors()
        attributes = {'id': self.id,
                      'timeDependence': str(self.timeDependence),
                      'category': str(self.category),
                      'method': str(self.method),
                      'options': str(self.options),
                      'startTime': str(self.startTime),
                      'equilibrationTime': str(self.equilibrationTime),
                      'recordingTime': str(self.recordingTime),
                      'numberOfFrames': str(self.numberOfFrames),
                      'numberOfBins': str(self.numberOfBins),
                      'multiplicity': str(self.multiplicity)}
        if self.maximumSteps is not None:
            attributes['maximumSteps'] = str(self.maximumSteps)
        if self.solverParameter is not None:
            attributes['solverParameter'] = str(self.solverParameter)
        writer.writeEmptyElement('method', attributes)

    def readXml(self, attributes):
        """Read from an attributes dictionary. Return any errors encountered."""
        errors = ''
        # The attribute "dictionary" may not work as it should. In particular
        # the test "x in attributes" may not work. Thus we need to directly 
        # use attributes.keys().
        keys = attributes.keys()
        if not 'id' in keys:
            errors += 'Missing id attribute in method.\n'
            # Fatal error.
            return errors
        self.id = attributes['id']
        # Time dependence.
        if 'timeDependence' in keys:
            timeDependence = int(attributes['timeDependence'])
            if timeDependence in\
                    range(simulationMethods.numberOfTimeDependence):
                self.timeDependence = timeDependence
            else:
                errors += 'Bad time dependence value in method.\n'
        else:
            # The first time dependence category is the default.
            self.timeDependence = 0
        # Category.
        if 'category' in keys:
            category = int(attributes['category'])
            if category in range(simulationMethods.numberOfCategories\
                                     [self.timeDependence]):
                self.category = category
            else:
                errors += 'Bad category value in method.\n'
        else:
            # The first category is the default.
            self.category = 0
        # Method.
        if 'method' in keys:
            method = int(attributes['method'])
            if method in range(simulationMethods.numberOfMethods\
                                   [self.timeDependence][self.category]):
                self.method = method
            else:
                errors += 'Bad method value in method.\n'
        else:
            # The first method is the default.
            self.method = 0
        # Options.
        if 'options' in keys:
            options = int(attributes['options'])
            if options in \
                    range(simulationMethods.numberOfOptions\
                              [self.timeDependence][self.category][self.method]):
                self.options = options
            else:
                errors += 'Bad options value in method.\n'
        else:
            # The first options are the default.
            self.options = 0
        if 'startTime' in keys:
            self.startTime = float(attributes['startTime'])
        else:
            self.startTime = 0.
        if 'equilibrationTime' in keys:
            self.equilibrationTime = float(attributes['equilibrationTime'])
        else:
            self.equilibrationTime = 0.
        if 'recordingTime' in keys:
            self.recordingTime = float(attributes['recordingTime'])
        else:
            # CONTINUE: This is an old attribute from verson 0.12.
            if 'endTime' in keys:
                self.recordingTime = float(attributes['endTime'])
            else:
                self.recordingTime = 1.
        if 'maximumSteps' in keys:
            self.maximumSteps = float(attributes['maximumSteps'])
        if 'numberOfFrames' in keys:
            self.numberOfFrames = int(attributes['numberOfFrames'])
        if 'numberOfBins' in keys:
            self.numberOfBins = int(attributes['numberOfBins'])
        if 'multiplicity' in keys:
            self.multiplicity = int(attributes['multiplicity'])
        else:
            self.multiplicity = 4
        if 'solverParameter' in keys:
            self.solverParameter = float(attributes['solverParameter'])
        else:
            value = simulationMethods.parameterValues1[self.timeDependence]\
                [self.category][self.method][self.options]
            # If the solver parameter is required.
            if value:
                # Set it to the default value.
                self.solverParameter = float(value)
        return errors

def writeMethodXml(simulation, out=None):
    if out:
        writer = XmlWriter(out)
    else:
        writer = XmlWriter()
    writer.beginDocument()
    simulation.writeXml(writer)
    writer.endDocument()
