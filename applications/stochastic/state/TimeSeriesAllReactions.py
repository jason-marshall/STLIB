"""Implements the TimeSeriesAllReactions class."""

import numpy

from SimulationOutput import SimulationOutput

def isSorted(x):
    if len(x) == 0:
        return True
    for i in range(len(x) - 1):
        if x[i] > x[i + 1]:
            return False
    return True

class TimeSeriesAllReactions(SimulationOutput):
    """Record the reaction indices and times."""
    
    def __init__(self, recordedSpecies, recordedReactions, initialTime,
                 finalTime):
        """The recorded species and reactions should be all of the species
        and reactions."""
        SimulationOutput.__init__(self, recordedSpecies, recordedReactions)
        # The initial time.
        self.initialTime = initialTime
        # The final time.
        self.finalTime = finalTime
        # The initial species populations.
        self.initialPopulations = []
        # The list of reaction index arrays.
        self.indices = []
        # The list of reaction time arrays.
        self.times = []

    def setRecordedSpecies(self, recordedSpecies):
        self.recordedSpecies = recordedSpecies

    def appendInitialPopulations(self, initialPopulations):
        self.initialPopulations.append(numpy.array(initialPopulations,
                                                   numpy.float64))

    def appendIndices(self, indices):
        self.indices.append(numpy.array(indices, numpy.int32))

    def appendTimes(self, times):
        self.times.append(numpy.array(times, numpy.float64))

    def empty(self):
        """Return true if there are no trajectories."""
        return not self.indices

    def size(self):
        """Return the number of trajectories."""
        return len(self.indices)

    def hasErrors(self):
        """Return None if the trajectory is valid. Otherwise return an error
        message."""
        error = SimulationOutput.hasErrors(self)
        if error:
            return error
        numberOfReactions = len(self.recordedReactions)
        if numberOfReactions == 0:
            return 'There are no reactions.'
        if not (self.initialTime < self.finalTime):
            return 'Invalid time interval: [' + str(self.initialTime) +\
                ' .. ' + str(self.finalTime) + '].'
        if len(self.initialPopulations) != len(self.indices):
            return 'The number of initial populations and reation indices does not match.'
        if len(self.initialPopulations) != len(self.times):
            return 'The number of initial populations and reation times does not match.'
        for i in range(len(self.initialPopulations)):
            initialPopulations = self.initialPopulations[i]
            indices = self.indices[i]
            times = self.times[i]
            if min(initialPopulations) < 0:
                return 'The initial populations must be non-negative.'
            if len(indices) != len(times):
                return 'The number of reaction indices does not match the number of reaction times.'
            if min(indices) < 0 or max(indices) >= numberOfReactions:
                return 'Invalid reaction index.'
            if min(times) < self.initialTime or max(times) > self.finalTime:
                return 'Reaction time is outside the simulation time interval.'
            if not isSorted(times):
                return 'The reaction times are not ordered.'
        return None

    def writeXml(self, writer, model, method):
        writer.beginElement('timeSeriesAllReactions',
                            {'model':model, 'method':method,
                             'initialTime':repr(self.initialTime),
                             'finalTime':repr(self.finalTime)})
        for initialPopulations in self.initialPopulations:
            writer.beginElement('initialPopulations')
            writer.writeData(' '.join([repr(x) for x in initialPopulations]))
            writer.endElement() # initialPopulations
        for indices in self.indices:
            writer.beginElement('indices')
            writer.writeData(' '.join([repr(x) for x in indices]))
            writer.endElement() # indices
        for times in self.times:
            writer.beginElement('times')
            writer.writeData(' '.join([repr(x) for x in times]))
            writer.endElement() # times
        writer.endElement() # timeSeriesAllReactions

def main():
    import sys
    sys.path.insert(1, '..')
    from io.XmlWriter import XmlWriter

    # One species, two reactions.
    trajectory = TimeSeriesAllReactions([0], [0, 1], 1., 5.)
    trajectory.appendInitialPopulations([7.])
    trajectory.appendIndices([0, 1, 1, 0])
    trajectory.appendTimes([1, 2, 3, 4])

    writer = XmlWriter()
    writer.beginDocument()
    trajectory.writeXml(writer, 'model', 'method')
    writer.endDocument()

if __name__ == '__main__':
    main()
