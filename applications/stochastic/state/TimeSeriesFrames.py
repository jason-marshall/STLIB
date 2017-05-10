"""Implements the TimeSeriesFrames class."""

import numpy

from SimulationOutput import SimulationOutput, isSorted

class TimeSeriesFrames(SimulationOutput):
    """A trajectory records the species populations and the reaction counts."""
    
    def __init__(self, frameTimes=[], recordedSpecies=[], recordedReactions=[]):
        SimulationOutput.__init__(self, recordedSpecies, recordedReactions)
        # The frame times.
        self.setFrameTimes(frameTimes)
        # The list of population arrays.
        self.populations = []
        # The list of reaction count arrays.
        self.reactionCounts = []

    def setFrameTimes(self, frameTimes):
        self.frameTimes = numpy.array(frameTimes, numpy.float64)

    def setRecordedSpecies(self, recordedSpecies):
        self.recordedSpecies = recordedSpecies

    def appendPopulations(self, populations):
        x = numpy.array(populations, numpy.float64)
        x.shape = (len(self.frameTimes), -1)
        assert x.shape[1] == len(self.recordedSpecies)
        self.populations.append(x)

    def appendReactionCounts(self, reactionCounts):
        x = numpy.array(reactionCounts, numpy.float64)
        x.shape = (len(self.frameTimes), -1)
        assert x.shape[1] == len(self.recordedReactions)
        self.reactionCounts.append(x)
        
    def empty(self):
        """Return true if there are no trajectories."""
        return not self.populations

    def size(self):
        """Return the number of trajectories."""
        return len(self.populations)

    def hasErrors(self):
        """Return None if the trajectory is valid. Otherwise return an error
        message."""
        error = SimulationOutput.hasErrors(self)
        if error:
            return error
        if self.frameTimes is None or len(self.frameTimes) <= 0:
            return 'There are no frame times.'
        if not isSorted(self.frameTimes):
            return 'The frame times are not in order.'
        if len(self.populations) != len(self.reactionCounts):
            return 'The number of population and reaction count trajectories are not the same.'
        for x in self.populations:
            # Populations must be non-negative.
            flat = numpy.reshape(x, (-1,))
            if min(flat) < 0:
                return 'There are negative populations.'
        for x in self.reactionCounts:
            # Reaction counts must be non-negative.
            flat = numpy.reshape(x, (-1,))
            if min(flat) < 0:
                return 'There are negative reaction counts.'
            # Reaction counts must be non-decreasing.
            for i in range(x.shape[1]):
                if not isSorted(x[:,i]):
                    return 'There are decreasing reaction counts.'
        return None

    def writeXml(self, writer, model, method):
        writer.beginElement('timeSeriesFrames', {'model':model,
                                                 'method':method})
        writer.beginElement('frameTimes')
        writer.writeData(' '.join([repr(x) for x in self.frameTimes]))
        writer.endElement() # frameTimes
        writer.beginElement('recordedSpecies')
        writer.writeData(' '.join([repr(x) for x in self.recordedSpecies]))
        writer.endElement() # recordedSpecies
        writer.beginElement('recordedReactions')
        writer.writeData(' '.join([repr(x) for x in self.recordedReactions]))
        writer.endElement() # recordedSpecies
        for populations in self.populations:
            writer.beginElement('populations')
            flat = numpy.reshape(populations, (-1,))
            writer.writeData(' '.join([repr(x) for x in flat]))
            writer.endElement() # populations
        for reactionCounts in self.reactionCounts:
            writer.beginElement('reactionCounts')
            flat = numpy.reshape(reactionCounts, (-1,))
            writer.writeData(' '.join([repr(x) for x in flat]))
            writer.endElement() # reactionCounts
        writer.endElement() # timeSeriesFrames

def main():
    import sys
    sys.path.insert(1, '..')
    from io.XmlWriter import XmlWriter

    trajectory = TimeSeriesFrames([0, 1])
    trajectory.recordedSpecies = [0, 1]
    trajectory.recordedReactions = [0, 1]
    trajectory.appendPopulations([2, 3, 7, 11])
    trajectory.appendReactionCounts([13, 17, 19, 23])

    writer = XmlWriter()
    writer.beginDocument()
    trajectory.writeXml(writer, 'model', 'method')
    writer.endDocument()

if __name__ == '__main__':
    main()
