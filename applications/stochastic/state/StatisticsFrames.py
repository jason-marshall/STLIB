"""Records the mean and standard deviation for solutions."""

import numpy

from SimulationOutput import SimulationOutput, isSorted

class StatisticsFrames(SimulationOutput):
    """The mean and standard deviation of the species populations at each frame.
    This class is used for representing solutions that are generated outside of
    Cain. The solution may be exact or emperical. We do not record the
    number of trajectories used to generate the solution. In Cain, the
    solution is treated as a reference whether it is exact or approximate."""
    
    def __init__(self, recordedSpecies=[]):
        """Construct an empty data structure."""
        # No recorded reactions.
        SimulationOutput.__init__(self, recordedSpecies, [])
        # The list of frame times.
        self.frameTimes = None
        # The 2-D list of (mean, standard deviation) tuples.
        # self.statistics[frame][species] gives the tuple for the specified
        # frame and species indices.
        self.statistics = None

    def setFrameTimes(self, frameTimes):
        self.frameTimes = numpy.array(frameTimes, numpy.float64)

    def setRecordedSpecies(self, recordedSpecies):
        self.recordedSpecies = recordedSpecies

    def setStatistics(self, data):
        """Use the flat list of mean/standard deviation pairs to set the
        statistics data."""
        f = len(self.frameTimes)
        n = len(self.recordedSpecies)
        assert len(data) == 2 * f * n
        self.statistics = [[(data[r*2*n+2*i], data[r*2*n+2*i+1]) for i in
                            range(n)] for r in range(f)]

    def size(self):
        """The number of trajectories is not tracked. Thus return the
        string 'n. a.' for not applicable."""
        return 'n. a.'

    def empty(self):
        """Return true if the statistics have not been set."""
        return not bool(self.statistics)

    def hasErrors(self):
        """Return None if data structure is valid. Otherwise return an error
        message."""
        error = SimulationOutput.hasErrors(self)
        if error:
            return error
        if self.frameTimes is None or len(self.frameTimes) <= 0:
            return 'There are no frame times.'
        if not isSorted(self.frameTimes):
            return 'The frame times are not in order.'
        if not len(self.statistics) == len(self.frameTimes):
            return 'The list of frames is incomplete.'
        for x in self.statistics:
            if not len(x) == len(self.recordedSpecies):
                return 'The list of statistics is incomplete.'
        return None

    def writeXml(self, writer, model, method):
        writer.beginElement('statisticsFrames',
                            {'model':model, 'method':method})
        writer.beginElement('frameTimes')
        writer.writeData(' '.join([repr(x) for x in self.frameTimes]))
        writer.endElement()
        writer.beginElement('recordedSpecies')
        writer.writeData(' '.join([repr(x) for x in self.recordedSpecies]))
        writer.endElement()
        writer.beginElement('statistics')
        writer.writeData(' '.join([' '.join([repr(x[0]) + ' ' + repr(x[1])
                                             for x in f]) for f in
                                   self.statistics]))
        writer.endElement()
        writer.endElement() # statisticsFrames

def main():
    import sys
    sys.path.insert(1, '..')
    from io.XmlWriter import XmlWriter

    frameTimes = [0, 1]
    recordedSpecies = [0, 1, 2]
    x = StatisticsFrames(recordedSpecies)
    x.setFrameTimes(frameTimes)
    x.setStatistics([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

    writer = XmlWriter()
    writer.beginDocument()
    x.writeXml(writer, 'model', 'method')
    writer.endDocument()

if __name__ == '__main__':
    main()
