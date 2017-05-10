"""Records the time averaged mean and standard deviation for solutions."""

import numpy

from SimulationOutput import SimulationOutput

class StatisticsAverage(SimulationOutput):
    """The time-averaged mean and standard deviation of the species populations.
    This class is used for representing solutions that are generated outside of
    Cain. The solution may be exact or emperical. We do not record the
    number of trajectories used to generate the solution. In Cain, the
    solution is treated as a reference whether it is exact or approximate."""
    
    def __init__(self, recordedSpecies=[]):
        """Construct an empty data structure."""
        # No recorded reactions.
        SimulationOutput.__init__(self, recordedSpecies, [])
        # The list of (mean, standard deviation) tuples.
        # self.statistics[species] gives the tuple for the specified species
        # index.
        self.statistics = None

    def setRecordedSpecies(self, recordedSpecies):
        self.recordedSpecies = recordedSpecies

    def setStatistics(self, data):
        """Use the flat list of mean/standard deviation pairs to set the
        statistics data."""
        n = len(self.recordedSpecies)
        assert len(data) == 2 * n
        self.statistics = [(data[2*i], data[2*i+1]) for i in range(n)]

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
        if self.statistics is None or \
               len(self.statistics) != len(self.recordedSpecies):
            return 'The list of statistics is incomplete.'
        return None

    def writeXml(self, writer, model, method):
        writer.beginElement('statisticsAverage',
                            {'model':model, 'method':method})
        writer.beginElement('recordedSpecies')
        writer.writeData(' '.join([repr(x) for x in self.recordedSpecies]))
        writer.endElement()
        writer.beginElement('statistics')
        writer.writeData(' '.join([repr(x[0]) + ' ' + repr(x[1])
                                   for x in self.statistics]))
        writer.endElement()
        writer.endElement() # statisticsAverage

def main():
    import sys
    sys.path.insert(1, '..')
    from io.XmlWriter import XmlWriter

    recordedSpecies = [0, 1, 2]
    x = StatisticsAverage(recordedSpecies)
    x.setStatistics([1, 2, 3, 4, 5, 6])
    writer = XmlWriter()
    writer.beginDocument()
    x.writeXml(writer, 'model', 'method')
    writer.endDocument()

if __name__ == '__main__':
    main()
