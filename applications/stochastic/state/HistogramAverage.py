"""Records sets of histograms that hold average species populations."""

import numpy

from Histogram import Histogram
from SimulationOutput import SimulationOutput

class HistogramAverage(SimulationOutput):
    """A set of histograms."""
    
    def __init__(self, numberOfBins, multiplicity, recordedSpecies=[]):
        """Construct an empty data structure."""
        SimulationOutput.__init__(self, recordedSpecies, [])
        self.numberOfBins = numberOfBins
        self.multiplicity = multiplicity
        self.numberOfTrajectories = 0
        self.setRecordedSpecies(recordedSpecies)

    def setRecordedSpecies(self, recordedSpecies):
        """Construct histograms for each recorded species."""
        self.recordedSpecies = recordedSpecies
        self.histograms = []
        for s in self.recordedSpecies:
            self.histograms.append(Histogram(self.numberOfBins,
                                             self.multiplicity))

    def setCurrentToMinimum(self):
        """Set the current histogram to the one with the minimum sum."""
        index = self.histograms[0].findMinimum()
        for h in self.histograms:
            h.setCurrent(index)

    def empty(self):
        """Return true if there are no trajectories."""
        return self.numberOfTrajectories == 0

    def size(self):
        """Return the number of trajectories."""
        return self.numberOfTrajectories

    def hasErrors(self):
        """Return None if data structure is valid. Otherwise return an error
        message."""
        error = SimulationOutput.hasErrors(self)
        if error:
            return error
        if len(self.histograms) != len(self.recordedSpecies):
            return 'The list of histograms is incomplete.'
        # CONTINUE: Check the histograms.
        return None

    def writeXml(self, writer, model, method):
        writer.beginElement('histogramAverage',
                            {'model':model, 'method':method,
                             'multiplicity':
                                 str(self.histograms[0].multiplicity()),
                             'numberOfTrajectories':
                                 repr(self.numberOfTrajectories)})
        writer.beginElement('recordedSpecies')
        writer.writeData(' '.join([repr(x) for x in self.recordedSpecies]))
        writer.endElement() # recordedSpecies
        for i in range(len(self.histograms)):
            self.histograms[i].writeXml(writer, species=i)
        writer.endElement() # histogramAverage

