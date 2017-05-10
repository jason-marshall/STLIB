"""Records sets of histograms."""

import numpy

from Histogram import Histogram
from SimulationOutput import SimulationOutput, isSorted

class HistogramFrames(SimulationOutput):
    """A set of histograms."""
    
    def __init__(self, numberOfBins, multiplicity, recordedSpecies=[]):
        """Construct an empty data structure."""
        SimulationOutput.__init__(self, recordedSpecies, [])
        self.numberOfBins = numberOfBins
        self.multiplicity = multiplicity
        self.numberOfTrajectories = 0
        self.frameTimes = None
        self.histograms = None

    def setFrameTimes(self, frameTimes):
        self.frameTimes = numpy.array(frameTimes, numpy.float64)
        # If both the frame times and the recorded species have been defined
        # initialize the data structure.
        if self.recordedSpecies:
            self.initialize()

    def setRecordedSpecies(self, recordedSpecies):
        self.recordedSpecies = recordedSpecies
        # If both the frame times and the recorded species have been defined
        # initialize the data structure.
        if self.frameTimes is not None:
            self.initialize()

    def initialize(self):
        """Construct histograms for each frame and recorded species."""
        assert self.frameTimes is not None and self.recordedSpecies
        self.histograms = []
        for t in self.frameTimes:
            f = []
            for s in self.recordedSpecies:
                f.append(Histogram(self.numberOfBins, self.multiplicity))
            self.histograms.append(f)

    def setCurrentToMinimum(self):
        """Set the current histogram to the one with the minimum sum."""
        index = self.histograms[0][0].findMinimum()
        for frame in self.histograms:
            for histogram in frame:
                histogram.setCurrent(index)

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
        if self.frameTimes is None or len(self.frameTimes) <= 0:
            return 'There are no frame times.'
        if not isSorted(self.frameTimes):
            return 'The frame times are not in order.'
        if not len(self.histograms) == len(self.frameTimes):
            return 'The list of frames is incomplete.'
        for x in self.histograms:
            if not len(x) == len(self.recordedSpecies):
                return 'The list of histograms is incomplete.'
        # CONTINUE: Check the histograms.
        return None

    def writeXml(self, writer, model, method):
        writer.beginElement('histogramFrames',
                            {'model':model, 'method':method,
                             'multiplicity':str(self.multiplicity),
                             'numberOfTrajectories':
                                 repr(self.numberOfTrajectories)})
        writer.beginElement('frameTimes')
        writer.writeData(' '.join([repr(x) for x in self.frameTimes]))
        writer.endElement() # frameTimes
        writer.beginElement('recordedSpecies')
        writer.writeData(' '.join([repr(x) for x in self.recordedSpecies]))
        writer.endElement() # recordedSpecies
        for frame in range(len(self.histograms)):
            for species in range(len(self.histograms[frame])):
                self.histograms[frame][species].writeXml(writer, frame, species)
        writer.endElement() # histogramFrames
