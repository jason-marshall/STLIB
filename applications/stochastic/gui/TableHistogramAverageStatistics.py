"""Display population statistics in a table."""

import wx
import math
from TableBase import TableBase

def flatten(theList):
    """Flatten the list."""
    result = []
    for x in theList:
        if hasattr(x, '__iter__') and not isinstance(x, basestring):
            result.extend(flatten(x))
        else:
            result.append(x)
    return result

class Grid(TableBase):
    def __init__(self, parent, model, output):
        TableBase.__init__(self, parent)
        identifiers = [model.speciesIdentifiers[_i] for _i in
                       output.recordedSpecies]
        # Allow for a space before and after the longest species identifier.
        length = max([len(_x) for _x in identifiers])
        self.SetRowLabelSize(10 * (length + 2))

        # A row for each recorded species. The first column is the mean, the 
        # second is the standard deviation.
        self.resize(len(output.recordedSpecies), 2)
        self.setColumnLabels([u'\u03bc', u'\u03c3'])
        # For each recorded species.
        for row in range(len(output.recordedSpecies)):
            # Species identifier.
            self.SetRowLabelValue(row, identifiers[row])
            h = output.histograms[row]
            self.SetCellValue(row, 0, '%g' % h.getMean())
            if h.isVarianceDefined():
                self.SetCellValue(row, 1,
                                  '%g' % math.sqrt(h.getUnbiasedVariance()))
            else:
                self.SetCellValue(row, 1, '-')
                             
class Panel(wx.Panel):
    def __init__(self, parent, model, output):
        wx.Panel.__init__(self, parent)
        self.grid = Grid(self, model, output)
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.grid, 1, wx.EXPAND)
        self.SetSizer(sizer)
        self.Fit()

class TableHistogramAverageStatistics(wx.Frame):
    def __init__(self, model, output, title='Statistics', parent=None):
        wx.Frame.__init__(self, parent, title=title, size=(600,600))
        display = Panel(self, model, output)
        display.grid.AutoSize()
        self.Layout()

#
# Test Code.
#

if __name__ == '__main__':
    from StringIO import StringIO

    import sys
    sys.path.insert(1, '..')
    from state.HistogramAverage import HistogramAverage
    from state.Model import Model

    numberOfBins = 20
    multiplicity = 2
    recordedSpecies = [0]
    output = HistogramAverage(numberOfBins, multiplicity, recordedSpecies)
    # Poisson with mean 10. PMF = e^-lambda lambda^n / n!
    poisson = [math.exp(-10)]
    for n in range(1,numberOfBins):
        poisson.append(poisson[-1] * 10. / n)
    cardinality = len(poisson)
    sumOfWeights = 1
    mean = 0.
    for i in range(len(poisson)):
        mean += i * poisson[i]
    summedSecondCenteredMoment = 0.
    for i in range(len(poisson)):
        summedSecondCenteredMoment += poisson[i] * (i - mean)**2
    lowerBound = 0
    width = 1
    stream = StringIO('%r\n' * 6 % (cardinality, sumOfWeights, mean,
                                    summedSecondCenteredMoment, lowerBound,
                                    width) +
                      ''.join([str(_x) + ' ' for _x in poisson]) + '\n' +
                      '0 ' * len(poisson) + '\n')
    output.histograms[0].read(stream, multiplicity)

    app = wx.PySimpleApp()
    model = Model()
    model.speciesIdentifiers = ['X']
    TableHistogramAverageStatistics(model, output).Show()
    app.MainLoop()

