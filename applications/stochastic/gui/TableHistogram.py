"""Display a histogram in a table."""

import wx
from TableBase import TableBase

class TableHistogram(TableBase):
    def __init__(self, parent, histogram):
        TableBase.__init__(self, parent)
        self.SetRowLabelSize(16*10)

        self.resize(len(histogram.histograms[0]), 1)
        self.setColumnLabels(['Probability'])
        p = histogram.getProbabilities()
        l = histogram.lowerBound
        w = histogram.getWidth()
        for i in range(len(p)):
            # Bin range.
            self.SetRowLabelValue(i, str(l+i*w))
            # Probability.
            self.SetCellValue(i, 0, '%g' % p[i])

class TableHistogramPanel(wx.Panel):
    def __init__(self, parent, histogram):
        wx.Panel.__init__(self, parent)
        self.grid = TableHistogram(self, histogram)
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.grid, 1, wx.EXPAND)
        self.SetSizer(sizer)
        self.Fit()

class TableHistogramFrame(wx.Frame):
    def __init__(self, histogram, title='Histogram', parent=None):
        wx.Frame.__init__(self, parent, title=title, size=(300,600))
        display = TableHistogramPanel(self, histogram)
        display.grid.AutoSize()
        self.Layout()

#
# Test Code.
#

def main():
    import math
    from StringIO import StringIO

    import sys
    sys.path.insert(1, '..')
    from state.Histogram import Histogram

    numberOfBins = 20
    multiplicity = 2
    histogram = Histogram(numberOfBins, multiplicity)
    # Poisson with mean 10. PMF = e^-lambda lambda^n / n!
    poisson = [math.exp(-10)]
    for n in range(1, numberOfBins):
        poisson.append(poisson[-1] * 10. / n)
    cardinality = numberOfBins
    sumOfWeights = sum(poisson)
    mean = 0.
    for i in range(numberOfBins):
        mean += poisson[i] * i
    mean /= sumOfWeights
    summedSecondCenteredMoment = 0.
    for i in range(numberOfBins):
        summedSecondCenteredMoment += poisson[i] * (i - mean)**2

    stream = StringIO(repr(cardinality) + '\n' + 
                      repr(sumOfWeights) + '\n' +
                      repr(mean) + '\n' + 
                      repr(summedSecondCenteredMoment) + '\n' + 
                      '0\n1\n' +
                      ''.join([repr(_x) + ' ' for _x in poisson]) + '\n' +
                      '0 ' * len(poisson) + '\n')
    histogram.read(stream, multiplicity)

    app = wx.PySimpleApp()
    TableHistogramFrame(histogram).Show()
    app.MainLoop()

if __name__ == '__main__':
    main()

