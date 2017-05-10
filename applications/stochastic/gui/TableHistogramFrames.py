"""Display histogram errors in a table."""

import wx
import math
from TableBase import TableBase

class TableError(TableBase):
    def __init__(self, parent, model, output):
        TableBase.__init__(self, parent)
        self.SetRowLabelSize(16*10)

        numberOfRows = len(output.frameTimes)
        self.resize(numberOfRows, len(output.recordedSpecies))
        self.setColumnLabels([model.speciesIdentifiers[_i] for _i in
                              output.recordedSpecies])
        row = 0
        # For each frame.
        for i in range(len(output.frameTimes)):
            # Frame time.
            self.SetRowLabelValue(row, str(output.frameTimes[i]))
            # For each recorded species.
            for j in range(len(output.recordedSpecies)):
                # Estimated error in the distribution.
                self.SetCellValue(row, j, '%g' %
                                  output.histograms[i][j].errorInDistribution())
            row += 1

class TableErrorPanel(wx.Panel):
    def __init__(self, parent, model, output):
        wx.Panel.__init__(self, parent)
        self.grid = TableError(self, model, output)
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.grid, 1, wx.EXPAND)
        self.SetSizer(sizer)
        self.Fit()

class TableErrorFrame(wx.Frame):
    def __init__(self, model, output, title='Error', parent=None):
        wx.Frame.__init__(self, parent, title=title, size=(600,600))
        display = TableErrorPanel(self, model, output)
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
    from state.HistogramFrames import HistogramFrames
    from state.Model import Model

    numberOfBins = 20
    multiplicity = 2
    recordedSpecies = [0]
    output = HistogramFrames(numberOfBins, multiplicity, recordedSpecies)
    output.setFrameTimes([1e20/9.])
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
    output.histograms[0][0].read(stream, multiplicity)

    app = wx.PySimpleApp()
    model = Model()
    model.speciesIdentifiers = ['X']
    TableErrorFrame(model, output).Show()
    app.MainLoop()

if __name__ == '__main__':
    main()

