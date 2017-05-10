"""P-value for the null hypothesis that the means are equal."""

# If we are running the unit tests.
if __name__ == '__main__':
    import sys
    sys.path.insert(1, '..')

import wx
import wx.grid
import numpy
import scipy.stats
from math import sqrt

from pylab import figure, plot, title, xlabel, ylabel

def studentTTest(m1, s1, n1, m2):
    """Arguments:
    m denotes the mean.
    s denotes the standard deviation.
    n denotes the cardinality."""
    # If the cardinalities are not greater than unity, the variance is not
    # defined.
    assert n1 > 1
    # Check the case that the standard deviation is zero.
    if s1 == 0:
        if m1 == m2:
            return 1.
        else:
            return 0.
    # t-statistic.
    t = - abs((m1 - m2)) * sqrt(n1) / s1
    # Degrees of freedom.
    df = n1 - 1
    # 2-sided test.
    return 2 * scipy.stats.t.cdf(t, df)

def welchTTest(m1, s1, n1, m2, s2, n2):
    """Arguments:
    m denotes the mean.
    s denotes the standard deviation.
    n denotes the cardinality."""
    # If the cardinalities are not greater than unity, the variance is not
    # defined.
    assert n1 > 1 and n2 > 1
    # Check the cases that one or more standard deviation is zero.
    if s1 == 0 or s2 == 0:
        if m1 == m2:
            return 1.
        else:
            return 0.
    # Weighted variance.
    wv = s1 * s1 / n1 + s2 * s2 / n2
    # t-statistic.
    t = - abs((m1 - m2)) / sqrt(wv)
    # Degrees of freedom.
    df = wv * wv / (s1**4 / (n1*n1*(n1-1)) + s2**4 / (n2*n2*(n2-1)))
    # 2-sided test.
    return 2 * scipy.stats.t.cdf(t, df)

def statistics(x):
    """Return a tuple of the mean, standard deviation, and cardinality."""
    if type(x) is type(()):
        return (x[0], x[1], float('inf'))
    elif type(x) is type([]):
        if len(x) > 1:
            return (numpy.mean(x), sqrt(numpy.var(x)), len(x))
    else:
        assert x.__class__.__name__ == 'Histogram'
        if x.isVarianceDefined():
            return (x.mean, sqrt(x.getUnbiasedVariance()), x.cardinality)
    return None

def oneSampleTTest(x, y):
    s1 = statistics(x)
    if s1:
        if type(y) is type(()):
            return studentTTest(s1[0], s1[1], s1[2], y[0])
        elif y.__class__.__name__ == 'Histogram':
            return studentTTest(s1[0], s1[1], s1[2], y.mean)
        else:
            assert False
    else:
        return 0.

def twoSampleTTest(x, y):
    s1 = statistics(x)
    s2 = statistics(y)
    if s1 and s2:
        return welchTTest(s1[0], s1[1], s1[2], s2[0], s2[1], s2[2])
    else:
        return 0.

def pValue(x1, r1, x2, r2):
    assert not (r1 and r2)
    if r1:
        return oneSampleTTest(x2, x1)
    elif r2:
        return oneSampleTTest(x1, x2)
    else:
        return twoSampleTTest(x1, x2)

class Selection(wx.Panel):
    def __init__(self, parent, state):
        wx.Panel.__init__(self, parent, -1)
        self.state = state
        self.outputKeys = []

        sizer = wx.BoxSizer(wx.VERTICAL)
        self.output = wx.Choice(self, choices=[])
        self.Bind(wx.EVT_CHOICE, self.onOutput, self.output)
        sizer.Add(self.output, 1, wx.EXPAND | wx.ALL, 5)
        self.species = wx.Choice(self, choices=[])
        sizer.Add(self.species, 1, wx.EXPAND | wx.ALL, 5)
        self.frame = wx.Choice(self, choices=[])
        sizer.Add(self.frame, 1, wx.EXPAND | wx.ALL, 5)
        self.SetSizer(sizer)

        self.refresh()

    def onOutput(self, event):
        self.update()
        event.Skip()

    def update(self):
        index = self.output.GetSelection()
        if index == wx.NOT_FOUND:
            return
        # Check that the simulation output has not disappeared.
        if not self.outputKeys[index] in self.state.output:
            self.refresh()
            return
        output = self.state.output[self.outputKeys[index]]
        modelId = self.outputKeys[index][0]
        model = self.state.models[modelId]

        # The species choice.
        selection = self.species.GetSelection()
        self.species.Clear()
        self.species.Append('All species')
        for i in output.recordedSpecies:
            self.species.Append(model.speciesIdentifiers[i])
        if selection != wx.NOT_FOUND and selection < self.species.GetCount():
            self.species.SetSelection(selection)
        else:
            self.species.SetSelection(0)

        # The frame choice.
        selection = self.frame.GetSelection()
        self.frame.Clear()
        if output.__class__.__name__ in ('HistogramFrames', 'TimeSeriesFrames',
                                         'StatisticsFrames'):
            self.frame.Append('All frames')
            for time in output.frameTimes:
                self.frame.Append(str(time))
            self.frame.Enable()
            if selection != wx.NOT_FOUND and selection < self.frame.GetCount():
                self.frame.SetSelection(selection)
            else:
                self.frame.SetSelection(0)
        else:
            self.frame.Disable()

    def getSelections(self):
        """Return a tuple of the selection indices."""
        return (self.output.GetSelection(), self.species.GetSelection(),
                self.frame.GetSelection())

    def getOutput(self):
        """Return a tuple of the following:
        - The list of selected species.
        - The list of selected frame times. The empty string indicates a
        steady state solution instead of a frame.
        - The list of selected output.
        - A boolean value that indicates if the solution is to be used as
        a reference. Currently, steady state solutions are used as a reference
        solution, because I don't know how to define the number of degrees of
        freedom."""
        index = self.output.GetSelection()
        if index == wx.NOT_FOUND:
            return None, None, None, None
        # Check that the simulation output has not disappeared.
        if not self.outputKeys[index] in self.state.output:
            self.refresh()
            return None, None, None, None
        data = self.state.output[self.outputKeys[index]]

        s = self.species.GetSelection()
        if s == wx.NOT_FOUND:
            return None, None, None, None
        if s == 0:
            species = [self.species.GetString(n) for n in
                       range(1, self.species.GetCount())]
            speciesIndices = range(self.species.GetCount() - 1)
        else:
            species = [self.species.GetString(s)]
            speciesIndices = [s-1]

        # First check the *Average cases because they do not use frames.
        if data.__class__.__name__ == 'HistogramAverage':
            return species, [''], [[data.histograms[s]]], True
        if data.__class__.__name__ == 'StatisticsAverage':
            return species, [''], [[data.statistics[s]]], True

        # Then deal with output that has frames.
        f = self.frame.GetSelection()
        if f == wx.NOT_FOUND:
            return None, None, None, None
        if f == 0:
            frames = [self.frame.GetString(n) for n in
                       range(1,self.frame.GetCount())]
            frameIndices = range(self.frame.GetCount() - 1)
        else:
            frames = [self.frame.GetString(f)]
            frameIndices = [f-1]

        if data.__class__.__name__ == 'TimeSeriesFrames':
            output = [[[x[i, j] for x in data.populations] for i in
                       frameIndices] for j in speciesIndices]
            isReference = False
        elif data.__class__.__name__ == 'HistogramFrames':
            output = [[data.histograms[i][j] for i in frameIndices] for j in
                      speciesIndices]
            isReference = False
        elif data.__class__.__name__ == 'StatisticsFrames':
            output = [[data.statistics[i][j] for i in frameIndices] for j in
                      speciesIndices]
            isReference = True
        else:
            assert(False)
        return species, frames, output, isReference

    def refresh(self):
        # Get the appropriate outputs.
        self.outputKeys = []
        for key in self.state.output:
            if self.state.output[key].__class__.__name__ in\
                    ('TimeSeriesFrames', 'HistogramFrames', 'HistogramAverage',
                     'StatisticsFrames', 'StatisticsAverage'):
                self.outputKeys.append(key)
        outputChoices = [x[0] + ', ' + x[1] for x in self.outputKeys]
        selection = self.output.GetSelection()
        self.output.Clear()
        for choice in outputChoices:
            self.output.Append(choice)
        # Set the selection.
        if selection != wx.NOT_FOUND and selection < self.output.GetCount():
            self.output.SetSelection(selection)
        else:
            self.output.SetSelection(0)
        # Updated the species and frame for this output.
        self.update()

class PValueMean(wx.Frame):
    def __init__(self, state, title, parent=None):
        wx.Frame.__init__(self, parent, -1, title, size=(600,600))
        self.state = state

        # Selections.
        selectionsSizer = wx.BoxSizer(wx.HORIZONTAL)
        self.selections = [Selection(self, state), Selection(self, state)]
        for s in self.selections:
            selectionsSizer.Add(s, 1, wx.EXPAND | wx.ALL, 5)
        sizer = wx.BoxSizer(wx.VERTICAL)
        # Don't expand in the vertical direction.
        sizer.Add(selectionsSizer, 0, wx.EXPAND | wx.ALIGN_TOP, 5)

        # Calculate and plot.
        buttonsSizer = wx.BoxSizer(wx.HORIZONTAL)
        b = wx.Button(self, -1, 'Calculate')
        self.Bind(wx.EVT_BUTTON, self.onCalculate, b)
        buttonsSizer.Add(b, 0)
        b = wx.Button(self, -1, 'Plot')
        self.Bind(wx.EVT_BUTTON, self.onPlot, b)
        buttonsSizer.Add(b, 0)
        sizer.Add(buttonsSizer, 0, wx.ALL, 5)

        # Grid.
        self.grid = wx.grid.Grid(self)
        self.grid.CreateGrid(0, 0)
        self.grid.SetRowLabelSize(12*12)
        sizer.Add(self.grid, 1, wx.EXPAND)

        self.SetSizer(sizer)
        # Intercept the close event.
        self.Bind(wx.EVT_CLOSE, self.onClose)

    def onClose(self, event):
        # If there is a parent, it stores a dictionary of these frames.
        if self.GetParent():
            del self.GetParent().children[self.GetId()]
        self.Destroy()

    def refresh(self):
        # CONTINUE: Store the current selections.
        for s in self.selections:
            s.refresh()

    def onCalculate(self, event):
        # Check that they are not trying to compare a selection with itself.
        if self.selections[0].getSelections() ==\
               self.selections[1].getSelections():
            wx.MessageBox('The two selections must be distinct.',
                          'Error!', style=wx.OK|wx.ICON_EXCLAMATION)
            return

        s1, f1, o1, r1 = self.selections[0].getOutput()
        s2, f2, o2, r2 = self.selections[1].getOutput()
        if not (s1 and s2):
            wx.MessageBox('The two selections are invalid.',
                          'Error!', style=wx.OK|wx.ICON_EXCLAMATION)
            return

        # Both selections may not be reference solutions.
        if r1 and r2:
            wx.MessageBox('One cannot calculate p-values for two reference solutions.',
                          'Error!', style=wx.OK|wx.ICON_EXCLAMATION)
            return
            
        # Check for incompatible lengths.
        if min(len(s1), len(s2)) != 1 and len(s1) != len(s2):
            wx.MessageBox('The first selection has %s species while the other '\
                          'has %s.\nThe lengths are not compatible.' %
                          (len(s1), len(s2)),
                          'Error!', style=wx.OK|wx.ICON_EXCLAMATION)
            return
        if min(len(f1), len(f2)) != 1 and len(f1) != len(f2):
            wx.MessageBox('The first selection has %s frames while the other '\
                          'has %s.\nThe lengths are not compatible.' %
                          (len(f1), len(f2)),
                          'Error!', style=wx.OK|wx.ICON_EXCLAMATION)
            return

        # Make the list of column (species) index pairs.
        if len(s1) == 1:
            cols = [(0, i) for i in range(len(s2))]
        elif len(s2) == 1:
            cols = [(i, 0) for i in range(len(s1))]
        else:
            assert len(s1) == len(s2)
            cols = [(i, i) for i in range(len(s1))]
        # Set the number of columns.
        if len(cols) > self.grid.GetNumberCols():
            self.grid.AppendCols(len(cols) - self.grid.GetNumberCols())
        elif self.grid.GetNumberCols() > len(cols):
            self.grid.DeleteCols(0, self.grid.GetNumberCols() - len(cols))
        # Set the column labels.
        if len(s1) == len(s2) and all([s1[i] == s2[i] for i in range(len(s1))]):
            for i in range(len(cols)):
                self.grid.SetColLabelValue(i, s1[cols[i][0]])
                self.grid.SetColSize(i, 12*12)
        else:
            for i in range(len(cols)):
                self.grid.SetColLabelValue(i, s1[cols[i][0]] + ', ' +
                                           s2[cols[i][1]])
                self.grid.SetColSize(i, 12*12)
            
        # Make the list of row (frame) index pairs.
        if len(f1) == 1:
            rows = [(0, i) for i in range(len(f2))]
        elif len(f2) == 1:
            rows = [(i, 0) for i in range(len(f1))]
        else:
            assert len(f1) == len(f2)
            rows = [(i, i) for i in range(len(f1))]
        # Set the number of rows.
        if len(rows) > self.grid.GetNumberRows():
            self.grid.AppendRows(len(rows) - self.grid.GetNumberRows())
        elif self.grid.GetNumberRows() > len(rows):
            self.grid.DeleteRows(0, self.grid.GetNumberRows() - len(rows))
        # Set the row labels.
        if len(f1) == len(f2) and all([f1[i] == f2[i] for i in range(len(f1))]):
            for i in range(len(rows)):
                self.grid.SetRowLabelValue(i, f1[rows[i][0]])
        else:
            for i in range(len(rows)):
                self.grid.SetRowLabelValue(i, f1[rows[i][0]] + ', ' +
                                           f2[rows[i][1]])

        # Calculate the p-values.
        for j in range(len(cols)):
            for i in range(len(rows)):
                a = o1[cols[j][0]][rows[i][0]]
                b = o2[cols[j][1]][rows[i][1]]
                self.grid.SetCellValue(i, j, str(pValue(a, r1, b, r2)))
                self.grid.SetReadOnly(i, j)
            
    def onPlot(self, event):
        """Plot the columns of the grid."""
        if self.grid.GetNumberCols() == 0 or self.grid.GetNumberRows() == 0:
            wx.MessageBox('The grid is empty.',
                          'Error!', style=wx.OK|wx.ICON_EXCLAMATION)
            return
        for j in range(self.grid.GetNumberCols()):
            y = [float(self.grid.GetCellValue(i, j)) for i in
                 range(self.grid.GetNumberRows())]
            figure()
            plot(y)
            title(self.grid.GetColLabelValue(j))
            xlabel('Frame Number')
            ylabel('P-value')
            
def main():
    import sys
    sys.path.insert(1, '..')
    from random import uniform

    from state.Model import Model
    from state.Histogram import Histogram
    from state.HistogramFrames import HistogramFrames

    # A histogram.
    numberOfBins = 4
    multiplicity = 2

    # Simulation output.
    frameTimes = [0, 1]
    recordedSpecies = [0, 1, 2]
    hf = HistogramFrames(numberOfBins, multiplicity, recordedSpecies)
    hf.setFrameTimes(frameTimes)
    for i in range(len(frameTimes)):
        for j in range(len(recordedSpecies)):
            h = Histogram(numberOfBins, multiplicity)
            h.setCurrentToMinimum()
            for b in range(numberOfBins):
                h.accumulate(b, uniform(0., 1.))
            hf.histograms[i][j].merge(h)

    # The model.
    model = Model()
    model.speciesIdentifiers = ['s1', 's2', 's3']

    # The state.
    class TestState:
        pass
    state = TestState()
    state.models = {}
    state.models['model'] = model
    state.output = {}
    state.output[('model', 'method')] = hf
    
    app = wx.PySimpleApp()
    PValueMean(state, 'P-value for equal means').Show()
    app.MainLoop()

if __name__ == '__main__':
    main()
