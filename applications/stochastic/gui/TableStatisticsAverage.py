"""Display population statistics in a table."""

import wx
import math
from TableBase import TableBase

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
            self.SetCellValue(row, 0, '%g' % output.statistics[row][0])
            self.SetCellValue(row, 1,
                              '%g' % math.sqrt(output.statistics[row][1]))
                             
class Panel(wx.Panel):
    def __init__(self, parent, model, output):
        wx.Panel.__init__(self, parent)
        self.grid = Grid(self, model, output)
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.grid, 1, wx.EXPAND)
        self.SetSizer(sizer)
        self.Fit()

class TableStatisticsAverage(wx.Frame):
    def __init__(self, model, output, title='Statistics', parent=None):
        wx.Frame.__init__(self, parent, title=title, size=(300,600))
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
    from state.StatisticsAverage import StatisticsAverage
    from state.Model import Model

    recordedSpecies = [0]
    output = StatisticsAverage(recordedSpecies)
    output.setStatistics([1., 2.])

    app = wx.PySimpleApp()
    model = Model()
    model.speciesIdentifiers = ['X']
    TableStatisticsAverage(model, output).Show()
    app.MainLoop()

