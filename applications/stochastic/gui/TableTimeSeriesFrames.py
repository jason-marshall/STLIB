"""Display simulation output in a table."""

# If we are running the unit tests.
if __name__ == '__main__':
    import sys
    sys.path.insert(1, '..')

import wx
import numpy
from statistics import meanStdDev
from TableBase import TableBase

class TableLast(TableBase):
    def __init__(self, parent, identifiers, listOfArrays):
        TableBase.__init__(self, parent)

        self.resize(len(listOfArrays), len(identifiers))
        self.setColumnLabels(identifiers)
            
        for row in range(len(listOfArrays)):
            # Row number.
            self.SetRowLabelValue(row, str(row + 1))
            col = 0
            # Data.
            for x in listOfArrays[row]:
                self.SetCellValue(row, col, '%g' % x)
                col += 1

class TableAll(TableBase):
    def __init__(self, parent, identifiers, times, listOfArrays):
        TableBase.__init__(self, parent)

        self.resize(len(times) * len(listOfArrays),
                    2 + len(identifiers))
        self.setColumnLabels(['Trajectory', 'Time'] + identifiers)
            
        row = 0
        # For each trajectory.
        for n in range(len(listOfArrays)):
            for frame in range(len(times)):
                self.SetRowLabelValue(row, str(row + 1))
                col = 0
                # Trajectory.
                self.SetCellValue(row, col, str(n + 1))
                col += 1
                # Time.
                self.SetCellValue(row, col, '%g' % times[frame])
                col += 1
                # Data.
                for x in listOfArrays[n][frame]:
                    self.SetCellValue(row, col, '%g' % x)
                    col += 1
                row += 1

class TableBinned(TableBase):
    def __init__(self, parent, identifiers, times, listOfArrays):
        TableBase.__init__(self, parent)

        assert len(times) > 1
        self.resize((len(times) - 1) * len(listOfArrays), 2 + len(identifiers))
        self.setColumnLabels(['Trajectory', 'Interval'] + identifiers)
            
        row = 0
        # For each trajectory.
        for n in range(len(listOfArrays)):
            for frame in range(len(times) - 1):
                self.SetRowLabelValue(row, str(row + 1))
                col = 0
                # Trajectory.
                self.SetCellValue(row, col, str(n + 1))
                col += 1
                # Interval.
                self.SetCellValue(row, col, '%g to %g' % 
                                  (times[frame], times[frame + 1]))
                col += 1
                # Data.
                a = listOfArrays[n]
                for i in range(len(identifiers)):
                    self.SetCellValue(row, col, '%g' % (a[frame + 1][i] - 
                                                        a[frame][i]))
                    col += 1
                row += 1

def statisticsLabels(identifiers):
    labels = []
    for id in identifiers:
        labels.append('m(' + id + ')')
        labels.append('s(' + id + ')')
    return labels

class TableStatisticsLast(TableBase):
    def __init__(self, parent, identifiers, listOfArrays):
        TableBase.__init__(self, parent)

        self.resize(1, 2 * len(identifiers))
        self.setColumnLabels(statisticsLabels(identifiers))

        mean, stdDev = meanStdDev(listOfArrays)

        row = 0
        self.SetRowLabelValue(row, str(row + 1))
        col = 0
        for i in range(len(mean)):
            self.SetCellValue(row, col, '%g' % mean[i])
            col += 1
            self.SetCellValue(row, col, '%g' % stdDev[i])
            col += 1

class TableStatisticsAll(TableBase):
    def __init__(self, parent, identifiers, times, listOfArrays):
        TableBase.__init__(self, parent)

        self.resize(len(times), 1 + 2 * len(identifiers))
        self.setColumnLabels(['Time'] + statisticsLabels(identifiers))
            
        mean, stdDev = meanStdDev(listOfArrays)

        row = 0
        for frame in range(len(times)):
            self.SetRowLabelValue(row, str(row + 1))
            col = 0
            # Time.
            self.SetCellValue(row, col, '%g' % times[frame])
            col += 1
            for i in range(len(mean[frame])):
                self.SetCellValue(row, col, '%g' % mean[frame][i])
                col += 1
                self.SetCellValue(row, col, '%g' % stdDev[frame][i])
                col += 1
            row += 1

class TableStatisticsBinned(TableBase):
    def __init__(self, parent, identifiers, times, listOfArrays):
        TableBase.__init__(self, parent)

        assert len(times) > 1
        self.resize(len(times) - 1, 1 + 2 * len(identifiers))
        self.setColumnLabels(['Interval'] + statisticsLabels(identifiers))
            
        mean, stdDev = meanStdDev([x[1:] - x[:-1] for x in listOfArrays])

        row = 0
        for frame in range(len(times) - 1):
            self.SetRowLabelValue(row, str(row + 1))
            col = 0
            # Interval.
            self.SetCellValue(row, col, '%g to %g' % 
                              (times[frame], times[frame + 1]))
            col += 1
            for i in range(len(mean[frame])):
                self.SetCellValue(row, col, '%g' % mean[frame][i])
                col += 1
                self.SetCellValue(row, col, '%g' % stdDev[frame][i])
                col += 1
            row += 1

class TablePanel(wx.Panel):
    def __init__(self, parent, model, trajectories, target):
        wx.Panel.__init__(self, parent)
        if target == 'PopulationsLast':
            self.grid = TableLast(self, model.speciesIdentifiers,
                                  [x[-1] for x in trajectories.populations])
        elif target == 'ReactionsLast':
            self.grid = TableLast(self, model.getReactionIdentifiers(),
                                  [x[-1] for x in trajectories.reactionCounts])
        elif target == 'PopulationsAll':
            self.grid = TableAll(self, model.speciesIdentifiers,
                                 trajectories.frameTimes,
                                 [x for x in trajectories.populations])
        elif target == 'ReactionsAll':
            self.grid = TableAll(self, model.getReactionIdentifiers(),
                                 trajectories.frameTimes,
                                 [x for x in trajectories.reactionCounts])
        elif target == 'ReactionsBinned':
            self.grid = TableBinned(self, model.getReactionIdentifiers(),
                                    trajectories.frameTimes,
                                    [x for x in trajectories.reactionCounts])
        elif target == 'PopulationStatisticsLast':
            self.grid = \
                TableStatisticsLast(self, model.speciesIdentifiers,
                                    [x[-1] for x in trajectories.populations])
        elif target == 'ReactionStatisticsLast':
            self.grid = TableStatisticsLast\
                (self, model.getReactionIdentifiers(),
                 [x[-1] for x in trajectories.reactionCounts])
        elif target == 'PopulationStatisticsAll':
            self.grid = \
                TableStatisticsAll(self, model.speciesIdentifiers,
                                   trajectories.frameTimes,
                                   [x for x in trajectories.populations])
        elif target == 'ReactionStatisticsAll':
            self.grid = TableStatisticsAll\
                (self, model.getReactionIdentifiers(),
                 trajectories.frameTimes,
                 [x for x in trajectories.reactionCounts])
        elif target == 'ReactionStatisticsBinned':
            self.grid = TableStatisticsBinned\
                (self, model.getReactionIdentifiers(),
                 trajectories.frameTimes,
                 [x for x in trajectories.reactionCounts])
        else:
            assert False
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.grid, 1, wx.EXPAND)
        self.SetSizer(sizer)
        self.Fit()

class TableFrame(wx.Frame):
    def __init__(self, model, trajectories, target, title='Trajectories',
                 parent=None):
        wx.Frame.__init__(self, parent, title=title, size=(900,600))
        display = TablePanel(self, model, trajectories, target)
        display.grid.AutoSize()
        self.Layout()

#
# Test Code.
#

class TestModel:
    def __init__(self, speciesIdentifiers, reactionIdentifiers):
        self.speciesIdentifiers = speciesIdentifiers
        self.reactionIdentifiers = reactionIdentifiers

    def getReactionIdentifiers(self):
        return self.reactionIdentifiers

def main():
    from state.TimeSeriesFrames import TimeSeriesFrames

    app = wx.PySimpleApp()

    t = TimeSeriesFrames([0, 0.5, 1], [0], [0])
    t.appendPopulations([1, 2, 3])
    t.appendReactionCounts([0, 2, 4])
    t.appendPopulations([2, 3, 5])
    t.appendReactionCounts([0, 3, 6])

    model = TestModel(['s1'], ['r1'])
    TableFrame(model, t, 'PopulationsLast', 
               'Populations from the last frame of each trajectory.').Show()
    TableFrame(model, t, 'ReactionsLast', 
               'Reaction counts from the last frame of each trajectory.').Show()
    TableFrame(model, t, 'PopulationsAll', 
               'Populations from each frame of each trajectory.').Show()
    TableFrame(model, t, 'ReactionsAll', 
               'Reaction counts from each frame of each trajectory.').Show()
    TableFrame(model, t, 'ReactionsBinned', 
               'Binned reaction counts between each frame of each trajectory.').Show()

    TableFrame(model, t, 'PopulationStatisticsLast', 
               'Populations statistics for the last frame.').Show()
    TableFrame(model, t, 'ReactionStatisticsLast',
               'Cumulative reaction count statistics.').Show()
    TableFrame(model, t, 'PopulationStatisticsAll', 
               'Populations statistics for each frame.').Show()
    TableFrame(model, t, 'ReactionStatisticsAll',
               'Cumulative reaction count statistics for each frame.').Show()
    TableFrame(model, t, 'ReactionStatisticsBinned',
               'Binned reaction count statistics.').Show()

    app.MainLoop()

if __name__ == '__main__':
    main()
