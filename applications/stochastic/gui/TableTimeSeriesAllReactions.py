"""Display all-reaction trajectories in a table."""

# If we are running the unit tests.
if __name__ == '__main__':
    import sys
    sys.path.insert(1, '..')

import wx
import numpy
from state.TrajectoryCalculator import TrajectoryCalculator
from TableBase import TableBase

class TableEnd(TableBase):
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
    def __init__(self, parent, identifiers, listOfTimes, listOfArrays):
        TableBase.__init__(self, parent)

        numberOfRows = sum([len(x) for x in listOfTimes])
        self.resize(numberOfRows, 2 + len(identifiers))
        self.setColumnLabels(['Trajectory', 'Time'] + identifiers)
            
        row = 0
        # For each trajectory.
        for n in range(len(listOfArrays)):
            times = listOfTimes[n]
            array = listOfArrays[n]
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
                for x in array[frame]:
                    self.SetCellValue(row, col, '%g' % x)
                    col += 1
                row += 1

class TablePanel(wx.Panel):
    def __init__(self, parent, model, trajectories, target):
        wx.Panel.__init__(self, parent)
        tc = TrajectoryCalculator(model)
        reactionIdentifiers = [r.id for r in model.reactions]
        if target == 'PopulationsEnd':
            self.grid = TableEnd(self, model.speciesIdentifiers,
                                  [tc.computeFinalPopulations(trajectories, i)
                                   for i in range(len(trajectories.indices))])
        elif target == 'ReactionsEnd':
            self.grid = TableEnd(self, reactionIdentifiers,
                                  [tc.computeFinalReactionCounts(trajectories, i)
                                   for i in range(len(trajectories.indices))])
        elif target == 'PopulationsAll':
            listOfTimes = []
            listOfPopulations = []
            for i in range(len(trajectories.indices)):
                # Include the start time, but not the end time.
                times, populations, reactionCounts =\
                    tc.makeFramesAtReactionEvents(trajectories, i, True, False)
                listOfTimes.append(times)
                listOfPopulations.append(populations)
            self.grid = TableAll(self, model.speciesIdentifiers, listOfTimes,
                                 listOfPopulations)
        elif target == 'ReactionsAll':
            listOfTimes = []
            listOfReactionCounts = []
            for i in range(len(trajectories.indices)):
                # Include the start time, but not the end time.
                times, populations, reactionCounts =\
                    tc.makeFramesAtReactionEvents(trajectories, i, True, False)
                listOfTimes.append(times)
                listOfReactionCounts.append(reactionCounts)
            self.grid = TableAll(self, reactionIdentifiers,
                                 listOfTimes, listOfReactionCounts)
        else:
            assert False
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.grid, 1, wx.EXPAND)
        self.SetSizer(sizer)
        self.Fit()

class TableFrame(wx.Frame):
    def __init__(self, model, trajectories, target,
                 title='Trajectories', parent=None):
        wx.Frame.__init__(self, parent, title=title, size=(900,600))
        display = TablePanel(self, model, trajectories, target)
        display.grid.AutoSize()
        self.Layout()

#
# Test Code.
#

def main():
    from state.Model import Model
    from state.Species import Species
    from state.SpeciesReference import SpeciesReference
    from state.Reaction import Reaction
    from state.TimeSeriesAllReactions import TimeSeriesAllReactions

    app = wx.PySimpleApp()

    # Define a simple model.
    model = Model()
    model.id = 'model'
    model.speciesIdentifiers.append('s1')
    model.species['s1'] = Species('C1', 'species 1', '13')
    model.speciesIdentifiers.append('s2')
    model.species['s2'] = Species('C1', 'species 2', '17')
    # s1 -> s2
    model.reactions.append(
        Reaction('r1', 'reaction 1', [SpeciesReference('s1')], 
                 [SpeciesReference('s2')], True, '1.5'))
    # s1 + s2 -> 2 s1
    model.reactions.append(
        Reaction('r2', 'reaction 2', 
                 [SpeciesReference('s1'), SpeciesReference('s2')], 
                 [SpeciesReference('s1', 2)], True, '2.5'))

    # Trajectories.
    initialTime = 0.
    finalTime = 3.
    # One species, two reactions.
    t = TimeSeriesAllReactions([0], [0, 1], initialTime, finalTime)
    t.appendInitialPopulations([13, 17])
    t.appendIndices([0, 1, 0])
    t.appendTimes([0, 0.5, 1])
    t.appendInitialPopulations([13, 17])
    t.appendIndices([0, 1, 1, 0])
    t.appendTimes([0, 0.33, 0.67, 1])

    TableFrame(model, t, 'PopulationsEnd', 
               'Populations at the end time.').Show()
    TableFrame(model, t, 'ReactionsEnd', 
               'Reaction counts at the end time.').Show()
    TableFrame(model, t, 'PopulationsAll', 
               'Populations at the start time and reaction events.').Show()
    TableFrame(model, t, 'ReactionsAll', 
               'Reaction counts at the start time and reaction events.').Show()

    app.MainLoop()

if __name__ == '__main__':
    main()
