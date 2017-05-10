"""Implements exporting of time series data."""

# If we are running the unit tests.
import sys
if __name__ == '__main__':
    sys.path.insert(1, '..')

import wx
import os.path
import csv
from statistics import mean, meanStdDev
from ExportTimeSeriesGrid import ExportTimeSeriesGrid
from state.TrajectoryCalculator import TrajectoryCalculator
from messages import openWrite

delimiterChoices = ['comma', 'space', 'tab']
delimiterValues = [',', ' ', '\t']

class Configuration(wx.Panel):
    """Pick the species or reactions to export. If a trajectory calculator is
    specified, then the trajectories record every reaction event. Otherwise
    the trajectories record the state at frames."""
    
    def __init__(self, parent, state):
        wx.Panel.__init__(self, parent, -1)
        self.parent = parent
        self.state = state
        self.outputKeys = []

        sizer = wx.BoxSizer(wx.VERTICAL)

        # Output choice.
        self.outputChoice = wx.Choice(self, size=(400,-1), choices=[])
        self.Bind(wx.EVT_CHOICE, self.onOutput, self.outputChoice)
        sizer.Add(self.outputChoice, 0, wx.EXPAND, 5)

        # Species, cumulative reactions, or binned reactions.
        horizontal = wx.BoxSizer(wx.HORIZONTAL)
        # Species.
        self.species = wx.RadioButton(self, -1, 'Species', style=wx.RB_GROUP)
        self.species.SetValue(True)
        self.Bind(wx.EVT_RADIOBUTTON, self.onSpecies, self.species)
        horizontal.Add(self.species, 0, wx.ALL, 5)
        # Cumulative reactions.
        self.cumulativeReactions = wx.RadioButton(self, -1,
                                                  'Cumulative Reactions')
        self.Bind(wx.EVT_RADIOBUTTON, self.onCumulativeReactions,
                  self.cumulativeReactions)
        horizontal.Add(self.cumulativeReactions, 0, wx.ALL, 5)
        # Binned reactions.
        self.binnedReactions = wx.RadioButton(self, -1,
                                              'Binned Reactions')
        self.Bind(wx.EVT_RADIOBUTTON, self.onBinnedReactions,
                  self.binnedReactions)
        horizontal.Add(self.binnedReactions, 0, wx.ALL, 5)
        # Events.
        self.events = wx.RadioButton(self, -1, 'Events')
        self.Bind(wx.EVT_RADIOBUTTON, self.onEvents, self.events)
        horizontal.Add(self.events, 0, wx.ALL, 5)
        sizer.Add(horizontal, 0, wx.ALL, 5)

        # Trajectories or statistics.
        horizontal = wx.BoxSizer(wx.HORIZONTAL)
        self.trajectories = wx.RadioButton(self, -1, 'Trajectories',
                                           style=wx.RB_GROUP)
        self.trajectories.SetValue(True)
        self.Bind(wx.EVT_RADIOBUTTON, self.onTrajectories, self.trajectories)
        horizontal.Add(self.trajectories, 0, wx.ALL, 5)
        self.statistics = wx.RadioButton(self, -1, 'Statistics')
        self.Bind(wx.EVT_RADIOBUTTON, self.onStatistics, self.statistics)
        horizontal.Add(self.statistics, 0, wx.ALL, 5)
        sizer.Add(horizontal, 0, wx.ALL, 5)

        # The grid of species or reactions.
        self.grid = ExportTimeSeriesGrid(self)
        sizer.Add(self.grid, 1, wx.EXPAND)
        sizer.Add(wx.StaticLine(self), 0, wx.EXPAND|wx.ALL, 5)

        # Delimiter.
        box = wx.BoxSizer(wx.HORIZONTAL)
        box.Add(wx.StaticText(self, -1, 'Delimiter '), 0,
                wx.ALIGN_CENTER_VERTICAL)
        self.delimiter = wx.Choice(self, choices=delimiterChoices)
        self.delimiter.SetSelection(0)
        box.Add(self.delimiter, 0, wx.ALIGN_CENTER_VERTICAL)
        sizer.Add(box, 0, wx.ALIGN_TOP, 5)

        # Export button.
        buttons = wx.BoxSizer(wx.HORIZONTAL)
        b = wx.Button(self, -1, 'Export')
        self.Bind(wx.EVT_BUTTON, self.onExport, b)
        buttons.Add(b, 0, wx.ALIGN_RIGHT, 5)
        sizer.Add(buttons, 0, wx.ALIGN_RIGHT | wx.ALIGN_TOP, 5)

        self.SetSizer(sizer)
        self.refresh()
        self.Layout()
        self.Fit()

    def onOutput(self, event):
        self.update()
        event.Skip()

    def onTrajectories(self, event):
        self.grid.hideStdDev()
        event.Skip()

    def onStatistics(self, event):
        self.grid.showStdDev()
        event.Skip()

    def onSpecies(self, event):
        self.update()
        event.Skip()

    def onCumulativeReactions(self, event):
        self.update()
        event.Skip()

    def onBinnedReactions(self, event):
        self.update()
        event.Skip()

    def onEvents(self, event):
        self.update()
        event.Skip()

    def update(self):
        """Update the window for a new output selection. This is called when
        the user selects a new output. It is also called through refresh() 
        when the list of outputs changes."""
        index = self.outputChoice.GetSelection()
        if index == wx.NOT_FOUND:
            # Clear the grid.
            self.grid.setIdentifiers([])
            return
        # Check that the simulation output has not disappeared.
        if not self.outputKeys[index] in self.state.output:
            self.refresh()
            return

        # Update the radio buttons.
        output = self.state.output[self.outputKeys[index]]
        name = output.__class__.__name__
        if name == 'TimeSeriesFrames':
            # You can't export all reaction events for frame style 
            # trajectories.
            if self.events.GetValue():
                self.species.SetValue(True)
            self.events.Disable()
            self.binnedReactions.Enable()
            self.statistics.Enable()
        else:
            # You can't export binned reaction counts for all-reaction style 
            # trajectories.
            if self.binnedReactions.GetValue():
                self.species.SetValue(True)
            self.binnedReactions.Disable()
            self.events.Enable()
            # You can only export trajectories, not statistics, for all-reaction
            # stlye trajectories.
            self.trajectories.SetValue(True)
            self.statistics.Disable()
        # Update the grid.
        modelId = self.outputKeys[index][0]
        model = self.state.models[modelId]
        if self.species.GetValue():
            identifiers = [model.speciesIdentifiers[_i]
                           for _i in output.recordedSpecies]
        else:
            identifiers = [model.reactions[_i].id
                           for _i in output.recordedReactions]
        self.grid.setIdentifiers(identifiers)

        # Show or hide the standard deviation field.
        if self.trajectories.GetValue():
            self.grid.hideStdDev()
        else:
            self.grid.showStdDev()


    def refresh(self):
        """This is called when the list of outputs changes in the
        application."""
        # Get the time series outputs.
        self.outputKeys = []
        for key in self.state.output:
            if self.state.output[key].__class__.__name__ in\
                    ('TimeSeriesFrames', 'TimeSeriesAllReactions'):
                self.outputKeys.append(key)
        outputChoices = [x[0] + ', ' + x[1] for x in self.outputKeys]
        selection = self.outputChoice.GetSelection()
        self.outputChoice.Clear()
        for choice in outputChoices:
            self.outputChoice.Append(choice)
        # Set the selection.
        if selection != wx.NOT_FOUND and\
                selection < self.outputChoice.GetCount():
            self.outputChoice.SetSelection(selection)
        else:
            self.outputChoice.SetSelection(0)
        # Updated the species and frame for this output.
        self.update()

    def onExport(self, event):
        self.export()

    def export(self):
        index = self.outputChoice.GetSelection()
        if index == wx.NOT_FOUND:
            wx.MessageBox('There is no time series simulation output.',
                          'Error!', style=wx.OK|wx.ICON_EXCLAMATION)
            return
        # Save any values being edited in the grid.
        self.grid.saveEditControlValue()
        # The items to export.
        selected = self.grid.getCheckedItems()
        # Check that at least one row has been selected.
        if not selected:
            wx.MessageBox('No rows are selected.', 'Error.')
            return

        # Get the output file.
        delimiter = delimiterValues[self.delimiter.GetSelection()]
        if delimiter == ',':
            wildcard = 'CSV files (*.csv)|*.csv|'
            extension = '.csv'
        else:
            wildcard = 'Text files (*.txt)|*.txt|'
            extension = '.txt'
        wildcard += 'All files (*.*)|*.*'
        dialog = wx.FileDialog(self, 'Save as...', os.getcwd(),
                               style=wx.SAVE|wx.OVERWRITE_PROMPT,
                               wildcard=wildcard)
        if dialog.ShowModal() == wx.ID_OK:
            os.chdir(dialog.GetDirectory())
            filename = dialog.GetPath()
            if not os.path.splitext(filename)[1]:
                filename = filename + extension
            outputFile = openWrite(filename)
            if not outputFile:
                return
        dialog.Destroy()

        modelId, methodId = self.outputKeys[index]
        model = self.state.models[modelId]
        output = self.state.output[self.outputKeys[index]]
        # Construct the CSV writer.
        writer = csv.writer(outputFile, delimiter=delimiter)

        # Choose between the different kinds of output.
        name = output.__class__.__name__
        if name == 'TimeSeriesFrames':
            if self.trajectories.GetValue():
                if self.species.GetValue():
                    self.framesSpeciesTraj(model, output, selected,
                                                     writer)
                elif self.cumulativeReactions.GetValue():
                    self.framesCumulativeTraj(model, output,
                                                        selected, writer)
                else:
                    self.framesBinnedTraj(model, output, selected,
                                                    writer)
            else:
                if self.species.GetValue():
                    self.framesSpeciesStat(model, output, selected,
                                                     writer)
                elif self.cumulativeReactions.GetValue():
                    self.framesCumulativeStat(model, output,
                                                        selected, writer)
                else:
                    self.framesBinnedStat(model, output, selected,
                                                    writer)
        elif name == 'TimeSeriesAllReactions':
            if self.species.GetValue():
                self.allReactionsSpecies(model, output, selected,
                                                   writer)
            elif self.cumulativeReactions.GetValue():
                self.allReactionsCumulative(model, output, selected,
                                                      writer)
            else:
                self.allReactionsEvents(model, output, selected,
                                                  writer)
        else:
            assert False

    def framesSpeciesTraj(self, model, output, selected, writer):
        # The headers.
        headers = ['Time'] +\
                  [model.speciesIdentifiers[output.recordedSpecies[i]] for i
                   in selected]
        for i in range(len(output.populations)):
            # Write the headers.
            writer.writerow(['Trajectory', i])
            writer.writerow(headers)
            # Write each data row.
            for j in range(len(output.frameTimes)):
                writer.writerow([output.frameTimes[j]] +
                                [output.populations[i][j][s] for s in selected])
            # Blank line between trajectories.
            writer.writerow([])

    def framesSpeciesStat(self, model, output, selected, writer):
        # Whether to output the std. dev.
        stdDev = [self.grid.GetCellValue(i, 1) for i in selected]
        # The headers.
        headers = ['Time']
        for i in selected:
            identifier = model.speciesIdentifiers[output.recordedSpecies[i]]
            headers.append('m(%s)' % identifier)
            if stdDev[i]:
                headers.append('s(%s)' % identifier)
        writer.writerow(headers)
        # Write each data row.
        for f in range(len(output.frameTimes)):
            row = [output.frameTimes[f]]
            for s in selected:
                data = [x[f, s] for x in output.populations]
                # If the standard deviation box is checked.
                if stdDev[s]:
                    row.extend(list(meanStdDev(data)))
                else:
                    row.append(mean(data))
            writer.writerow(row)

    def framesCumulativeTraj(self, model, output, selected, writer):
        # The headers.
        headers = ['Time'] +\
                  [model.reactions[output.recordedReactions[i]].id for i
                   in selected]
        for i in range(len(output.populations)):
            # Write the headers.
            writer.writerow(['Trajectory', i])
            writer.writerow(headers)
            # Write each data row.
            for j in range(len(output.frameTimes)):
                writer.writerow([output.frameTimes[j]] +
                                [output.reactionCounts[i][j][s] for s in
                                 selected])
            # Blank line between trajectories.
            writer.writerow([])

    def framesCumulativeStat(self, model, output, selected, writer):
        # Whether to output the std. dev.
        stdDev = [self.grid.GetCellValue(i, 1) for i in selected]
        # The headers.
        headers = ['Time']
        for i in selected:
            identifier = model.reactions[output.recordedReactions[i]].id
            headers.append('m(%s)' % identifier)
            if stdDev[i]:
                headers.append('s(%s)' % identifier)
        writer.writerow(headers)
        # Write each data row.
        for f in range(len(output.frameTimes)):
            row = [output.frameTimes[f]]
            for s in selected:
                data = [x[f, s] for x in output.reactionCounts]
                # If the standard deviation box is checked.
                if stdDev[s]:
                    row.extend(list(meanStdDev(data)))
                else:
                    row.append(mean(data))
            writer.writerow(row)

    def framesBinnedTraj(self, model, output, selected, writer):
        # The headers.
        headers = ['Time'] +\
                  [model.reactions[output.recordedReactions[i]].id for i
                   in selected]
        for i in range(len(output.populations)):
            # Write the headers.
            writer.writerow(['Trajectory', i])
            writer.writerow(headers)
            # Write each data row.
            for j in range(len(output.frameTimes) - 1):
                # Use the beginning of the time interval.
                writer.writerow([output.frameTimes[j]] +
                                [output.reactionCounts[i][j+1][s] -
                                 output.reactionCounts[i][j][s] for s in
                                 selected])
            # Blank line between trajectories.
            writer.writerow([])

    def framesBinnedStat(self, model, output, selected, writer):
        # Whether to output the std. dev.
        stdDev = [self.grid.GetCellValue(i, 1) for i in selected]
        # The headers.
        headers = ['Time']
        for i in selected:
            identifier = model.reactions[output.recordedReactions[i]].id
            headers.append('m(%s)' % identifier)
            if stdDev[i]:
                headers.append('s(%s)' % identifier)
        writer.writerow(headers)
        # Write each data row.
        for f in range(len(output.frameTimes) - 1):
            row = [output.frameTimes[f]]
            for s in selected:
                data = [x[f+1, s] - x[f, s] for x in output.reactionCounts]
                # If the standard deviation box is checked.
                if stdDev[s]:
                    row.extend(list(meanStdDev(data)))
                else:
                    row.append(mean(data))
            writer.writerow(row)

    def allReactionsSpecies(self, model, output, selected, writer):
        # Create the trajectory calculator.
        tc = TrajectoryCalculator(model)
        # The list of reaction identifiers.
        speciesIdentifiers = [x for x in model.species]
        headers = ['Time'] + [speciesIdentifiers[i] for i in selected]
        # For each trajectory.
        for i in range(len(output.indices)):
            # Write the headers.
            writer.writerow(['Trajectory', i])
            writer.writerow(headers)
            # The initial state.
            writer.writerow([output.initialTime] +
                            [output.initialPopulations[i][index] for index in
                             selected])
            # Don't include the start or end times.
            times, populations, reactionCounts =\
                tc.makeFramesAtReactionEvents(output, i, False, False)
            # The state after each reaction event.
            for n in range(len(times)):
                writer.writerow([times[n]] +
                                [populations[n, index] for index in selected])
            # The final state.
            writer.writerow([output.finalTime] +
                            [populations[-1, index] for index in selected])
            # Blank line between trajectories.
            writer.writerow([])

    def allReactionsCumulative(self, model, output, selected, writer):
        # Create the trajectory calculator.
        tc = TrajectoryCalculator(model)
        # The list of reaction identifiers.
        reactionIdentifiers = [x.id for x in model.reactions]
        headers = ['Time'] + [reactionIdentifiers[i] for i in selected]
        # For each trajectory.
        for i in range(len(output.indices)):
            # Write the header.
            writer.writerow(['Trajectory', str(i)])
            writer.writerow(headers)
            # The initial state.
            writer.writerow([output.initialTime] + [0 for index in selected])
            # Don't include the start or end times.
            times, populations, reactionCounts =\
                tc.makeFramesAtReactionEvents(output, i, False, False)
            # The state after each reaction event.
            for n in range(len(times)):
                writer.writerow([times[n]] +
                                [reactionCounts[n, index] for index in
                                 selected])
            # The final state.
            writer.writerow([output.finalTime] +
                            [reactionCounts[-1, index] for index in selected])
            # Blank line between trajectories.
            writer.writerow([])

    def allReactionsEvents(self, model, output, selected, writer):
        # The list of reaction identifiers.
        reactionIdentifiers = [x.id for x in model.reactions]
        # Whether each reaction is selected for output.
        isActive = [x in selected for x in range(len(model.reactions))]
        # For each trajectory.
        for i in range(len(output.indices)):
            # Write the header.
            writer.writerow(['Trajectory', str(i)])
            writer.writerow(['Time', 'Reaction'])
            times = output.times[i]
            indices = output.indices[i]
            # Write each reaction.
            for i in range(len(indices)):
                if isActive[indices[i]]:
                    writer.writerow([times[i], reactionIdentifiers[indices[i]]])
            # Blank line between trajectories.
            writer.writerow([])


def main():
    import os
    from state.TimeSeriesFrames import TimeSeriesFrames
    from state.TimeSeriesAllReactions import TimeSeriesAllReactions
    from state.State import State
    from state.Model import Model
    from state.Reaction import Reaction
    from state.Species import Species
    from state.SpeciesReference import SpeciesReference

    class TestConfiguration(wx.Frame):
        """Test the Configuration panel."""

        def __init__(self, parent, title, state):
            wx.Frame.__init__(self, parent, -1, title)
            panel = Configuration(self, state)

            bestSize = self.GetBestSize()
            # Add twenty to avoid an unecessary horizontal scroll bar.
            size = (bestSize[0] + 80, min(bestSize[1], 700))
            self.SetSize(size)
            self.Fit()

    app = wx.PySimpleApp()

    # Many species.
    s = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i',
         'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r',
         's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    r = ['r1', 'r2', 'r3', 'r4']
    t = TimeSeriesFrames()
    t.setFrameTimes([0, 1, 2])
    t.recordedSpecies = range(len(s))
    t.recordedReactions = range(len(r))
    t.appendPopulations([1]*len(s) + [2]*len(s) + [3]*len(s))
    t.appendReactionCounts([0]*len(r) + [2]*len(r) + [4]*len(r))
    t.appendPopulations([2]*len(s) + [3]*len(s) + [5]*len(s))
    t.appendReactionCounts([0]*len(r) + [3]*len(r) + [6]*len(r))
    state = State()
    # Set the species identifiers.
    modelId = state.insertNewModel()
    model = state.models[modelId]
    model.id = modelId
    model.speciesIdentifiers = s
    # Dummy reactions.
    model.reactions = [Reaction(_id, '', [], [], True, '0') for _id in r]
    # Store the trajectories.
    state.output[(modelId, 'method')] = t
    TestConfiguration(None, 'Time series frames.', state).Show()
    
    s = ['a', 'b', 'c']
    r = ['r1', 'r2', 'r3', 'r4']
    t = TimeSeriesFrames()
    t.setFrameTimes([0, 1, 2])
    t.recordedSpecies = range(len(s))
    t.recordedReactions = range(len(r))
    t.appendPopulations([1]*len(s) + [2]*len(s) + [3]*len(s))
    t.appendReactionCounts([0]*len(r) + [2]*len(r) + [4]*len(r))
    t.appendPopulations([2]*len(s) + [3]*len(s) + [5]*len(s))
    t.appendReactionCounts([0]*len(r) + [3]*len(r) + [6]*len(r))
    state = State()
    # Set the species identifiers.
    modelId = state.insertNewModel()
    model = state.models[modelId]
    model.id = modelId
    model.speciesIdentifiers = s
    # Dummy reactions.
    model.reactions = [Reaction(_id, '', [], [], True, '0') for _id in r]
    # Store the trajectories.
    state.output[(modelId, 'method')] = t
    TestConfiguration(None, 'Time series frames.', state).Show()

    initialTime = 0.
    finalTime = 1.
    t = TimeSeriesAllReactions([0, 1], [0, 1], initialTime, finalTime)
    t.appendIndices([0])
    t.appendTimes([0.5])
    t.appendInitialPopulations([13, 17])
    state = State()
    # Set the species identifiers.
    modelId = state.insertNewModel()
    model = state.models[modelId]
    model.id = modelId
    model.speciesIdentifiers.append('s1')
    model.species['s1'] = Species('C1', 'species 1', '13')
    model.speciesIdentifiers.append('s2')
    model.species['s2'] = Species('C1', 'species 2', '17')
    model.reactions.append(
        Reaction('r1', 'reaction 1', [SpeciesReference('s1')], 
                 [SpeciesReference('s2')], True, '1.5'))
    model.reactions.append(
        Reaction('r2', 'reaction 2', 
                 [SpeciesReference('s1'), SpeciesReference('s2')], 
                 [SpeciesReference('s1', 2)], True, '2.5'))
    # Store the trajectories.
    state.output[(modelId, 'method')] = t
    TestConfiguration(None, 'Time series all reactions.', state).Show()

    app.MainLoop()

if __name__ == '__main__':
    main()

