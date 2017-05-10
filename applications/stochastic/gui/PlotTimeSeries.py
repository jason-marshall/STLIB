"""Implements plotting of time series data."""

# If we are running the unit tests.
import sys
if __name__ == '__main__':
    sys.path.insert(1, '..')

import wx
import numpy
from statistics import mean, meanStdDev
from PlotTimeSeriesGrid import PlotTimeSeriesGrid
from PlotOptions import PlotOptions
from state.TrajectoryCalculator import TrajectoryCalculator

from pylab import close, errorbar, figure, plot, draw

def closeAll():
    close('all')

class Configuration(wx.Panel):
    """Pick the species or reactions to plot. If a trajectory calculator is
    specified, then the trajectories record every reaction event. Otherwise
    the trajectories record the state at frames."""
    
    def __init__(self, parent, state, figureNumber):
        wx.Panel.__init__(self, parent, -1)
        self.state = state
        self.figureNumber = figureNumber
        self.outputKeys = []

        sizer = wx.BoxSizer(wx.VERTICAL)

        # Output choice.
        self.outputChoice = wx.Choice(self, size=(400,-1), choices=[])
        self.Bind(wx.EVT_CHOICE, self.onOutput, self.outputChoice)
        sizer.Add(self.outputChoice, 0, wx.EXPAND, 5)

        # Species, cumulative reactions, or binned reactions.
        horizontal = wx.BoxSizer(wx.HORIZONTAL)
        self.species = wx.RadioButton(self, -1, 'Species', style=wx.RB_GROUP)
        self.species.SetValue(True)
        self.Bind(wx.EVT_RADIOBUTTON, self.onSpecies, self.species)
        horizontal.Add(self.species, 0, wx.ALL, 5)
        self.cumulativeReactions = wx.RadioButton(self, -1,
                                                  'Cumulative Reactions')
        self.Bind(wx.EVT_RADIOBUTTON, self.onCumulativeReactions,
                  self.cumulativeReactions)
        horizontal.Add(self.cumulativeReactions, 0, wx.ALL, 5)
        self.binnedReactions = wx.RadioButton(self, -1,
                                              'Binned Reactions')
        self.Bind(wx.EVT_RADIOBUTTON, self.onBinnedReactions,
                  self.binnedReactions)
        horizontal.Add(self.binnedReactions, 0, wx.ALL, 5)
        sizer.Add(horizontal, 0, wx.ALL, 5)

        # Trajectories, mean, or standard deviation.
        horizontal = wx.BoxSizer(wx.HORIZONTAL)
        # Trajectories.
        self.trajectories = wx.RadioButton(self, -1, 'Trajectories',
                                           style=wx.RB_GROUP)
        self.trajectories.SetValue(True)
        self.Bind(wx.EVT_RADIOBUTTON, self.onTrajectories, self.trajectories)
        horizontal.Add(self.trajectories, 0, wx.ALL, 5)
        # Mean.
        self.mean = wx.RadioButton(self, -1, 'Mean')
        self.Bind(wx.EVT_RADIOBUTTON, self.onMean, self.mean)
        horizontal.Add(self.mean, 0, wx.ALL, 5)
        # Standard deviation.
        self.stdDev = wx.RadioButton(self, -1, 'Std. Dev.')
        self.Bind(wx.EVT_RADIOBUTTON, self.onStdDev, self.stdDev)
        horizontal.Add(self.stdDev, 0, wx.ALL, 5)
        sizer.Add(horizontal, 0, wx.ALL, 5)

        # The grid of species or reactions.
        self.grid = PlotTimeSeriesGrid(self)
        sizer.Add(self.grid, 1, wx.EXPAND)
        sizer.Add(wx.StaticLine(self), 0, wx.EXPAND|wx.ALL, 5)

        # The plot options.
        self.options = PlotOptions(self)
        sizer.Add(self.options, 0, wx.EXPAND)

        # Plot buttons.
        buttons = wx.BoxSizer(wx.HORIZONTAL)
        b = wx.Button(self, -1, 'Plot')
        self.Bind(wx.EVT_BUTTON, self.onPlot, b)
        buttons.Add(b, 0, wx.ALIGN_RIGHT, 5)
        b = wx.Button(self, -1, 'New plot')
        self.Bind(wx.EVT_BUTTON, self.onNewPlot, b)
        buttons.Add(b, 0, wx.ALIGN_RIGHT, 5)
        sizer.Add(buttons, 0, wx.ALIGN_RIGHT | wx.ALIGN_TOP, 5)

        self.SetSizer(sizer)
        self.refresh()
        #self.Layout()
        self.Fit()

    def onOutput(self, event):
        self.update()
        event.Skip()

    def onTrajectories(self, event):
        self.grid.hideStdDev()
        event.Skip()

    def onMean(self, event):
        self.grid.showStdDev()
        event.Skip()

    def onStdDev(self, event):
        self.grid.hideStdDev()
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
            self.binnedReactions.Enable()
            self.mean.Enable()
            self.stdDev.Enable()
        else:
            # You can't plot binned reaction counts for all-reaction style 
            # trajectories.
            if self.binnedReactions.GetValue():
                self.species.SetValue(True)
            self.binnedReactions.Disable()
            # You can only plot trajectories, not statistics, for all-reaction
            # stlye trajectories.
            self.trajectories.SetValue(True)
            self.mean.Disable()
            self.stdDev.Disable()
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
        if self.mean.GetValue():
            self.grid.showStdDev()
        else:
            self.grid.hideStdDev()


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

    def onPlot(self, event):
        size = self.options.getCustomFigureSize()
        figure(num=self.figureNumber(), figsize=size)
        # Draw the plot.
        self.plot()

    def onNewPlot(self, event):
        # Start a new figure.
        self.figureNumber += 1
        size = self.options.getCustomFigureSize()
        figure(self.figureNumber(), figsize=size)
        # Draw the plot.
        self.plot()

    def _showLegendAndLabels(self, indices):
        # Legend.
        if self.options.legend.IsChecked():
            # Make empty plots to register the labels for the legend.
            for index in indices:
                if self.grid.useMarkers(index):
                    plot([], [], label=self.grid.getLegendLabel(index),
                         **self.grid.getLineAndMarkerStyles(index))
                else:
                    plot([], [], label=self.grid.getLegendLabel(index),
                         **self.grid.getLineStyles(index))
        self.options.showLegendAndLabels()

    def plot(self):
        index = self.outputChoice.GetSelection()
        if index == wx.NOT_FOUND:
            wx.MessageBox('There is no time series simulation output.',
                          'Error!', style=wx.OK|wx.ICON_EXCLAMATION)
            return
        # Save any values being edited in the grid.
        self.grid.saveEditControlValue()
        # Choose the appropriate kind of plot.
        output = self.state.output[self.outputKeys[index]]
        name = output.__class__.__name__
        if name == 'TimeSeriesFrames':
            self.plotFrames(output)
        elif name == 'TimeSeriesAllReactions':
            self.plotAllReactions(output, index)
        else:
            assert(False)

    def plotFrames(self, output):
        # Record the kind of plot to generate.
        isSpeciesSelected = self.species.GetValue()
        isCumulativeReactionsSelected = self.cumulativeReactions.GetValue()
        #isBinnedReactionsSelected = self.binnedReactions.GetValue()
        isTrajectoriesSelected = self.trajectories.GetValue()
        isMeanSelected = self.mean.GetValue()
        #isStdDevSelected = self.stdDev.GetValue()

        # Check that at least one row has been selected.
        if not self.grid.areAnyItemsSelected():
            wx.MessageBox('No rows are selected.', 'Error.')
            return
        # The items to plot.
        indices = self.grid.getCheckedItems()
        if not indices:
            return
        if isTrajectoriesSelected:
            # Plot each trajectory.
            for i in range(len(output.populations)):
                for index in indices:
                    if isSpeciesSelected:
                        p = output.populations[i][:, index]
                        times = output.frameTimes
                    elif isCumulativeReactionsSelected:
                        p = output.reactionCounts[i][:, index]
                        times = output.frameTimes
                    else: # Binned reaction counts.
                        p = output.reactionCounts[i][1:, index] - \
                            output.reactionCounts[i][:-1, index]
                        # Midpoints.
                        times = 0.5 * (output.frameTimes[0:-1] + 
                                       output.frameTimes[1:])
                    if self.grid.useMarkers(index):
                        plot(times, p,
                             **self.grid.getLineAndMarkerStyles(index))
                    else:
                        plot(times, p, **self.grid.getLineStyles(index))
        elif isMeanSelected:
            # Plot the mean and optionally the standard deviation.
            for index in indices:
                if isSpeciesSelected:
                    data = [x[:, index] for x in output.populations]
                    times = output.frameTimes
                elif isCumulativeReactionsSelected:
                    data = [x[:, index] for x in output.reactionCounts]
                    times = output.frameTimes
                else: # Binned reaction counts.
                    data = [x[1:, index] - x[:-1, index] for x in 
                            output.reactionCounts]
                    # Midpoints.
                    times = output.frameTimes
                    times = 0.5 * (times[0:-1] + times[1:])
                # If the standard deviation box is checked.
                if self.grid.GetCellValue(index, 1):
                    y, yerr = meanStdDev(data)
                else:
                    y = mean(data)
                    yerr = None
                if self.grid.useMarkers(index):
                    errorbar(times, y, yerr = yerr, 
                             **self.grid.getLineAndMarkerStyles(index))
                else:
                    errorbar(times, y, yerr=yerr,
                             **self.grid.getLineStyles(index))
        else: # isStdDevSelected
            # Plot the standard deviation.
            for index in indices:
                if isSpeciesSelected:
                    data = [x[:, index] for x in output.populations]
                    times = output.frameTimes
                elif isCumulativeReactionsSelected:
                    data = [x[:, index] for x in output.reactionCounts]
                    times = output.frameTimes
                else: # Binned reaction counts.
                    data = [x[1:, index] - x[:-1, index] for x in 
                            output.reactionCounts]
                    # Midpoints.
                    times = output.frameTimes
                    times = 0.5 * (times[0:-1] + times[1:])
                # If the standard deviation box is checked.
                y, yerr = meanStdDev(data)
                if self.grid.useMarkers(index):
                    plot(times, yerr, **self.grid.getLineAndMarkerStyles(index))
                else:
                    plot(times, yerr, **self.grid.getLineStyles(index))
        self._showLegendAndLabels(indices)
        self.options.setLimits()
        draw()

    def plotAllReactions(self, output, outputIndex):
        # Create the trajectory calculator.
        modelId = self.outputKeys[outputIndex][0]
        model = self.state.models[modelId]
        tc = TrajectoryCalculator(model)

        # Record the kind of plot to generate.
        isSpeciesSelected = self.species.GetValue()
        #isCumulativeReactionsSelected = self.cumulativeReactions.GetValue()

        # Check that at least one row has been selected.
        if not self.grid.areAnyItemsSelected():
            wx.MessageBox('No rows are selected.', 'Error.')
            return
        # The items to plot.
        indices = self.grid.getCheckedItems()
        if not indices:
            return
        # Plot each trajectory.
        for i in range(len(output.indices)):
            # Don't include the start or end times.
            times, populations, reactionCounts =\
                tc.makeFramesAtReactionEvents(output, i, False, False)
            # The times for plotting.
            t = numpy.zeros(2 * len(times) + 2)
            t[0] = output.initialTime
            for n in range(len(times)):
                t[2*n+1] = times[n]
                t[2*n+2] = times[n]
            t[-1] = output.finalTime
            # For each selected item.
            for index in indices:
                # Array for the a species population or reaction count.
                x = numpy.zeros(2 * len(times) + 2)
                if isSpeciesSelected:
                    # Slice to get a species population array.
                    p = populations[:, index]
                    # Initial population.
                    x[0] = output.initialPopulations[i][index]
                    for n in range(len(p)):
                        # Population before reaction.
                        x[2*n+1] = x[2*n]
                        # Population after reaction.
                        x[2*n+2] = p[n]
                    # Final population.
                    x[-1] = x[-2]
                else:
                    # Slice to get a species population array.
                    p = reactionCounts[:, index]
                    # Initial reaction count.
                    x[0] = 0
                    for n in range(len(p)):
                        # Count before reaction.
                        x[2*n+1] = x[2*n]
                        # Count after reaction.
                        x[2*n+2] = p[n]
                    # Final Count.
                    x[-1] = x[-2]
                if self.grid.useMarkers(index):
                    plot(t, x, **self.grid.getLineAndMarkerStyles(index))
                else:
                    plot(t, x, **self.grid.getLineStyles(index))
        self._showLegendAndLabels(indices)
        self.options.setLimits()
        draw()

        
def main():
    from FigureNumber import FigureNumber
    from state.TimeSeriesFrames import TimeSeriesFrames
    from state.TimeSeriesAllReactions import TimeSeriesAllReactions
    from state.State import State
    from state.Model import Model
    from state.Reaction import Reaction
    from state.Species import Species
    from state.SpeciesReference import SpeciesReference

    class TestConfiguration(wx.Frame):
        """Test the Configuration panel."""

        def __init__(self, parent, title, state, figureNumber):
            wx.Frame.__init__(self, parent, -1, title)
            panel = Configuration(self, state, figureNumber)

            bestSize = self.GetBestSize()
            # Add twenty to avoid an unecessary horizontal scroll bar.
            size = (bestSize[0] + 80, min(bestSize[1], 700))
            self.SetSize(size)
            self.Fit()

    app = wx.PySimpleApp()
    figureNumber = FigureNumber()

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
    TestConfiguration(None, 'Populations.', state, figureNumber).Show()
    
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
    TestConfiguration(None, 'Populations.', state, figureNumber).Show()

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
    TestConfiguration(None, 'Populations.', state, figureNumber).Show()

    app.MainLoop()

if __name__ == '__main__':
    main()

