"""Implements plotting of time series statistics."""

# If we are running the unit tests.
import sys
if __name__ == '__main__':
    sys.path.insert(1, '..')

import wx
import numpy
from PlotTimeSeriesGrid import PlotTimeSeriesGrid
from PlotOptions import PlotOptions

from pylab import errorbar, figure, plot, draw

class Configuration(wx.Panel):
    """Pick the species or reactions to plot."""
    
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

        # Mean or standard deviation.
        horizontal = wx.BoxSizer(wx.HORIZONTAL)
        # Mean.
        self.mean = wx.RadioButton(self, -1, 'Mean')
        self.mean.SetValue(True)
        self.Bind(wx.EVT_RADIOBUTTON, self.onMean, self.mean)
        horizontal.Add(self.mean, 0, wx.ALL, 5)
        # Standard deviation.
        self.stdDev = wx.RadioButton(self, -1, 'Std. Dev.')
        self.Bind(wx.EVT_RADIOBUTTON, self.onStdDev, self.stdDev)
        horizontal.Add(self.stdDev, 0, wx.ALL, 5)
        sizer.Add(horizontal, 0, wx.ALL, 5)

        # The grid of species.
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
        self.Fit()

    def onOutput(self, event):
        self.update()
        event.Skip()

    def onMean(self, event):
        self.grid.showStdDev()
        event.Skip()

    def onStdDev(self, event):
        self.grid.hideStdDev()
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

        # Update the grid.
        modelId = self.outputKeys[index][0]
        model = self.state.models[modelId]
        output = self.state.output[self.outputKeys[index]]
        identifiers = [model.speciesIdentifiers[_i]
                       for _i in output.recordedSpecies]
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
            if self.state.output[key].__class__.__name__ == 'StatisticsFrames':
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
            wx.MessageBox('There is no selected simulation output.',
                          'Error!', style=wx.OK|wx.ICON_EXCLAMATION)
            return
        # Save any values being edited in the grid.
        self.grid.saveEditControlValue()
        # Choose the appropriate kind of plot.
        output = self.state.output[self.outputKeys[index]]
        self.plotFrames(output)

    def plotFrames(self, output):
        # Check that at least one row has been selected.
        if not self.grid.areAnyItemsSelected():
            wx.MessageBox('No rows are selected.', 'Error.')
            return
        # The items to plot.
        indices = self.grid.getCheckedItems()
        if not indices:
            return
        if self.mean.GetValue():
            # Plot the mean and optionally the standard deviation.
            for index in indices:
                times = output.frameTimes
                y = [frame[index][0] for frame in output.statistics]
                # If the standard deviation box is checked.
                if self.grid.GetCellValue(index, 1):
                    yerr = [frame[index][1] for frame in output.statistics]
                else:
                    yerr = None
                if self.grid.useMarkers(index):
                    errorbar(times, y, yerr=yerr, 
                             **self.grid.getLineAndMarkerStyles(index))
                else:
                    errorbar(times, y, yerr=yerr,
                             **self.grid.getLineStyles(index))
        else:
            # Plot the standard deviation.
            for index in indices:
                times = output.frameTimes
                y = [frame[index][1] for frame in output.statistics]
                if self.grid.useMarkers(index):
                    plot(times, y, **self.grid.getLineAndMarkerStyles(index))
                else:
                    plot(times, y, **self.grid.getLineStyles(index))
        self._showLegendAndLabels(indices)
        self.options.setLimits()
        draw()

        
def main():
    from FigureNumber import FigureNumber
    from state.StatisticsFrames import StatisticsFrames
    from state.State import State
    from state.Model import Model
    #from state.Reaction import Reaction

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

    s = ['a', 'b', 'c']
    t = StatisticsFrames([0, 1, 2])
    t.setFrameTimes([0, 1, 2])
    t.setStatistics([1, 0.1, 2, 0.2, 3, 0.3] * 3)
    state = State()
    # Set the species identifiers.
    modelId = state.insertNewModel()
    model = state.models[modelId]
    model.id = modelId
    model.speciesIdentifiers = s
    # Dummy reactions.
    #model.reactions = [Reaction(_id, '', [], [], True, '0') for _id in r]
    # Store the trajectories.
    state.output[(modelId, 'method')] = t
    TestConfiguration(None, 'Populations.', state, figureNumber).Show()

    app.MainLoop()

if __name__ == '__main__':
    main()

