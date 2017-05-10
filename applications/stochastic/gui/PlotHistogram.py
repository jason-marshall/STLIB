"""Implements plotting for histograms."""

import wx
import numpy
from colorsys import hsv_to_rgb

from CheckBoxSelectAll import CheckBoxSelectAll
from PlotHistogramGrid import PlotHistogramGrid
from PlotOptions import PlotOptions

from pylab import figure, draw, bar, plot

class Configuration(wx.Panel):
    """Pick the frames and species to plot."""
    
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

        # Multi-species.
        horizontal = wx.BoxSizer(wx.HORIZONTAL)
        self.multiSpecies = wx.RadioButton(self, -1,
                                           'Multi-species. Select frame:',
                                           style=wx.RB_GROUP)
        self.multiSpecies.SetValue(True)
        self.Bind(wx.EVT_RADIOBUTTON, self.onMultiSpecies, self.multiSpecies)
        horizontal.Add(self.multiSpecies, 0, wx.ALL, 5)
        self.frame = wx.Choice(self, size=(200,-1), choices=[])
        self.Bind(wx.EVT_CHOICE, self.onMultiSpecies, self.frame)
        horizontal.Add(self.frame, 0, wx.ALL, 5)
        sizer.Add(horizontal, 0, wx.ALL, 0)

        # Multi-frame.
        horizontal = wx.BoxSizer(wx.HORIZONTAL)
        self.multiFrame = wx.RadioButton(self, -1,
                                         'Multi-frame. Select species:')
        self.Bind(wx.EVT_RADIOBUTTON, self.onMultiFrame, self.multiFrame)
        horizontal.Add(self.multiFrame, 0, wx.ALL, 5)
        self.species = wx.Choice(self, size=(200,-1), choices=[])
        self.Bind(wx.EVT_CHOICE, self.onMultiSpecies, self.species)
        horizontal.Add(self.species, 0, wx.ALL, 5)
        sizer.Add(horizontal, 0, wx.ALL, 0)

        # The grid of frames or species.
        self.grid = PlotHistogramGrid(self)
        sizer.Add(self.grid, 1, wx.EXPAND)
        sizer.Add(wx.StaticLine(self), 0, wx.EXPAND|wx.ALL, 5)

        # New figure.
        self.newFigure = wx.CheckBox(self, -1, 'Plot in new figure')
        self.newFigure.SetValue(True)
        sizer.Add(self.newFigure, 0, wx.ALIGN_TOP, 5)

        # The plot options.
        self.options = PlotOptions(self)
        sizer.Add(self.options, 0, wx.EXPAND)

        # Plot buttons.
        buttons = wx.BoxSizer(wx.HORIZONTAL)
        b = wx.Button(self, -1, 'Plot separately')
        self.Bind(wx.EVT_BUTTON, self.onPlotSeparately, b)
        buttons.Add(b, 0, wx.ALIGN_RIGHT, 5)
        b = wx.Button(self, -1, 'Plot together')
        self.Bind(wx.EVT_BUTTON, self.onPlotTogether, b)
        buttons.Add(b, 0, wx.ALIGN_RIGHT, 5)
        sizer.Add(buttons, 0, wx.ALIGN_RIGHT | wx.ALIGN_TOP, 5)

        self.SetSizer(sizer)
        self.refresh()
        self.Layout()
        self.Fit()

    def onOutput(self, event):
        self.update()
        event.Skip()

    def update(self):
        index = self.outputChoice.GetSelection()
        if index == wx.NOT_FOUND:
            # Clear the species and frames.
            self.species.Clear()
            self.frame.Clear()
            return
        # Check that the simulation output has not disappeared.
        if not self.outputKeys[index] in self.state.output:
            self.refresh()
            return
        output = self.state.output[self.outputKeys[index]]
        modelId = self.outputKeys[index][0]
        model = self.state.models[modelId]

        # Multi-species, frame selection.
        if output.__class__.__name__ == 'HistogramFrames':
            frames = [str(_t) for _t in output.frameTimes]
            self.multiFrame.Enable()
            self.species.Enable()
        else:
            # For steady state histograms there are no frames.
            frames = []
            # Multi-species is the only viable option.
            self.multiSpecies.SetValue(True)
            self.multiFrame.Disable()
            self.species.Disable()
        # Record the current selection.
        selection = self.frame.GetSelection()
        # Rebuild the list of choices.
        self.frame.Clear()
        for x in frames:
            self.frame.Append(x)
        # Set the new selection.
        if selection != wx.NOT_FOUND and\
                selection < self.frame.GetCount():
            self.frame.SetSelection(selection)
        else:
            self.frame.SetSelection(0)

        # Multi-frame, species selection.
        # Record the current selection.
        selection = self.species.GetSelection()
        # Rebuild the list of choices.
        self.species.Clear()
        for i in output.recordedSpecies:
            self.species.Append(model.speciesIdentifiers[i])
        # Set the new selection.
        if selection != wx.NOT_FOUND and\
                selection < self.species.GetCount():
            self.species.SetSelection(selection)
        else:
            self.species.SetSelection(0)

        # Update the grid.
        if self.multiSpecies.GetValue():
            self.grid.setIdentifiers([model.speciesIdentifiers[_i]
                                      for _i in output.recordedSpecies])
        else:
            self.grid.setIdentifiers(frames)


    def refresh(self):
        # Get the histogram outputs.
        self.outputKeys = []
        for key in self.state.output:
            if self.state.output[key].__class__.__name__ in\
                    ('HistogramFrames', 'HistogramAverage'):
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

    def onMultiSpecies(self, event):
        self.update()
        event.Skip()

    def onMultiFrame(self, event):
        self.update()
        event.Skip()

    def onPlotSeparately(self, event):
        """Choose between plotSeparatelyFrames and plotSeparatelyAverage."""
        index = self.outputChoice.GetSelection()
        if index == wx.NOT_FOUND:
            wx.MessageBox('There is no histogram simulation output.',
                          'Error!', style=wx.OK|wx.ICON_EXCLAMATION)
            return
        # Save any values being edited in the grid.
        self.grid.saveEditControlValue()
        # Check that at least one row has been selected.
        if not self.grid.areAnyItemsSelected():
            wx.MessageBox('No rows are selected.', 'Error.')
            return

        output = self.state.output[self.outputKeys[index]]
        name = output.__class__.__name__
        modelId = self.outputKeys[index][0]
        model = self.state.models[modelId]
        if name == 'HistogramFrames':
            self.plotSeparatelyFrames(output, model)
        elif name == 'HistogramAverage':
            self.plotSeparatelyAverage(output, model)
        else:
            assert(False)

    def onPlotTogether(self, event):
        """Choose between plotTogetherFrames and plotTogetherAverage."""
        index = self.outputChoice.GetSelection()
        if index == wx.NOT_FOUND:
            wx.MessageBox('There is no histogram simulation output.',
                          'Error!', style=wx.OK|wx.ICON_EXCLAMATION)
            return
        # Save any values being edited in the grid.
        self.grid.saveEditControlValue()
        # Check that at least one row has been selected.
        if not self.grid.areAnyItemsSelected():
            wx.MessageBox('No rows are selected.', 'Error.')
            return

        output = self.state.output[self.outputKeys[index]]
        name = output.__class__.__name__
        modelId = self.outputKeys[index][0]
        model = self.state.models[modelId]
        if name == 'HistogramFrames':
            self.plotTogetherFrames(output, model)
        elif name == 'HistogramAverage':
            self.plotTogetherAverage(output, model)
        else:
            assert(False)

    def plotSeparatelyFrames(self, output, model):
        """Make a plot for each specified species and frame."""
        # The items to plot.
        indices = self.grid.getCheckedItems()
        if not indices:
            return
        useNewFigure = self.newFigure.GetValue()
        size = self.options.getCustomFigureSize()
        if self.multiSpecies.GetValue():
            frame = self.frame.GetSelection()
            for species in indices:
                if useNewFigure:
                    # Start a new figure.
                    self.figureNumber += 1
                    figure(num=self.figureNumber(), figsize=size)
                useNewFigure = True
                self.plotHistogram(output.histograms[frame][species], species)
                # The title, x label, and y label.
                self.options.showLegendAndLabels()
                self.options.setLimits()
                draw()
        else:
            species = self.species.GetSelection()
            for frame in indices:
                if useNewFigure:
                    # Start a new figure.
                    self.figureNumber += 1
                    figure(num=self.figureNumber(), figsize=size)
                useNewFigure = True
                self.plotHistogram(output.histograms[frame][species], frame)
                # The title, x label, and y label.
                self.options.showLegendAndLabels()
                self.options.setLimits()
                draw()

    def plotTogetherFrames(self, output, model):
        """Make a single plot with each specified species and frame."""
        # The items to plot.
        indices = self.grid.getCheckedItems()
        if not indices:
            return
        if self.newFigure.GetValue():
            # Start a new figure.
            self.figureNumber += 1
            size = self.options.getCustomFigureSize()
            figure(num=self.figureNumber(), figsize=size)
        # Plot the histograms without yet rendering them.
        if self.multiSpecies.GetValue():
            frame = self.frame.GetSelection()
            for species in indices:
                self.plotHistogram(output.histograms[frame][species], species)
        else:
            species = self.species.GetSelection()
            for frame in indices:
                self.plotHistogram(output.histograms[frame][species], frame)
        # The legend, title, x label, and y label.
        self.options.showLegendAndLabels()
        self.options.setLimits()
        draw()

    def plotSeparatelyAverage(self, output, model):
        """Make a plot for each specified species."""
        # The items to plot.
        indices = self.grid.getCheckedItems()
        if not indices:
            return
        useNewFigure = self.newFigure.GetValue()
        size = self.options.getCustomFigureSize()
        for species in indices:
            if useNewFigure:
                # Start a new figure.
                self.figureNumber += 1
                figure(num=self.figureNumber(), figsize=size)
            useNewFigure = True
            self.plotHistogram(output.histograms[species], species)
            # The title, x label, and y label.
            self.options.showLegendAndLabels()
            self.options.setLimits()
            draw()


    def plotTogetherAverage(self, output, model):
        """Make a single plot with each specified species."""
        # The items to plot.
        indices = self.grid.getCheckedItems()
        if not indices:
            return
        if self.newFigure.GetValue():
            # Start a new figure.
            self.figureNumber += 1
            size = self.options.getCustomFigureSize()
            figure(num=self.figureNumber(), figsize=size)
        # Plot the histograms without yet rendering them.
        frame = self.frame.GetSelection()
        for species in indices:
            self.plotHistogram(output.histograms[species], species)
        # The legend, title, x label, and y label.
        self.options.showLegendAndLabels()
        self.options.setLimits()
        draw()


    def plotHistogram(self, histogram, row):
        # The empirical PMF.
        height = histogram.getPmf()
        # Remove the trailing zeros.
        last = 0
        for i in range(len(height)):
            if height[i] != 0:
                last = i
        height = height[0:last + 1]
        numBins = len(height)
        width = histogram.getWidth()
        
        # Plot the filled portion of the histogram if specified.
        isFilled, fillColor, fillAlpha = self.grid.getFillStyle(row)
        if isFilled:
            # The left edges of the bins.
            left = numpy.zeros(numBins)
            for i in range(len(left)):
                left[i] = histogram.lowerBound + i * width
            bar(left, height, width, fill=True, color=fillColor,
                linewidth=0, alpha=fillAlpha)

        # The x coordinates for the line plot.
        x = numpy.zeros(2 * (numBins + 1))
        for i in range(numBins + 1):
            x[2 * i] = x[2 * i + 1] = histogram.lowerBound + i * width
        # The y coordinates for the line plot.
        y = numpy.zeros(2 * (numBins + 1))
        y[0] = 0.
        for i in range(numBins):
            y[2 * i + 1] = y[2 * i + 2] = height[i]
        y[-1] = 0.
        plot(x, y, **self.grid.getLineStyles(row))
        
class TestConfiguration(wx.Frame):
    """Test the Configuration panel."""

    def __init__(self, parent, title, state, figureNumber):
        wx.Frame.__init__(self, parent, -1, title)
        panel = Configuration(self, state, figureNumber)

        self.SetSize(self.GetBestSize())
        self.Fit()

def frames():
    from FigureNumber import FigureNumber
    from state.State import State
    from state.Model import Model
    from state.Reaction import Reaction
    from state.Species import Species
    from state.SpeciesReference import SpeciesReference
    from state.Histogram import Histogram
    from state.HistogramFrames import HistogramFrames

    # Figure number.
    figureNumber = FigureNumber()

    # A histogram.
    numberOfBins = 4
    multiplicity = 2
    h = Histogram(numberOfBins, multiplicity)
    h.setCurrentToMinimum()
    h.accumulate(0, 1)
    h.accumulate(1, 2)
    h.accumulate(2, 2)
    h.accumulate(3, 1)

    # Simulation output.
    frameTimes = [0, 1]
    recordedSpecies = [0, 1, 2]
    hf = HistogramFrames(numberOfBins, multiplicity, recordedSpecies)
    hf.setFrameTimes(frameTimes)
    for i in range(len(frameTimes)):
        for j in range(len(recordedSpecies)):
            hf.histograms[i][j].merge(h)

    # The model.
    model = Model()
    model.speciesIdentifiers = ['s1', 's2', 's3']

    # The state.
    state = State()
    state.models['model'] = model
    state.output[('model', 'method')] = hf

    app = wx.PySimpleApp()
    TestConfiguration(None, 'Populations.', state, figureNumber).Show()
    app.MainLoop()

def average():
    from FigureNumber import FigureNumber
    from state.State import State
    from state.Model import Model
    from state.Reaction import Reaction
    from state.Species import Species
    from state.SpeciesReference import SpeciesReference
    from state.Histogram import Histogram
    from state.HistogramAverage import HistogramAverage

    # Figure number.
    figureNumber = FigureNumber()

    # A histogram.
    numberOfBins = 4
    multiplicity = 2
    h = Histogram(numberOfBins, multiplicity)
    h.setCurrentToMinimum()
    h.accumulate(0, 1)
    h.accumulate(1, 2)
    h.accumulate(2, 2)
    h.accumulate(3, 1)

    # Simulation output.
    recordedSpecies = [0, 1, 2]
    output = HistogramAverage(numberOfBins, multiplicity, recordedSpecies)
    for x in output.histograms:
        x.merge(h)

    # The model.
    model = Model()
    model.speciesIdentifiers = ['s1', 's2', 's3']

    # The state.
    state = State()
    state.models['model'] = model
    state.output[('model', 'method')] = output

    app = wx.PySimpleApp()
    TestConfiguration(None, 'Populations.', state, figureNumber).Show()
    app.MainLoop()

if __name__ == '__main__':
    frames()
    average()
