"""Implements exporting of histogram data."""

# If we are running the unit tests.
import sys
if __name__ == '__main__':
    sys.path.insert(1, '..')

import wx
import os.path
import csv
from ExportHistogramGrid import ExportHistogramGrid
from messages import openWrite

delimiterChoices = ['comma', 'space', 'tab']
delimiterValues = [',', ' ', '\t']

class Configuration(wx.Panel):
    """Pick the species to export."""
    
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

        # The grid of species.
        self.grid = ExportHistogramGrid(self)
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
        # Update the grid.
        modelId = self.outputKeys[index][0]
        model = self.state.models[modelId]
        self.grid.setIdentifiers([model.speciesIdentifiers[_i]
                                  for _i in output.recordedSpecies])

    def refresh(self):
        """This is called when the list of outputs changes in the
        application."""
        # Get the time series outputs.
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

    def onExport(self, event):
        self.export()

    def export(self):
        index = self.outputChoice.GetSelection()
        if index == wx.NOT_FOUND:
            wx.MessageBox('There is no histogram simulation output.',
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
        if name == 'HistogramFrames':
            self.frames(model, output, selected, writer)
        elif name == 'HistogramAverage':
            self.average(model, output, selected, writer)
        else:
            assert False

    def frames(self, model, output, selected, writer):
        # For each frame.
        for frameIndex in range(len(output.frameTimes)):
            # The time.
            writer.writerow(['Time = ' + str(output.frameTimes[frameIndex])])
            # Blank line.
            writer.writerow([])
            # For each species to record.
            for s in selected:
                # The species identifier.
                speciesIndex = output.recordedSpecies[s]
                writer.writerow(['Bin',
                                 model.speciesIdentifiers[speciesIndex]])
                # The histogram for this frame and species.
                h = output.histograms[frameIndex][s]
                # The probabilities.
                p = h.getProbabilities()
                # For each bin.
                for i in range(len(p)):
                    # The lower bound for the bin and the probability.
                    writer.writerow([h.lowerBound + i * h.getWidth(), p[i]])
                # Blank line between species.
                writer.writerow([])

    def average(self, model, output, selected, writer):
        # For each species to record.
        for s in selected:
            # The species identifier.
            speciesIndex = output.recordedSpecies[s]
            writer.writerow(['Bin',
                             model.speciesIdentifiers[speciesIndex]])
            # The histogram for this frame and species.
            h = output.histograms[s]
            # The probabilities.
            p = h.getProbabilities()
            # For each bin.
            for i in range(len(p)):
                # The lower bound for the bin and the probability.
                writer.writerow([h.lowerBound + i * h.getWidth(), p[i]])
            # Blank line between species.
            writer.writerow([])
        
class TestConfiguration(wx.Frame):
    """Test the Configuration panel."""

    def __init__(self, parent, title, state):
        wx.Frame.__init__(self, parent, -1, title)
        panel = Configuration(self, state)
        self.directory = os.getcwd()

        bestSize = self.GetBestSize()
        # Add twenty to avoid an unecessary horizontal scroll bar.
        size = (bestSize[0] + 80, min(bestSize[1], 700))
        self.SetSize(size)
        self.Fit()

def frames():
    from state.State import State
    from state.Model import Model
    from state.Reaction import Reaction
    from state.Species import Species
    from state.SpeciesReference import SpeciesReference
    from state.Histogram import Histogram
    from state.HistogramFrames import HistogramFrames

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
    TestConfiguration(None, 'Frames.', state).Show()
    app.MainLoop()

def average():
    from state.State import State
    from state.Model import Model
    from state.Reaction import Reaction
    from state.Species import Species
    from state.SpeciesReference import SpeciesReference
    from state.Histogram import Histogram
    from state.HistogramAverage import HistogramAverage

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
    TestConfiguration(None, 'Average.', state).Show()
    app.MainLoop()

if __name__ == '__main__':
    frames()
    average()
