"""Interface for measuring distance between histograms."""

# If we are running the unit tests.
if __name__ == '__main__':
    import sys
    sys.path.insert(1, '..')

import wx

from state.Histogram import histogramDistance

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
        if output.__class__.__name__ == 'HistogramFrames':
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

    def histograms(self):
        """Return a list of the selected histograms."""
        index = self.output.GetSelection()
        if index == wx.NOT_FOUND:
            return None
        # Check that the simulation output has not disappeared.
        if not self.outputKeys[index] in self.state.output:
            self.refresh()
            return None
        output = self.state.output[self.outputKeys[index]]

        s = self.species.GetSelection()
        if s == wx.NOT_FOUND:
            return None

        if output.__class__.__name__ == 'HistogramFrames':
            f = self.frame.GetSelection()
            if f == wx.NOT_FOUND:
                return None
            if f == 0:
                if s == 0:
                    result = []
                    for frame in output.histograms:
                        result.extend(frame)
                    return result
                else:
                    result = []
                    for frame in output.histograms:
                        result.append(frame[s-1])
                    return result
            else:
                if s == 0:
                    return output.histograms[f-1]
                else:
                    return [output.histograms[f-1][s-1]]
        elif output.__class__.__name__ == 'HistogramAverage':
            if s == 0:
                return output.histograms
            else:
                return [output.histograms[s-1]]
        else:
            assert(False)

    def refresh(self):
        # Get the histogram outputs.
        self.outputKeys = []
        for key in self.state.output:
            if self.state.output[key].__class__.__name__ in\
                    ('HistogramFrames', 'HistogramAverage'):
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

class HistogramDistance(wx.Frame):
    def __init__(self, state, title, parent=None):
        wx.Frame.__init__(self, parent, -1, title)
        self.state = state

        # Selections.
        selectionsSizer = wx.BoxSizer(wx.HORIZONTAL)
        self.selections = [Selection(self, state), Selection(self, state)]
        for s in self.selections:
            selectionsSizer.Add(s, 1, wx.EXPAND | wx.ALL, 5)
            self.Bind(wx.EVT_CHOICE, self.onModification, s.output)
            self.Bind(wx.EVT_CHOICE, self.onModification, s.species)
            self.Bind(wx.EVT_CHOICE, self.onModification, s.frame)
        sizer = wx.BoxSizer(wx.VERTICAL)
        # Don't expand in the vertical direction.
        sizer.Add(selectionsSizer, 0, wx.EXPAND | wx.ALIGN_TOP, 5)

        # Distance.
        self.distance = wx.StaticText(self, -1, '')
        sizer.Add(self.distance, 0, wx.ALIGN_TOP, 5)

        self.SetSizer(sizer)
        self.Fit()
        self.SetSize(self.GetBestSize())
        self.computeDistance()
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
        self.computeDistance()

    def onModification(self, event):
        self.computeDistance()

    def computeDistance(self):
        self.distance.SetLabel('')
        a = self.selections[0].histograms()
        b = self.selections[1].histograms()
        if not (a and b):
            return
        if len(a) != len(b):
            return
        distance = 0.
        for i in range(len(a)):
            distance += histogramDistance(a[i], b[i])
        distance /= len(a)
        if (len(a) == 1):
            self.distance.SetLabel(' Distance = ' + str(distance))
        else:
            self.distance.SetLabel(' Average distance = ' + str(distance))

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
    HistogramDistance(state, 'Histogram distace').Show()
    app.MainLoop()

if __name__ == '__main__':
    main()
