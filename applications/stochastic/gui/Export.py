"""Implements exporting of time series and histogram data."""

# If we are running the unit tests.
if __name__ == '__main__':
    import sys
    sys.path.insert(1, '..')

import ExportTimeSeries
import ExportHistogram
import wx

class Export(wx.Frame):
    """Hold the panels for exporting time series and histogram data
    in a notebook."""
    
    def __init__(self, parent, title, state):
        wx.Frame.__init__(self, parent, -1, title)

        panel = wx.Panel(self, -1)
        notebook = wx.Notebook(panel)

        self.timeSeries = ExportTimeSeries.Configuration(notebook, state)
        notebook.AddPage(self.timeSeries, 'Time Series')
        self.histograms = ExportHistogram.Configuration(notebook, state)
        notebook.AddPage(self.histograms, 'Histograms')
        
        sizer = wx.BoxSizer()
        sizer.Add(notebook, 1, wx.EXPAND)
        panel.SetSizer(sizer)

        bestSize = self.timeSeries.GetBestSize()
        # Add 80 to reserve space for the initially hidden standard deviation
        # field.
        size = (bestSize[0] + 80, 600)
        self.SetSize(size)
        # Intercept the close event.
        self.Bind(wx.EVT_CLOSE, self.onClose)

    def onClose(self, event):
        # If there is a parent, it stores a dictionary of these frames.
        if self.GetParent():
            del self.GetParent().children[self.GetId()]
        self.Destroy()

    def refresh(self):
        """Refresh the output choices in each tab."""
        self.timeSeries.refresh()
        self.histograms.refresh()

def main():
    from state.TimeSeriesFrames import TimeSeriesFrames
    from state.TimeSeriesAllReactions import TimeSeriesAllReactions
    from state.State import State
    from state.Model import Model
    from state.Reaction import Reaction
    from state.Species import Species
    from state.SpeciesReference import SpeciesReference

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
    Export(None, 'Populations.', state).Show()
    
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
    Export(None, 'Populations.', state).Show()

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
    Export(None, 'Populations.', state).Show()

    app.MainLoop()

if __name__ == '__main__':
    main()

