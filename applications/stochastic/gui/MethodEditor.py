"""Implements the method editor."""

# If we are running the unit tests.
if __name__ == '__main__':
    import sys
    sys.path.insert(1, '..')

import wx

from StateModified import StateModified
import state.simulationMethods as simulationMethods

class MethodEditor(wx.Panel, StateModified):
    """The method editor."""
    
    def __init__(self, parent, application):
        """."""
        wx.Panel.__init__(self, parent)
        self.application = application

        # Create the text controls.
        ctrlSize=(80,-1)
        title = wx.StaticText(self, -1, 'Method Editor')
        title.SetToolTip(wx.ToolTip('Choose the solver with the pull-down menus on the left. Set the simulation parameters on the right.'))
        startTimeLabel = wx.StaticText(self, -1, 'Start time:')
        startTimeLabel.SetToolTip(wx.ToolTip('The simulation start time.'))
        self.startTime = wx.TextCtrl(self, size=ctrlSize)
        self.Bind(wx.EVT_TEXT, self.onModified, self.startTime)
        equilibrationTimeLabel = wx.StaticText(self, -1, 'Equil. time:')
        equilibrationTimeLabel.SetToolTip(wx.ToolTip('The length of time to simulate before recording.'))
        self.equilibrationTime = wx.TextCtrl(self, size=ctrlSize)
        self.Bind(wx.EVT_TEXT, self.onModified, self.equilibrationTime)
        recordingTimeLabel = wx.StaticText(self, -1, 'Rec. time:')
        recordingTimeLabel.SetToolTip(wx.ToolTip('The length of time to simulate and record results.'))
        # CONTINUE: Add a validator.
        self.recordingTime = wx.TextCtrl(self, size=ctrlSize)
        self.Bind(wx.EVT_TEXT, self.onModified, self.recordingTime)
        maximumStepsLabel = wx.StaticText(self, -1, 'Max steps:')
        maximumStepsLabel.SetToolTip(wx.ToolTip('The maximum number of allowed steps in a trajectory. Leave blank for no limit.'))
        self.maximumSteps = wx.TextCtrl(self, size=ctrlSize)
        self.Bind(wx.EVT_TEXT, self.onModified, self.maximumSteps)
        self.framesLabel = wx.StaticText(self, -1, "Frames:")
        self.framesLabel.SetToolTip(wx.ToolTip('The state is recorded at equally-spaced frames.'))
        self.binsLabel = wx.StaticText(self, -1, "Bins:")
        self.binsLabel.SetToolTip(wx.ToolTip('The number of bins in the histograms used to record the state.'))
        self.multiplicityLabel = wx.StaticText(self, -1, "Hist. mult.:")
        self.multiplicityLabel.SetToolTip(wx.ToolTip('Multiple histograms allow one to assess the error in the distributions.'))
        self.solverParameterLabel =\
            wx.StaticText(self, -1,
                          simulationMethods.parameterNames1[0][0][0][0])
        self.solverParameterLabel.SetToolTip\
            (wx.ToolTip(simulationMethods.parameterToolTips1[0][0][0][0]))

        self.frames = wx.SpinCtrl(self, value='11', size=(7*12, 2*12), min=1,
                                  max=1000000, initial=11)
        self.Bind(wx.EVT_TEXT, self.onModified, self.frames)
        self.bins = wx.SpinCtrl(self, value='32', size=(7*12, 2*12), min=2,
                                max=1000000, initial=32)
        self.Bind(wx.EVT_TEXT, self.onModified, self.bins)
        self.multiplicity = wx.SpinCtrl(self, value='4', size=(7*12, 2*12),
                                        min=1, max=1024, initial=4)
        self.Bind(wx.EVT_TEXT, self.onModified, self.multiplicity)
        self.solverParameter =\
            wx.TextCtrl(self,
                        value=simulationMethods.parameterValues1[0][0][0][0],
                        size=ctrlSize)
        self.Bind(wx.EVT_TEXT, self.onModified, self.solverParameter)

        # Collect so we can easily enable or disable them.
        self.controls = [self.startTime, self.equilibrationTime,
                         self.recordingTime, self.maximumSteps,
                         self.frames, self.bins, self.multiplicity,
                         self.solverParameter]

        # Choice controllers.
        self.timeDependence =\
            wx.Choice(self, choices=simulationMethods.timeDependence,
                      size=(240,-1))
        self.timeDependence.SetSelection(0)
        self.timeDependence.SetToolTip(wx.ToolTip('Select whether the solver allows explicit time dependence.'))
        self.category = wx.Choice(self, choices=simulationMethods.categories[0],
                                  size=(240,-1))
        self.category.SetSelection(0)
        self.category.SetToolTip(wx.ToolTip('Select the kind of output to generate.'))
        self.method = wx.Choice(self, choices=simulationMethods.methods[0][0],
                                size=(240,-1))
        self.method.SetSelection(0)
        self.method.SetToolTip(wx.ToolTip('Select the simulation method.'))
        self.options =\
            wx.Choice(self, choices=simulationMethods.options[0][0][0],
                      size=(240,-1))
        self.options.SetSelection(0)
        self.options.SetToolTip(wx.ToolTip('Select a solver option.'))

        self.Bind(wx.EVT_CHOICE, self.onTimeDependence, self.timeDependence)
        self.Bind(wx.EVT_CHOICE, self.onCategory, self.category)
        self.Bind(wx.EVT_CHOICE, self.onMethod, self.method)
        self.Bind(wx.EVT_CHOICE, self.onOptions, self.options)
        
        # Layout with sizers.
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(title, 0, wx.ALL, 0)
        sizer.Add(wx.StaticLine(self), 0, wx.EXPAND|wx.ALL, 1)


        left = wx.BoxSizer(wx.VERTICAL)
        for choice in [self.timeDependence, self.category, self.method,
                       self.options]:
            left.Add(choice, 0, wx.ALL, 1)

        entrySizer = wx.GridBagSizer(hgap=5, vgap=2)
        row = 0
        entrySizer.Add(startTimeLabel, (row, 0))
        entrySizer.Add(self.startTime, (row, 1))
        row += 1
        entrySizer.Add(equilibrationTimeLabel, (row, 0))
        entrySizer.Add(self.equilibrationTime, (row, 1))
        row += 1
        entrySizer.Add(recordingTimeLabel, (row, 0))
        entrySizer.Add(self.recordingTime, (row, 1))
        row += 1
        entrySizer.Add(maximumStepsLabel, (row, 0))
        entrySizer.Add(self.maximumSteps, (row, 1))
        row += 1
        entrySizer.Add(self.framesLabel, (row, 0))
        entrySizer.Add(self.frames, (row, 1))
        row += 1
        entrySizer.Add(self.binsLabel, (row, 0))
        entrySizer.Add(self.bins, (row, 1))
        row += 1
        entrySizer.Add(self.multiplicityLabel, (row, 0))
        entrySizer.Add(self.multiplicity, (row, 1))
        row += 1
        entrySizer.Add(self.solverParameterLabel, (row, 0))
        entrySizer.Add(self.solverParameter, (row, 1))

        content = wx.BoxSizer(wx.HORIZONTAL)
        content.Add(left, 0, wx.EXPAND|wx.ALL, 5)
        content.Add(entrySizer, 0, wx.EXPAND|wx.ALL, 5)

        sizer.Add(content, 0, wx.ALL, 0)

        self.SetSizer(sizer)
        sizer.Fit(self)
        self.onMethod(None)

    def _setOption(self, timeDependenceIndex, categoryIndex, methodIndex,
                   optionIndex):
        # Category.
        self.category.Clear()
        for item in simulationMethods.categories[timeDependenceIndex]:
            self.category.Append(item)
        self.category.SetSelection(categoryIndex)
        # Method.
        self.method.Clear()
        for item in simulationMethods.methods[timeDependenceIndex]\
                [categoryIndex]:
            self.method.Append(item)
        self.method.SetSelection(methodIndex)
        # Options.
        self.options.Clear()
        for item in simulationMethods.options[timeDependenceIndex]\
                [categoryIndex][methodIndex]:
            self.options.Append(item)
        self.options.SetSelection(optionIndex)
        # Set the label.
        self.setLabelForSelection(timeDependenceIndex, categoryIndex,
                                  methodIndex, optionIndex)

    def setMethod(self, timeDependenceIndex, categoryIndex, methodIndex,
                  optionIndex):
        self.timeDependence.SetSelection(timeDependenceIndex)
        self._setOption(timeDependenceIndex, categoryIndex, methodIndex,
                        optionIndex)

    def setLabel(self):
        self.setLabelForSelection(self.timeDependence.GetCurrentSelection(),
                                  self.category.GetCurrentSelection(),
                                  self.method.GetCurrentSelection(),
                                  self.options.GetCurrentSelection())

    def setLabelForSelection(self, timeDependenceIndex, categoryIndex,
                             methodIndex, optionIndex):
        label = simulationMethods.parameterNames1[timeDependenceIndex]\
            [categoryIndex][methodIndex][optionIndex]
        toolTip = simulationMethods.parameterToolTips1[timeDependenceIndex]\
            [categoryIndex][methodIndex][optionIndex]
        value = simulationMethods.parameterValues1[timeDependenceIndex]\
            [categoryIndex][methodIndex][optionIndex]
        self.solverParameterLabel.SetLabel(label)
        self.solverParameterLabel.SetToolTip(wx.ToolTip(toolTip))
        self.solverParameter.SetValue(value)
        if label:
            self.solverParameterLabel.Enable()
            self.solverParameter.Enable()
        else:
            self.solverParameterLabel.Disable()
            self.solverParameter.Disable()
        if simulationMethods.usesFrames(timeDependenceIndex, categoryIndex):
            self.framesLabel.Enable()
            self.frames.Enable()
        else:
            self.framesLabel.Disable()
            self.frames.Disable()
        if simulationMethods.categories[timeDependenceIndex][categoryIndex] in\
                ('Histograms, Transient Behavior', 'Histograms, Steady State'):
            self.binsLabel.Enable()
            self.bins.Enable()
            self.multiplicityLabel.Enable()
            self.multiplicity.Enable()
        else:
            self.binsLabel.Disable()
            self.bins.Disable()
            self.multiplicityLabel.Disable()
            self.multiplicity.Disable()
        # Indicate that the application state has been modified.
        self.processEventStateModified()

    def onTimeDependence(self, event):
        self._setOption(self.timeDependence.GetCurrentSelection(), 0, 0, 0)
        # Update the species and reactions that may be recorded.
        self.application.updateRecorded()
        # Update the launcher.
        self.application.syncSelectedMethod()
        self.application.launcher.update()
        
    def onCategory(self, event):
        self._setOption(self.timeDependence.GetCurrentSelection(),
                        self.category.GetCurrentSelection(), 0, 0)
        # Update the species and reactions that may be recorded.
        self.application.updateRecorded()
        # Update the launcher.
        self.application.syncSelectedMethod()
        self.application.launcher.update()
        
    def onMethod(self, event):
        self._setOption(self.timeDependence.GetCurrentSelection(),
                        self.category.GetCurrentSelection(),
                        self.method.GetCurrentSelection(), 0)
        # Update the launcher.
        self.application.syncSelectedMethod()
        self.application.launcher.update()
        
    def onOptions(self, event):
        self.setLabel()

    def onModified(self, event):
        self.processEventStateModified()


if __name__ == '__main__':
    class TestLauncher:
        def __init__(self):
            pass

        def update(self):
            pass

    class MethodEditorFrame(wx.Frame):
        def __init__(self, parent=None):
            wx.Frame.__init__(self, parent, title='Method Editor',
                              size=(420,250))
            self.launcher = TestLauncher()
            editor = MethodEditor(self, self)

        def syncSelectedMethod(self):
            pass

        def updateRecorded(self):
            pass

    app = wx.PySimpleApp()
    frame = MethodEditorFrame()
    frame.Show()
    app.MainLoop()
