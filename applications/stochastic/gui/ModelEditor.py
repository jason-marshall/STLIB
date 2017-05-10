"""A model editor."""

import wx
import sys
import os, os.path

from GridRowEditor import GridRowEditor, EVT_FIRST_COLUMN_MODIFIED
from StateModified import StateModified, EVT_STATE_MODIFIED

class SpeciesOrReactionsModifiedEvent(wx.PyCommandEvent):
    """Event that is processed when the species or reaction identifiers are
    modified."""
    def __init__(self, evtType, id):
        wx.PyCommandEvent.__init__(self, evtType, id)

# Generate an event type.
EVT_SPECIES_OR_REACTIONS_MODIFIED_TYPE = wx.NewEventType()
# Create a binder object.
EVT_SPECIES_OR_REACTIONS_MODIFIED =\
    wx.PyEventBinder(EVT_SPECIES_OR_REACTIONS_MODIFIED_TYPE, 1)

class ModelEditor(wx.SplitterWindow, StateModified):
    """The model editor.

    I could not figure out how to get the four editors to pass events up to 
    the application. (Skip does not do it). Thus I catch state modified events
    in this class and then process the same event to pass it to the
    application."""
    
    def __init__(self, parent):
        # Splitter for species/reactions and parameters/compartments.
        wx.SplitterWindow.__init__(self, parent,
                                   style=wx.SP_NOBORDER|wx.SP_3DSASH)

        # The splitters.
        self.top = wx.SplitterWindow(self, style=wx.SP_NOBORDER|wx.SP_3DSASH)
        self.topLeft = wx.SplitterWindow(self.top,
                                         style=wx.SP_NOBORDER|wx.SP_3DSASH)
        self.bottom = wx.SplitterWindow(self, style=wx.SP_NOBORDER|wx.SP_3DSASH)
        self.bottomLeft = wx.SplitterWindow(self.bottom,
                                            style=wx.SP_NOBORDER|wx.SP_3DSASH)
        self.speciesEditor = \
            GridRowEditor(self.topLeft, 'Species',
                          ['ID', 'Amount', 'Name', 'Compartment'],
                          [80, 80, 80, 100], details=[2, 3],
                          toolTip='The species editor shows a list of the species in the selected model.')
        self.Bind(EVT_FIRST_COLUMN_MODIFIED, self.onFirstColumnModified,
                  self.speciesEditor)
        self.Bind(EVT_STATE_MODIFIED, self.onStateModified, self.speciesEditor)

        self.reactionsEditor = \
            GridRowEditor(self.topLeft, 'Reactions',
                          ['ID', 'Reactants', 'Products', 'MA',
                           'Propensity', 'Name'], [80, 80, 80, 25, 120, 80],
                          boolean=[(3, '1')], details=[5],
                          toolTip='The reactions editor shows a list of the reactions in the selected model.')
        self.Bind(EVT_FIRST_COLUMN_MODIFIED, self.onFirstColumnModified,
                  self.reactionsEditor)
        self.Bind(EVT_STATE_MODIFIED, self.onStateModified,
                  self.reactionsEditor)

        self.compartmentsEditor = \
            GridRowEditor(self.top, 'Compartments',
                          ['ID', 'Size', 'Name'], [80, 120, 80], details=[2],
                          toolTip='Defining compartments is optional. If no compartments are defined then all species will be placed in a default compartment.')
        self.Bind(EVT_STATE_MODIFIED, self.onStateModified,
                  self.compartmentsEditor)

        # CONTINUE: Transfer tooltips.
        if False:
            parametersAndCompartments = wx.Notebook(self, -1)
            # CONTINUE: I want to display this information for the tabs, not on
            # the whole window. Currently this does not work for linux.
            if sys.platform in ('darwin', 'win32', 'win64'):
                parametersAndCompartments.SetToolTip(wx.ToolTip('Defining parameters or compartments is optional. The parameters editor shows a list of the parameters (constants) in the selected model. If no compartments are defined then all species will be placed in a default compartment.'))
                
        self.timeEventsEditor = \
            GridRowEditor(self.bottomLeft, 'Time events',
                          ['ID', 'Times', 'Assignments', 'Name'],
                          [80, 120, 120, 80], details=[3],
                          toolTip='The time events are executed at specified points in time.')
        self.Bind(EVT_STATE_MODIFIED, self.onStateModified,
                  self.timeEventsEditor)
        
        self.triggerEventsEditor = \
            GridRowEditor(self.bottomLeft, 'Trigger events',
                          ['ID', 'Trigger', 'Assignments', 'Delay', 'TT',
                           'Name'],
                          [80, 120, 120, 80, 25, 80], boolean=[(4, '')],
                          details=[5],
                          toolTip='These events are triggered when the "Trigger" predicate becomes true. The delay is optional. Values from the trigger time will be used if T.T. is checked.')
        self.Bind(EVT_STATE_MODIFIED, self.onStateModified,
                  self.triggerEventsEditor)
        
        self.parametersEditor = \
            GridRowEditor(self.bottom, 'Parameters',
                          ['ID', 'Value', 'Name'], [80, 120, 80], details=[2],
                          toolTip='Defining parameters is optional. The parameters editor shows a list of the parameters (constants) in the selected model.')
        self.Bind(EVT_STATE_MODIFIED, self.onStateModified,
                  self.parametersEditor)

        # Note: Order is important.
        self.components = [self.speciesEditor, self.reactionsEditor,
                           self.timeEventsEditor, self.triggerEventsEditor,
                           self.parametersEditor, self.compartmentsEditor]
        
        self.SetMinimumPaneSize(10)
        self.top.SetMinimumPaneSize(10)
        self.topLeft.SetMinimumPaneSize(10)
        self.bottom.SetMinimumPaneSize(10)
        self.bottomLeft.SetMinimumPaneSize(10)
        # Give twice as much expanding space to the top half.
        self.SetSashGravity(0.67)
        self.top.SetSashGravity(0.67)
        self.bottom.SetSashGravity(0.67)
        self.topLeft.SetSashGravity(0.5)
        self.bottomLeft.SetSashGravity(0.5)
        
        self.SplitHorizontally(self.top, self.bottom)
        self.top.SplitVertically(self.topLeft, self.compartmentsEditor)
        self.bottom.SplitVertically(self.bottomLeft, self.parametersEditor)
        self.topLeft.SplitVertically(self.speciesEditor, self.reactionsEditor)
        self.bottomLeft.SplitVertically(self.timeEventsEditor,
                                        self.triggerEventsEditor)
        # For unknown reasons I need to call this twice for it to work.
        # CONTINUE: Check again.
        wx.CallAfter(self.setSashes)
        #wx.CallAfter(self.setSashes)

    def setSashes(self):
        width, height = self.GetSize()
        self.SetSashPosition(6*height//10)
        self.top.SetSashPosition(3*width//4)
        self.bottom.SetSashPosition(3*width//4)
        self.topLeft.SetSashPosition(width//4)
        self.bottomLeft.SetSashPosition(7*width//20)

    def getTableData(self):
        return [x.getTableData() for x in self.components]

    def setTableData(self, data):
        assert len(data) == len(self.components)
        for i in range(len(data)):
            self.components[i].setTableData(data[i])

    def getSpeciesAndReactionIdentifiers(self):
        """Return lists of the species identifiers and reaction identifiers."""
        return (self.speciesEditor.getNonEmptyCellsInFirstColumn(),
                self.reactionsEditor.getNonEmptyCellsInFirstColumn())

    def clear(self):
        for x in self.components:
            x.setTableData([])
            x.disable()

    def enable(self):
        for x in self.components:
            x.enable()

    def disable(self):
        for x in self.components:
            x.disable()

    def onFirstColumnModified(self, event):
        # Create the event.
        evt = SpeciesOrReactionsModifiedEvent\
            (EVT_SPECIES_OR_REACTIONS_MODIFIED_TYPE, self.GetId())
        # Process the event.
        self.GetEventHandler().ProcessEvent(evt)

    def onStateModified(self, event):
        """Pass a state modified event to the application."""
        self.processEventStateModified()


if __name__ == '__main__':
    class TestFrame(wx.Frame):
        def __init__(self, parent=None):
            wx.Frame.__init__(self, parent, title='Record',
                              size=(1024,600))
            self.panel = ModelEditor(self)
            # Bind the event.
            self.Bind(EVT_SPECIES_OR_REACTIONS_MODIFIED,
                      self.speciesOrReactionsModified, self.panel)

        def speciesOrReactionsModified(self, event):
            print('Species or reactions modified.',
                  self.panel.getSpeciesAndReactionIdentifiers())

    app = wx.PySimpleApp()
    frame = TestFrame()
    frame.Show()
    app.MainLoop()
