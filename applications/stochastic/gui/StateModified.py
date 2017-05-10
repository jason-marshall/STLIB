"""Class that can indicate that the state has been modified."""

import wx

# Generate an event type.
EVT_STATE_MODIFIED_TYPE = wx.NewEventType()
# Create a binder object.
EVT_STATE_MODIFIED = wx.PyEventBinder(EVT_STATE_MODIFIED_TYPE, 1)

class EventStateModified(wx.PyCommandEvent):
    """Event that is processed when the state is modified."""
    def __init__(self, evtType, id):
        wx.PyCommandEvent.__init__(self, evtType, id)

class StateModified:
    """Class that can indicate that the state has been modified. This
    information is used on exit to warn the user about unsaved changes.
    Most of the panels in MainFrame inherit from this class."""
    def processEventStateModified(self):
        # The subclass must provide the GetEventHandler() and GetId() functions.
        # Create the event.
        event = EventStateModified(EVT_STATE_MODIFIED_TYPE, self.GetId())
        # Process the event. 
        self.GetEventHandler().ProcessEvent(event)

