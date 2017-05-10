"""Implements the TimeEvent class."""

from Event import Event

class TimeEvent(Event):

    def __init__(self, model, assignmentString, times):
        Event.__init__(self, model, assignmentString)
        self.times = times
