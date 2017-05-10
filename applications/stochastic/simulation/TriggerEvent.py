"""Implements the TriggerEvent class."""

from Event import Event
from math import *

class TriggerEvent(Event):

    def __init__(self, model, assignmentString, triggerExpression, delay,
                 useValuesFromTriggerTime):
        Event.__init__(self, model, assignmentString)
        self.triggerExpression = triggerExpression
        self.delay = delay
        self.useValuesFromTriggerTime = useValuesFromTriggerTime
        
    def initialize(self):
        Event.initialize(self)
        e = self.model.decorateExceptTime(self.triggerExpression)
        self.trigger = lambda m, t: eval(e)
        self.value = self.trigger(self.model, self.model.time)

    def evaluate(self, time):
        oldValue = self.value
        self.value = self.trigger(self.model, time)
        # Return True if the trigger became true.
        return not oldValue and self.value
