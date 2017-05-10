"""Implements the Event class."""

from math import *

class Event:

    def __init__(self, model, assignmentString=''):
        self.model = model
        self.assignmentString = assignmentString

    def initialize(self):
        self.count = 0
        # Build the list of assignments.
        self.assignments = []
        if self.assignmentString:
            for assignment in self.assignmentString.split(';'):
                s = assignment.split('=')
                if len(s) != 2:
                    raise Exception('Bad assignment "' + assignment +
                                    '" in Event.')
                self.assignments.append((s[0].strip(),
                                         lambda m, e=self.model.decorate(s[1]):
                                         eval(e)))

    def fire(self):
        self.count += 1
        m = self.model
        for (id, f) in self.assignments:
            if id in m.species:
                m.species[id].amount = f(m)
            elif id in m.parameters:
                m.parameters[id].value = f(m)
            else:
                raise Exception('The identifier ' + id + ' in the Event'\
                                ' is neither a species nor a parameter.')

def makeTriggerTimeEvent(event):
    """From the specified event make a new event that assigns the values at
    the current time."""
    e = Event(event.model)
    e.initialize()
    for (id, f) in event.assignments:
        v = f(event.model)
        e.assignments.append((id, lambda m: v))
    return e
