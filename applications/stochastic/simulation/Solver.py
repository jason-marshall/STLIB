"""Implements the Solver class."""

import heapq

class Solver:
    def __init__(self, model, maxSteps):
        self.model = model
        self.maxSteps = maxSteps
        # A null value indicates that there is no limit.
        if not self.maxSteps:
            self.maxSteps = 2**64
        # Force initialization
        self.stepCount = None

    def initialize(self):
        self.model.initialize()
        # Add the time events to the event queue.
        self.eventQueue = []
        for e in self.model.timeEvents.values():
            for t in e.times:
                self.eventQueue.append((t, e))
        # Add a null event at time = infinity, so we don't need to check if
        # the queue is empty.
        self.eventQueue.append((float('inf'), None))
        heapq.heapify(self.eventQueue)
        self.stepCount = 0

    def incrementStepCount(self):
        self.stepCount += 1
        if self.stepCount > self.maxSteps:
            raise Exception('The maximum step count of ' + str(self.maxSteps) +
                            ' has been exceeded.')
