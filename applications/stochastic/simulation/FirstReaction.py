"""Implements the FirstReaction class."""

from Solver import Solver
from Event import makeTriggerTimeEvent

import heapq
import random

class FirstReaction(Solver):

    def __init__(self, model, maxSteps):
        Solver.__init__(self, model, maxSteps)

    def initialize(self):
        # Initialize the state.
        Solver.initialize(self)
        # Compute the initial propensities and the initial time to the first
        # reaction.
        self.computeTimeToFirstReaction()

    def simulate(self, endTime):
        while(self.step(endTime)):
            pass

    def step(self, endTime):
        reactionTime = self.model.time + self.timeToFirstReaction

        # The time at the end of the next step or frame.
        tau = min(min(reactionTime, self.eventQueue[0][0]), endTime)
        # Check the trigger events.
        for e in self.model.triggerEvents.values():
            if e.evaluate(tau):
                t = self.model.time + e.delay
                if e.useValuesFromTriggerTime:
                    heapq.heappush(self.eventQueue,
                                   (t, makeTriggerTimeEvent(e)))
                else:
                    heapq.heappush(self.eventQueue, (t, e))
        
        # If we have reached the end time.
        if (reactionTime > endTime and self.eventQueue[0][0] > endTime):
            # Advance the time.
            self.timeToFirstReaction -= endTime - self.model.time
            self.model.time = endTime
            return False

        # Check that we have not exceeded the allowed number of steps.
        self.incrementStepCount()

        # If the reaction happens before the next event.
        if reactionTime < self.eventQueue[0][0]:
            # Fire the reaction.
            self.model.time += self.timeToFirstReaction
            self.firstReaction.fire()
        else:
            # Fire the event.
            self.model.time, e = heapq.heappop(self.eventQueue)
            e.fire()
            
        # Compute the time to the next reaction.
        self.computeTimeToFirstReaction()
        return True

    def computeTimeToFirstReaction(self):
        # Start with infinity.
        self.timeToFirstReaction = float('inf')
        self.firstReaction = None
        for id in self.model.reactions:
            r = self.model.reactions[id]
            propensity = r.propensity()
            if (propensity == 0):
                continue
            t = random.expovariate(propensity)
            if (t < self.timeToFirstReaction):
                self.timeToFirstReaction = t
                self.firstReaction = r
