"""Implements the Direct class."""

from Solver import Solver
from Event import makeTriggerTimeEvent

import heapq
import random
import numpy

class Direct(Solver):

    def __init__(self, model, maxSteps):
        Solver.__init__(self, model, maxSteps)
        self.sumOfPropensities = 0.
        self.propensities = numpy.zeros(len(model.reactions))
        self.reactionIdentifiers = self.model.reactions.keys()

    def initialize(self):
        # Initialize the state.
        Solver.initialize(self)
        # Compute the initial propensities and the initial time to the first
        # reaction.
        self.computeTimeToNextReaction()

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
            self.model.reactions[self.pickReaction()].fire()
        else:
            # Fire the event.
            self.model.time, e = heapq.heappop(self.eventQueue)
            e.fire()
            
        # Compute the time to the next reaction.
        self.computeTimeToNextReaction()
        return True

    def computeTimeToNextReaction(self):
        self.sumOfPropensities = 0.
        i = 0
        for id in self.model.reactions:
            self.propensities[i] = self.model.reactions[id].propensity()
            self.sumOfPropensities += self.propensities[i]
            i += 1
        if self.sumOfPropensities > 0:
            self.timeToFirstReaction =\
                                     random.expovariate(self.sumOfPropensities)
        else:
            self.timeToFirstReaction = float('inf')

    def pickReaction(self):
        """Pick the reaction to fire."""
        assert self.sumOfPropensities != 0
        # Loop until we chop-down search succeeds. (In rare cases it may fail
        # due to round-off error.)
        while True:
            s = self.sumOfPropensities * random.random()
            for i in range(len(self.propensities)):
                s -= self.propensities[i]
                if s <= 0:
                    # Invalidate the sum of propensities to make sure that we
                    # call this function only once after calling
                    # computeTimeToNextReaction().
                    self.sumOfPropensities = 0
                    # Return the identifier of the reaction.
                    return self.reactionIdentifiers[i]
