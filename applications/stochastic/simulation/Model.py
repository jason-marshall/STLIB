"""Implements the Model class."""

import math
import re
import string

class Model:
    """The model describes the parameters, compartments, species, and
    reactions."""

    def __init__(self, startingTime):
        """
        Make an empty model.

        Member data. For each, the keys are the identifiers.
        - species: A dictionary of species.
        - reactions: A dictionary of reactions.
        - timeEvents: A dictionary of time events.
        - triggerEvents: A dictionary of trigger events.
        - parameters: A dictionary of parameters. Note that the compartment
        volumes are grouped with the parameters.
        """
        # The simulation starting time.
        self.startingTime = startingTime
        self.time = None
        self.species = {}
        self.reactions = {}
        self.timeEvents = {}
        self.triggerEvents = {}
        self.parameters = {}
        self.isInitialized = False

    def initialize(self):
        """Initialize the species populations, reaction counts, events,
        and parameters."""
        self.isInitialized = True
        self.time = self.startingTime
        for id in self.species:
            self.species[id].initialize()
        for id in self.reactions:
            self.reactions[id].initialize()
        for id in self.timeEvents:
            self.timeEvents[id].initialize()
        for id in self.triggerEvents:
            self.triggerEvents[id].initialize()
        for id in self.parameters:
            self.parameters[id].initialize()

    def decorate(self, expression):
        # All species and parameters must be added for decoration to work.
        assert self.isInitialized
        return re.sub('([a-zA-z]|_)[a-zA-Z0-9_]*', self.decorateIdentifierTime,
                      expression)

    def decorateExceptTime(self, expression):
        """Decorate for a binary function of the model and the time."""
        # All species and parameters must be added for decoration to work.
        assert self.isInitialized
        return re.sub('([a-zA-z]|_)[a-zA-Z0-9_]*', self.decorateIdentifier,
                      expression)

    def decorateIdentifier(self, matchObject):
        """Change parameters and species to the variable name, assuming that
        the Model variable is 'm'."""
        mo = matchObject.group()
        # Special case: the e in 1e-5 is not a parameter.
        n = matchObject.start()
        if mo == 'e' and n != 0 and matchObject.string[n-1] in\
               ['.'] + list(string.digits):
            return mo
        if mo in self.parameters:
            return "m.parameters['" + mo + "'].value"
        elif mo in self.species:
            return "m.species['" + mo + "'].amount"
        else:
            return mo

    def decorateIdentifierTime(self, matchObject):
        """Change parameters and species to the variable name, assuming that
        the Model variable is 'm'."""
        if matchObject.group() == 't':
            return 'm.time'
        else:
            return self.decorateIdentifier(matchObject)
