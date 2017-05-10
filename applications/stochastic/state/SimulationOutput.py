"""Implements the SimulationOutput class."""

def isSorted(x):
    """Return true if the sequence is sorted."""
    if len(x) == 0:
        return True
    for i in range(len(x) - 1):
        if x[i] > x[i + 1]:
            return False
    return True

class SimulationOutput:
    """A base class for simulation output."""
    
    def __init__(self, recordedSpecies, recordedReactions):
        # The recorded species.
        self.recordedSpecies = recordedSpecies
        # The recorded reactions.
        self.recordedReactions = recordedReactions

    def hasErrors(self):
        """Return None if valid. Otherwise return an error message."""
        if not self.recordedSpecies and not self.recordedReactions:
            return 'There are no recorded species or reactions.'
        return None
