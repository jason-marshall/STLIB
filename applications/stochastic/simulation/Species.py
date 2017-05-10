"""Implements the Species class."""

class Species:
    """A species has an amount and an initial amount.
    Member data:
    - self.amount: The amount of the species in substance units.
    - self.initialAmount: The initial amount."""
    
    def __init__(self, initialAmount):
        """Construct from the initial amount.
        >>> from Species import Species
        >>> x = Species(1.)
        >>> x.initialAmount
        1.0
        >>> x.amount is None
        True
        """
        self.initialAmount = float(initialAmount)
        # Force initialization.
        self.amount = None

    def initialize(self):
        """
        >>> from Species import Species
        >>> x = Species(1.)
        >>> x.amount is None
        True
        >>> x.initialize()
        >>> x.amount
        1.0
        """
        self.amount = self.initialAmount

