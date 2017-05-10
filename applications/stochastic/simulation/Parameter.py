"""Implements the Parameter class."""

class Parameter:
    """These are meant to be stored in a dictionary with the parameter
    identifiers as the keys. Parameters have the following member data:
    - self.value is the floating point value of the parameter.
    - self.initialValue is the initialValue of the parameter."""
    
    def __init__(self, initialValue):
        r"""
        Construct a parameter.
        >>> from Parameter import Parameter
        >>> p = Parameter(7)
        >>> p.initialValue
        7.0
        >>> p.value is None
        True
        """
        self.initialValue = float(initialValue)
        # Enforce initialization.
        self.value = None

    def initialize(self):
        r"""
        >>> from Parameter import Parameter
        >>> p = Parameter(11)
        >>> p.initialize()
        >>> p.value
        11.0
        """
        self.value = self.initialValue
