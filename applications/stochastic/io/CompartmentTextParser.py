"""Parses a text representation of a compartment."""

if __name__ == '__main__':
    import sys
    sys.path.insert(1, '..')

from ValueTextParser import ValueTextParser

class CompartmentTextParser(ValueTextParser):
    """Parses a text representation of compartment."""
    
    def __init__(self):
        ValueTextParser.__init__(self, 'Compartment')
