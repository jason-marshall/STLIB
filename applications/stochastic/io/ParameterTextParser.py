"""Parses a text representation of a parameter."""

if __name__ == '__main__':
    import sys
    sys.path.insert(1, '..')

from ValueTextParser import ValueTextParser

class ParameterTextParser(ValueTextParser):
    """Parses a text representation of parameter."""
    
    def __init__(self):
        ValueTextParser.__init__(self, 'Parameter')
