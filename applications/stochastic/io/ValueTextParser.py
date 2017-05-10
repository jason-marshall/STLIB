"""Parses a text representation of a Value."""

if __name__ == '__main__':
    import sys
    sys.path.insert(1, '..')

from state.Value import Value

import re

class ValueTextParser:
    """Parses a text representation of a Value."""
    
    def __init__(self, derived):
        self.errorMessage = ''
        # The name of the class derived from Value.
        assert derived in ('Parameter', 'Compartment')
        self.derived = derived

    def parse(self, identifier, expression, name, identifiers,
              label=''):
        """identifier, expression and name are strings.
        If these can be parsed, define self.name, 
        and self.expression and return True."""
        self.errorMessage = ''

        # Check that the ID string is valid.
        matchObject = re.match('([a-zA-z]|_)[a-zA-Z0-9_]*', identifier)
        if not (matchObject and matchObject.group(0) == identifier):
            self.errorMessage = self.derived + ' ' + label +\
                ' has a bad identifier: ' + identifier + '\n' +\
                'Identifiers must begin with a letter or underscore and\n'+\
                'be composed only of letters, digits, and underscores.'
            return False
        # Check that the ID is distinct.
        if identifier in identifiers:
            self.errorMessage = self.derived + ' ' + label +\
                ' has a duplicate identifier: ' + identifier + '\n' +\
                'The identifiers must be distinct.'
            return False

        # No need to check the name. It can be anything.
        self.name = name
        # Check that the expression string is not empty.
        self.expression = expression
        if not self.expression:
            self.errorMessage = self.derived + ' ' + label + ' has an empty value.'
            return False

        return True

    
    def parseTable(self, table, identifiers):
        """Return a dictionary of the parameters or compartments."""
        self.errorMessage = ''
        result = {}
        count = 1
        for row in table:
            assert len(row) == 3
            if not self.parse(row[0], row[1], row[2], identifiers, str(count)):
                return {}
            # Record the parameter or compartment.
            identifiers.append(row[0])
            result[row[0]] = Value(self.name, self.expression)
            count = count + 1
            identifiers.append(row[0])

        return result

def main():
    parser = ValueTextParser('Parameter')
    assert parser.parse('_a', '0', '', [])
    assert parser.parse('a', 'expression', '', [])
    assert not parser.parse('', 'expression', '', [])
    assert not parser.parse('3a', 'expression', '', [])
    assert not parser.parse('a b', 'expression', '', [])
    assert not parser.parse('a', '', '', [])
    assert not parser.parse('a', 'expression', '', ['a'])

    print parser.parseTable([['a', 'expression', '']], [])
    assert not parser.errorMessage

    print parser.parseTable([['', 'expression', '']], [])
    assert parser.errorMessage
    print parser.errorMessage

    print parser.parseTable([['a', '', '']], [])
    assert parser.errorMessage
    print parser.errorMessage

    print parser.parseTable([['a', 'expression', ''], ['a', 'expression', '']],
                            [])
    assert parser.errorMessage
    print parser.errorMessage

if __name__ == '__main__':
    main()
