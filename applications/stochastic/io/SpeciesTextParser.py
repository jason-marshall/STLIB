"""Parses a text representation of species."""

if __name__ == '__main__':
    import sys
    sys.path.insert(1, '..')

from state.Species import Species

import re

class SpeciesTextParser:
    """Parses a text representation of species."""
    
    def __init__(self):
        self.errorMessage = ''

    def parse(self, identifier, initialAmount, name, compartment,
              identifiers, label=''):
        """identifier, initialAmount, name, and compartment are strings.
        If these can be parsed, define self.name, 
        self.initialAmount, and self.compartment and return True."""
        self.errorMessage = ''

        # Check that the ID string is valid.
        matchObject = re.match('([a-zA-z]|_)[a-zA-Z0-9_]*', identifier)
        if not (matchObject and matchObject.group(0) == identifier):
            self.errorMessage = 'Species ' + label +\
                ' has a bad identifier: ' + identifier + '\n' +\
                'Identifiers must begin with a letter or underscore and\n'+\
                'be composed only of letters, digits, and underscores.'
            return False
        # Check that the ID is distinct.
        if identifier in identifiers:
            self.errorMessage = 'Species ' + label +\
                ' has a duplicate identifier: ' + identifier + '\n' +\
                'The identifiers must be distinct.'
            return False

        # Check that the initial amount string is not empty.
        self.initialAmount = initialAmount
        if not self.initialAmount:
            self.errorMessage = 'Species ' + label +\
                ' has an empty initial amount.'
            return False

        # No need to check the name. It can be anything.
        self.name = name

        # The compartment identifier must be empty or a valid identifier.
        self.compartment = compartment
        # Check that the compartment ID string is valid.
        if compartment:
            matchObject = re.match('([a-zA-z]|_)[a-zA-Z0-9_]*', compartment)
            if not (matchObject and matchObject.group(0) == compartment):
                self.errorMessage = 'Species ' + label +\
                    ' has a bad compartment identifier: ' + identifier + '\n' +\
                    'Identifiers must begin with a letter or underscore and\n'+\
                    'be composed only of letters, digits, and underscores.'
                return False

        return True

    def parseTable(self, table, identifiers):
        """Return a list of the species identifiers and a dictionary of the
        species."""
        self.errorMessage = ''

        if not table:
            self.errorMessage = 'There are no species.'
            return [], {}

        speciesIdentifiers = []
        species = {}
        count = 1
        for row in table:
            assert len(row) == 4
            if not self.parse(row[0], row[1], row[2], row[3],
                              speciesIdentifiers, str(count)):
                return [], {}
            # Record the species.
            speciesIdentifiers.append(row[0])
            species[row[0]] = Species(self.compartment, self.name,
                                      self.initialAmount)
            count = count + 1
            identifiers.append(row[0])

        return speciesIdentifiers, species

def main():
    parser = SpeciesTextParser()
    assert parser.parse('_', '0', '', '__', [])
    assert parser.parse('_a', '2', 'b', 'c1', [])
    assert parser.parse('alpha_3', '100000', 'neat stuff', 'c1', [])
    assert parser.parse('s1', '0', '', '', [])
    assert parser.parse('a', 'a', '', 'c1', [])
    assert parser.parse('a', '1.2', '', 'c1', [])
    assert parser.parse('a', '-1', '', 'c1', [])
    assert not parser.parse('', '0', '', 'c1', [])
    assert not parser.parse('3a', '0', '', 'c1', [])
    assert not parser.parse('a b', '0', '', 'c1', [])
    assert not parser.parse('a', '', '', 'c1', [])
    assert not parser.parse('a', '1', '', 'c1', ['a'])

    print parser.parseTable([['s1', '100', 'species 1', 'c1']], [])
    assert not parser.errorMessage

    print parser.parseTable([['s1', '100', 'species 1', 'c1'],
                             ['s2', '200', 'species 2', 'c1']], [])
    assert not parser.errorMessage

    print parser.parseTable([['1', '100', 'species 1', 'c1']], [])
    assert parser.errorMessage
    print parser.errorMessage

    print parser.parseTable([['s1', '100', 'species 1', 'c1'],
                             ['s1', '200', 'species 2', 'c1']], [])
    assert parser.errorMessage
    print parser.errorMessage

    print parser.parseTable([['s1', '100', 'species 1', 'c1']], [])
    assert not parser.errorMessage

if __name__ == '__main__':
    main()
