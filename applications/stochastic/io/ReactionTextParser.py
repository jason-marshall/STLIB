"""Parses a text representation of a reaction."""

if __name__ == '__main__':
    import sys
    sys.path.insert(1, '..')

from state.SpeciesReference import SpeciesReference
from state.Reaction import Reaction

import string
import re

class ReactionTextParser:
    """Parses a text representation of a reaction."""
    
    def __init__(self):
        self.errorMessage = ''

    def parse(self, identifier, reactantsString, productsString, massAction,
              propensity, name, speciesIdentifiers, identifiers,
              label=''):
        """identifier, reactantsString, productsString, propensity and
        name are strings. If these can be parsed, define 
        self.reaction and return True."""
        self.errorMessage = ''

        # No need to check the name. It can be anything.
        # Check that the ID string is valid.
        matchObject = re.match('([a-zA-z]|_)[a-zA-Z0-9_]*', identifier)
        if not (matchObject and matchObject.group(0) == identifier):
            self.errorMessage = 'Reaction ' + label +\
                ' has a bad identifier: ' + identifier + '\n' +\
                'Identifiers must begin with a letter or underscore and\n'+\
                'be composed only of letters, digits, and underscores.'
            return False
        # Check that the ID is distinct.
        if identifier in identifiers:
            self.errorMessage = 'Reaction ' + label +\
                ' has a duplicate identifier: ' + identifier + '\n' +\
                'The identifiers must be distinct.'
            return False
        # Check the reactants.
        isValid, reactants = self.parseSpeciesReferenceList(reactantsString,
                                                            speciesIdentifiers)
        if not isValid:
            self.errorMessage = 'Reaction ' + label + ' has bad reactants.\n'
            return False
        # Check the products.
        isValid, products = self.parseSpeciesReferenceList(productsString,
                                                           speciesIdentifiers)
        if not isValid:
            self.errorMessage = 'Reaction ' + label + ' has bad products.\n'
            return False
        # A reaction must not have empty reactants and products.
        if not reactants and not products:
            self.errorMessage = 'Reaction ' + label +\
                ' has no reactants or products.\n'
            return False
        # If the propensity string is empty.
        if not propensity:
            self.errorMessage = 'Reaction ' + label +\
                ' has an empty propensity function.'
            return False
        # Build the reaction.
        self.reaction = Reaction(identifier, name, reactants, products, 
                                 massAction=='1', propensity)
        return True

    def merge(self, speciesReferenceList, speciesReference):
        # Check if the species appeared before.
        for x in speciesReferenceList:
            if x.species == speciesReference.species:
                x.stoichiometry += speciesReference.stoichiometry
                return
        # If not, add it to the end.
        speciesReferenceList.append(speciesReference)

    def parseSpeciesReferenceList(self, inputString, speciesIdentifiers):
        """Return the tuple: (isValid, speciesReferenceList)."""
        # If the string is only whitespace.
        if not inputString.strip():
            return True, []
        speciesReferenceList = []
        # For each species reference.
        for speciesReferenceString in inputString.split('+'):
            isValid, speciesReference =\
                self.parseSpeciesReference(speciesReferenceString.strip(),
                                           speciesIdentifiers)
            if not isValid:
                return False, []
            self.merge(speciesReferenceList, speciesReference)
        return True, speciesReferenceList

    def parseSpeciesReference(self, speciesReferenceString, speciesIdentifiers):
        """Parse the string and return the tuple: 
        (isValid, speciesReference)."""
        # Check for an empty string.
        if not speciesReferenceString:
            return False, None
        # If the string starts with an explicit stoichiometry.
        if speciesReferenceString[0] in string.digits:
            # Get the stoichiometry.
            matchObject = re.match('[0-9]*', speciesReferenceString)
            stoichiometry = int(matchObject.group(0))
            # Strip the stoichiometry.
            speciesReferenceString =\
                speciesReferenceString[len(matchObject.group(0)):]
            speciesReferenceString = speciesReferenceString.strip()
        else:
            stoichiometry = 1
        if stoichiometry < 1:
            return False, None
        # Get the species.
        if not speciesReferenceString in speciesIdentifiers:
            return False, None
        return True, SpeciesReference(speciesReferenceString, stoichiometry)

    def parseTable(self, table, speciesIdentifiers, identifiers):
        """Return a list of the reactions."""
        self.errorMessage = ''

        reactions = []
        count = 1
        for row in table:
            assert len(row) == 6
            if not self.parse(row[0], row[1], row[2], row[3], row[4], row[5],
                              speciesIdentifiers, identifiers,
                              str(count)):
                return []
            # Record the reaction.
            reactions.append(self.reaction)
            count = count + 1
            identifiers.append(row[0])

        return reactions

def main():
    parser = ReactionTextParser()
    assert parser.parse('r1', 's1 + 2 s2', 's3', '1', '2.5', 'reaction 1',
                        ['s1', 's2', 's3'], [])
    assert parser.reaction.id == 'r1'
    assert parser.reaction.name == 'reaction 1'
    assert parser.reaction.reactants[0].species == 's1'
    assert parser.reaction.reactants[0].stoichiometry == 1
    assert parser.reaction.reactants[1].species == 's2'
    assert parser.reaction.reactants[1].stoichiometry == 2
    assert parser.reaction.products[0].species == 's3'
    assert parser.reaction.products[0].stoichiometry == 1
    assert parser.reaction.propensity == '2.5'

    assert parser.parse('r1',  's1 + s1', 's3', '1', '2.5', 'reaction 1',
                        ['s1', 's2', 's3'], [])
    assert parser.reaction.reactants[0].species == 's1'
    assert parser.reaction.reactants[0].stoichiometry == 2

    assert parser.parse('r1', '2 s1 + 3 s1', 's3', '1', '2.5', 'reaction 1',
                        ['s1', 's2', 's3'], [])
    assert parser.reaction.reactants[0].species == 's1'
    assert parser.reaction.reactants[0].stoichiometry == 5

    assert not parser.parse('1', 's1 + 2 s2', 's3', '1', '2.5', 'reaction 1',
                            ['s1', 's2', 's3'], [])
    assert parser.errorMessage
    print parser.errorMessage
    print ''

    assert not parser.parse('r1', '1 + 2 s2', 's3', '1', '2.5', 'reaction 1',
                            ['s1', 's2', 's3'], [])
    assert parser.errorMessage
    print parser.errorMessage
    print ''

    assert not parser.parse('r1', 's1 - 2 s2', 's3', '1', '2.5', 'reaction 1',
                            ['s1', 's2', 's3'], [])
    assert parser.errorMessage
    print parser.errorMessage
    print ''

    assert not parser.parse('r1', 's1 + 2 s2', '3', '1', '2.5', 'reaction 1',
                            ['s1', 's2', 's3'], [])
    assert parser.errorMessage
    print parser.errorMessage
    print ''

    assert not parser.parse('r1', 's1 + 2 s2', 's3 + s4', '1', 'reaction 1',
                            '2.5', ['s1', 's2', 's3'], [])
    assert parser.errorMessage
    print parser.errorMessage
    print ''

    assert not parser.parse('r1', 's1 + 2 s2', 's3', '1', '', 'reaction 1',
                            ['s1', 's2', 's3'], [])
    assert parser.errorMessage
    print parser.errorMessage
    print ''

    # Does not check for negative numbers.
    assert parser.parse('r1', 's1 + 2 s2', 's3', '1', '-2.5', 'reaction 1',
                        ['s1', 's2', 's3'], [])

    assert not parser.parse('r1', 's1 + 2 s2', 's3', '1', '2.5', 'reaction 1',
                            ['s1', 's2', 's3'], ['r1'])
    assert parser.errorMessage
    print parser.errorMessage
    print ''

    assert parser.parseTable(
        [['r1', 's1 + 2 s2', 's3', '1', '2.5', 'reaction 1']],
        ['s1', 's2', 's3'], [])
    assert not parser.errorMessage

    assert parser.parseTable(
        [['r1', 's1 + 2 s2', 's3', '1', '2.5', 'reaction 1'],
         ['r2', 's1 + s2', 's3', '1', '2.5', 'reaction 2']],
        ['s1', 's2', 's3'], [])
    assert not parser.errorMessage

    assert not parser.parseTable(
        [['1', 's1 + 2 s2', 's3', '1', '2.5', 'reaction 1']],
        ['s1', 's2', 's3'], [])
    assert parser.errorMessage
    print parser.errorMessage
    print ''

    assert not parser.parseTable(
        [['r1', 's1 + 2 s2', 's3', '1', '2.5', 'reaction 1'],
         ['r1', 's1 + s2', 's3', '1', '2.5', 'reaction 2']],
        ['s1', 's2', 's3'], [])
    assert parser.errorMessage
    print parser.errorMessage
    print ''

if __name__ == '__main__':
    main()
