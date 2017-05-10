"""Implements the SpeciesReference class."""

class SpeciesReference:
    """Comprised of a species identifier and the stoichiometry.

    The stoichiometry is the multiplicity of the species in the reaction."""
    
    def __init__(self, species, stoichiometry=1):
        """Construct from the species identifier and the stoichiometry."""
        # The species identifier is a string.
        self.species = species
        # The stoichiometry is an integer.
        self.stoichiometry = stoichiometry

    def hasErrors(self, identifiers):
        """Return None if the reaction term is valid. Otherwise return an error
        message."""
        if not self.species in identifiers:
            return 'Invalid species identifier.'
        if self.stoichiometry != int(self.stoichiometry):
            return 'Non-integer value for the stoichiometry.'
        if self.stoichiometry < 1:
            return 'Non-positive value for the stoichiometry.'
        return None

    def writeXml(self, writer):
        attributes = {'species': self.species}
        if self.stoichiometry != 1:
            attributes['stoichiometry'] = str(self.stoichiometry)
        writer.writeEmptyElement('speciesReference', attributes)

def main():
    import sys
    sys.path.insert(1, '..')
    from io.XmlWriter import XmlWriter

    writer = XmlWriter()
    
    identifiers = ['s1', 's2']
    x = SpeciesReference('s1')
    x.writeXml(writer)

    x = SpeciesReference('s1', 1)
    x.writeXml(writer)

    x = SpeciesReference('s2', 2)
    x.writeXml(writer)

if __name__ == '__main__':
    main()
