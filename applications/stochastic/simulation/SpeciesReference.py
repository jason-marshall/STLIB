"""Implements the SpeciesReference class."""

class SpeciesReference:
    """Comprised of a species identifier and the stoichiometry.

    The stoichiometry is the multiplicity of the species in the reaction."""
    
    def __init__(self, species, stoichiometry):
        """
        Construct from the species identifier and the stoichiometry.
        >>> from SpeciesReference import SpeciesReference
        >>> x = SpeciesReference('a', 2)
        >>> x.species
        'a'
        >>> x.stoichiometry
        2
        """
        # The species identifier is a string.
        self.species = species
        # The stoichiometry is an integer.
        self.stoichiometry = stoichiometry
