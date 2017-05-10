"""Read a simple text representation of a model."""

# If we are running the unit tests.
if __name__ == '__main__':
    import sys
    sys.path.insert(1, '..')

from state.Model import Model

from state.Species import Species
from state.SpeciesReference import SpeciesReference
from state.Reaction import Reaction

def readModelText(input):
    """Read a model with the following format:
    <number of species>
    <number of reactions>
    <list of initial amounts>
    <packed reactions>
    <list of propensity factors>

    If successful, return the model. Otherwise return None.

    Note that the model ID and name will not be set as this information is 
    not in the simple text description.

    The species identifiers will be set to S1, S2, etc. The reaction
    identifiers will be set to R1, R2, etc. The names for species and 
    reactions will be empty.
    """
    try:
        # Start with an empty model. The ID and the name are empty. We will use
        # the unnamed compartment.
        model = Model()
        # Number of species.
        numberOfSpecies = int(input.readline())
        assert numberOfSpecies > 0
        # Number of reactions.
        numberOfReactions = int(input.readline())
        assert numberOfReactions > 0
        # Initial amounts. Read as strings.
        initialAmounts = input.readline().rstrip().split()
        assert len(initialAmounts) == numberOfSpecies
        # The species identifiers are S1, S2, etc.
        sid = ['S' + str(n + 1) for n in range(numberOfSpecies)]
        model.speciesIdentifiers = sid
        # Add the species.
        for n in range(numberOfSpecies):
            model.species[sid[n]] = Species('', '', initialAmounts[n])
        # Read the packed reactions into a list of integers.
        data = map(int, input.readline().rstrip().split())
        # The propensity factors.
        propensityFactors = input.readline().rstrip().split()
        # Add the reactions.
        n = 0
        for i in range(numberOfReactions):
            id = 'R' + str(i + 1)
            # Reactants.
            numberOfReactants = data[n]
            n = n + 1
            reactants = []
            for j in range(numberOfReactants):
                speciesIndex = data[n]
                n = n + 1
                stoichiometry = data[n]
                n = n + 1
                reactants.append(SpeciesReference(sid[speciesIndex],
                                                  stoichiometry))
            # Products.
            numberOfProducts = data[n]
            n = n + 1
            products = []
            for j in range(numberOfProducts):
                speciesIndex = data[n]
                n = n + 1
                stoichiometry = data[n]
                n = n + 1
                products.append(SpeciesReference(sid[speciesIndex],
                                                 stoichiometry))
            # Add the reaction.
            model.reactions.append(Reaction(id, '', reactants, products,
                                            True, propensityFactors[i]))
        # Return the model.
        return model
    except Exception, error:
        print(error)
        return None

def main():
    from state.Model import writeModelXml
    model = readModelText(open('DecayingDimerizing.txt'))
    model.id = 'DecayingDimerizing'
    assert model
    writeModelXml(model)

if __name__ == '__main__':
    main()
