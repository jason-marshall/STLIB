"""SBML reader.
This code is currently not used. I cannot get libsbml and wxPython to work
together. Thus I wrote my own SBML parser in ContentHandlerSbml."""

# If we are running the unit tests.
if __name__ == '__main__':
    import sys
    sys.path.insert(1, '..')

from state.Model import Model
from state.Species import Species
from state.SpeciesReference import SpeciesReference
from state.Reaction import Reaction
import libsbml

def readSbmlFile(fileName):
    # Create an empty model.
    model = Model()
    # Read the SBML file.
    print "Read the SBML file."
    print fileName
    #file = open(fileName)
    #print file.readlines()
    #file.close()

    #document = libsbml.SBMLReader().readSBML(fileName)
    # The SBML model.
    print 'The SBML model.'
    #sbmlModel = document.getModel()
    sbmlModel = libsbml.SBMLReader().readSBML(fileName).getModel()
    # Model identifier.
    model.id = sbmlModel.getId()
    # Model name.
    model.name = sbmlModel.getName()
    # For each species.
    for n in range(sbmlModel.getNumSpecies()):
        sbmlSpecies = sbmlModel.getSpecies(n)
        # Add to the dictionary of species.
        model.species[sbmlSpecies.getId()] = \
            Species(sbmlSpecies.getName(), str(sbmlSpecies.getInitialAmount()))
        # Add to the list of species identifiers.
	model.speciesIdentifiers.append(sbmlSpecies.getId())
    # For each reaction.
    for n in range(sbmlModel.getNumReactions()):
        sbmlReaction = sbmlModel.getReaction(n)
        reactants = []
        products = []
        # CONTINUE: reactants, products and modifiers may be listed multiple
        # times. In this case, accumulate the stoichiometry.
        for m in range(sbmlReaction.getNumReactants()):
            ref = sbmlReaction.getReactant(m)
            # CONTINUE: The stoichiometry may be defined with 
            # StoichiometryMath.
            reactants.append(SpeciesReference(ref.getSpecies(),
                                              int(ref.getStoichiometry())))
        for m in range(sbmlReaction.getNumProducts()):
            ref = sbmlReaction.getProduct(m)
            products.append(SpeciesReference(ref.getSpecies(),
                                             int(ref.getStoichiometry())))
        for m in range(sbmlReaction.getNumModifiers()):
            ref = sbmlReaction.getModifier(m)
            reactants.append(SpeciesReference(ref.getSpecies(), 1))
            products.append(SpeciesReference(ref.getSpecies(), 1))
        # Add the reaction.
        model.reactions.append(
            Reaction(sbmlReaction.getId(), sbmlReaction.getName(), reactants, 
                     products, True, '0'))
        # Add the reverse reaction if necessary.
        if sbmlReaction.getReversible():
            model.reactions.append(
                Reaction(sbmlReaction.getId() + 'r',
                         sbmlReaction.getName() + 'reverse',
                         products, reactants, True, '0'))
    return model



if __name__ == '__main__':
    if (len(sys.argv) != 2):
        print 'Usage:'
        print 'python SbmlReader.py file.xml'
        raise 'Bad command line arguments.'
    print sys.argv[1]
    model = readSbmlFile(sys.argv[1])
