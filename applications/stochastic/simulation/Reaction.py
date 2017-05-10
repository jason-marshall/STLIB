"""Implements the Reaction class."""

from SpeciesReference import SpeciesReference
from math import *

class Reaction:
    """A reaction is comprised of an identifier, name, reactants, products,
    mass action indicator, and propensity function. 
    Member data:
    - self.reactants: List of SpeciesReference.
    - self.products: List of SpeciesReference."""
    
    def __init__(self, model, reactants, products, propensityExpression):
        self.model = model
        # A list of SpeciesReference.
        self.reactants = reactants
        # A list of SpeciesReference.
        self.products = products
        # The string that defines the propensity function.
        self.propensityExpression = propensityExpression

    def propensity(self):
        return self.propensityFunction(self.model)

    def initialize(self):
        self.count = 0
        # Decorate the expression here so that we don't have to do it when
        # the reaction fires. Note that we can't do this in the constructor
        # because the propensity expression may depend on parameters that would
        # not yet have been defined.
        e = self.model.decorate(self.propensityExpression)
        self.propensityFunction = lambda m: eval(e)

    def fire(self):
        self.count += 1
        for x in self.reactants:
            self.model.species[x.species].amount -= x.stoichiometry
        for x in self.products:
            self.model.species[x.species].amount += x.stoichiometry

