"""Implements the Reaction class."""

# If we are running the unit tests.
if __name__ == '__main__':
    import sys
    sys.path.insert(1, '..')

from SpeciesReference import SpeciesReference
from io.MathematicaWriter import mathematicaForm

# Import the math functions for use in evaluatePropensityFactors().
from math import *
import sympy
import re

class Reaction:
    """A reaction is comprised of an identifier, name, reactants, products,
    mass action indicator, and propensity function. 
    Member data:
    - self.id: The reaction identifier string.
    - self.name: Optional descriptive name for the reaction.
    - self.reactants: List of SpeciesReference.
    - self.products: List of SpeciesReference.
    - self.massAction: Boolean value that indicates if the kinetic
    law uses mass action kinetics.
    - self.propensity: For mass action kinetics this string is a Python
    expression that evaluates to the propensity factor. For other
    kinetic laws, this expression is the propensity function in C++
    format. Either way, the expression may use the model parameters.
    - self.propensityFactor: For mass action kinetics this is a floating
    point number that is the leading factor in the propensity function.
    None indicates that its value has not been calculated.
    - self.reversible: Reversible reactions are not actually supported. 
    This is just used as an indication that this reaction should be replaced
    with two irreversible reactions.
        
    Consider the reaction 2 a -> b.  Suppose the model has parameters 'p'
    and 'q'. Below are some examples of mass action
    indicator, propensity strings, and the corresponding propensity functions.
    True, '3', 3 [a] ([a] - 1) / 2
    True, 'p*sqrt(q)', p sqrt(q) [a] ([a] - 1) / 2
    False, '5 * a', 5 [a]
    False, '5/p*a', 5 [a] / p
    False, '7', 7"""
    
    def __init__(self, id, name, reactants, products, massAction,
                 propensity):
        """Construct from the reactants and products."""
        self.id = id
        self.name = name
        # A list of SpeciesReference.
        self.reactants = reactants
        if self.reactants == None:
            self.reactants = []
        # A list of SpeciesReference.
        self.products = products
        if self.products == None:
            self.products = []
        self.massAction = massAction
        self.propensity = propensity
        self.propensityFactor = None
        self.reversible = False

    def order(self):
        """Return the order of the reaction."""
        order = 0
        for species in self.reactants:
            order += species.stoichiometry
        return order

    def simplify(self):
        """If the propensity is a mathematical expression that can be 
        evaluated, replace the propensity with its value."""
        try:
            value = float(eval(self.propensity))
        except:
            return
        self.propensity = repr(value)

    def influence(self, species):
        """Return the change in the species population that this reaction will
        affect."""
        change = 0
        for sr in self.reactants:
            if sr.species == species:
                change -= sr.stoichiometry
        for sr in self.products:
            if sr.species == species:
                change += sr.stoichiometry
        return change

    def hasErrors(self, identifiers, isDiscrete):
        """Return None if the reaction is valid. Otherwise return an error
        message.

        identifiers: A list of species identifiers.
        isDiscrete: True if this is a reaction is a stochastic model with 
        discrete species populations."""
        # The identifier must be non-null.
        if not self.id:
            return 'The identifier is empty.'
        # The reactants and products may not both be empty.
        if not self.reactants and not self.products:
            return 'The reactants and products are empty.'
        # Check each of the reactants.
        for reactant in self.reactants:
            error = reactant.hasErrors(identifiers)
            if error:
                return 'Error in reactants for reaction ' + self.id + '.\n' +\
                    error
        # Check each of the products.
        for product in self.products:
            error = product.hasErrors(identifiers)
            if error:
                return 'Error in products for reaction ' + self.id + '.\n' +\
                    error
        # No need to check the name.
        # Mass action indicator.
        if not (self.massAction == False or self.massAction == True):
            return 'Bad value for the mass action indicator.'
        # Only check that the propensity expression is not empty.
        if not self.propensity:
            return 'The propensity is empty.'
        if self.reversible:
            return 'Reversible reactions are not supported.'
        if isDiscrete and not self.massAction:
            # Check that if the propensity depends on a species then that 
            # species appears as a reactant. This is necessary to ensure that
            # propensities are recomputed appropriately.
            reactantIdentifiers = [_x.species for _x in self.reactants]
            used = re.findall(r'[a-zA-Z_]\w*', self.propensity)
            for id in used:
                if id in identifiers and not id in reactantIdentifiers:
                    return id + ' is used in the propensity function for a stochastic model but is not a reactant.\n' +\
                        'Add ' + id + ' as both a reactant and a product.'
        return None

    def massActionDenominator(self):
        """Return the denominator for the mass action propensity function."""
        denominator = 1.0
        for reactant in self.reactants:
            for i in range(1, reactant.stoichiometry + 1):
                denominator *= i
        return denominator

    def makeMassActionPropensityFunctionCoefficient(self):
        """Return the leading coefficient for the mass action 
        propensity function."""
        assert self.propensityFactor is not None
        return self.propensityFactor / self.massActionDenominator()

    def makeMassActionPropensityFunction(self, populationArrayName,
                                         speciesIdentifiers, isDiscrete):
        if isDiscrete:
            return self.makeMassActionPropensityFunctionDiscrete(\
                populationArrayName, speciesIdentifiers)
        else:
            return self.makeMassActionPropensityFunctionContinuous(\
                populationArrayName, speciesIdentifiers)

    def makePositiveReactantsCondition(self, populationArrayName,
                                       speciesIdentifiers):
        if self.reactants:
            return ' && '.join(['%s[%d]>0' % 
                                (populationArrayName, 
                                 speciesIdentifiers.index(r.species))
                                for r in self.reactants])
        else:
            return 'true'

    def makeMassActionPropensityFunctionDiscrete(\
        self, populationArrayName, speciesIdentifiers):
        """Make a string in C++ format for evaluating the discrete mass action 
        propensity function."""
        result = repr(self.makeMassActionPropensityFunctionCoefficient())
        for reactant in self.reactants:
            n = speciesIdentifiers.index(reactant.species)
            result += '*%s[%d]' % (populationArrayName, n)
            for i in range(1, reactant.stoichiometry):
                result += '*(%s[%d]-%d)' % (populationArrayName, n, i)
        return result

    def makeMassActionPropensityFunctionContinuous(\
        self, populationArrayName, speciesIdentifiers):
        """Make a string in C++ format for evaluating the continuous mass
        action propensity function."""
        result = repr(self.makeMassActionPropensityFunctionCoefficient())
        for reactant in self.reactants:
            n = speciesIdentifiers.index(reactant.species)
            result += ('*%s[%d]' % (populationArrayName, n)) *\
                reactant.stoichiometry
        return result

    def makeInhomogeneousMassActionPropensityFunction\
            (self, populationArrayName, speciesIdentifiers, isDiscrete):
        # Start with the rate function.
        result = ['(' + self.propensity + ')']
        # Multiply by the inverse of the mass action denominator.
        d = self.massActionDenominator()
        if d != 1:
            result.append('*' + repr(1./d))
        # Multiply by the appropriate species populations.
        for reactant in self.reactants:
            n = speciesIdentifiers.index(reactant.species)
            if isDiscrete:
                result.append('*%s[%d]' % (populationArrayName, n))
                for i in range(1, reactant.stoichiometry):
                    result.append('*(%s[%d]-%d)' % (populationArrayName, n, i))
            else:
                result.append(('*%s[%d]' % (populationArrayName, n)) *\
                                  reactant.stoichiometry)
        return ''.join(result)

    def makeMassActionPropensityFunctionMathematicaUnchecked(\
        self, speciesIdentifiers):
        return mathematicaForm\
            (self.makeMassActionPropensityFunctionCoefficient()) + ' ' +\
            ' '.join([((r.stoichiometry == 1 and '%s[t]' % r.species) or
                       '%s[t]^%d' % (r.species, r.stoichiometry))
                      for r in self.reactants])
            # This only works in Python 2.5 and beyond.
            #' '.join(['%s[t]' % r.species if r.stoichiometry == 1 else
            #             '%s[t]^%d' % (r.species, r.stoichiometry)
            #             for r in self.reactants])

    def makeMassActionPropensityFunctionMathematica(self, speciesIdentifiers):
        """Make a string in Mathematica format for evaluating the continuous
        mass action propensity function."""
        f = self.makeMassActionPropensityFunctionMathematicaUnchecked(
                speciesIdentifiers)
        if self.reactants:
            # Condition that all reactants have positive populations.
            result = 'If[' +\
                '&&'.join(['%s[t]>0' % r.species for r in self.reactants]) + ','
            # If true, return the propensity function.
            result += f
            # Otherwise, return 0. The reaction will not fire.
            result += ', 0]'
        else:
            # If there are no reactants, there is no need to check that that
            # the reactant populations are positive.
            result = f
        return result

    def makeMassActionTerm(self):
        """Make the reactant-dependent term in the mass action propensity
        function. This is used in convertCustomToMassAction()."""
        # Note that the resulting string will be used in sympy. Thus, it
        # doesn't need to be pretty.
        result = ['1']
        for reactant in self.reactants:
            for i in range(reactant.stoichiometry):
                result.append('*(%s-%s)' % (reactant.species, i))
        result += ['/', str(self.massActionDenominator())]
        return ''.join(result)

    def convertCustomToMassAction(self, species, parameters):
        """If this reaction has a custom propensity function, try to convert
        it to a mass action propensity function."""
        if self.massAction:
            return
        try:
            # Register the species and parameters.
            for x in species + parameters:
                exec("%s = sympy.Symbol('%s')" % (x, x))
            # The propensity function divided by the mass action term.
            exec("e = (%s)/(%s)" % (self.propensity, self.makeMassActionTerm()))
            # If the expression still depends on any species populations,
            # it is not a mass action rate law.
            p = str(e)
            for x in species:
                if re.search(x, p):
                    return
            self.propensity = p
            self.massAction = True
        except:
            pass

    def writeXml(self, writer):
        attributes = {'id': self.id, 'propensity': self.propensity}
        if self.name:
            attributes['name'] = self.name
        if self.massAction:
            attributes['massAction'] = 'true'
        else:
            attributes['massAction'] = 'false'
        writer.beginElement('reaction', attributes)
        if self.reactants:
            writer.beginElement('listOfReactants')
            for reactant in self.reactants:
                reactant.writeXml(writer)
            writer.endElement()
        if self.products:
            writer.beginElement('listOfProducts')
            for product in self.products:
                product.writeXml(writer)
            writer.endElement()
        writer.endElement() # reaction

    def readXml(self, attributes):
        """Read from an attributes dictionary. Return any errors encountered."""
        # The attribute "dictionary" may not work as it should. In particular
        # the test "x in attributes" may not work. Thus we need to directly 
        # use attributes.keys().
        keys = attributes.keys()
        if not 'id' in keys:
            return 'Missing id attribute in reaction.\n'
        self.id = attributes['id']
        if not 'massAction' in keys:
            return 'Missing massAction attribute in reaction.\n'
        self.massAction = bool(attributes['massAction'] == 'true')
        if 'name' in keys:
            self.name = attributes['name']
        else:
            self.name = ''
        if 'propensity' in keys:
            self.propensity = attributes['propensity']
        else:
            propensity = '0'
        self.propensityFactor = None
        self.reversible = False
        # The reactants and products are not defined by the attributes.
        return ''

    def writeSbml(self, writer):
        """Evaluate the propensity factors with evaluatePropensityFactors()
        before calling this function."""
        writer.beginElement('reaction', 
                            {'id': self.id, 'name': self.name,
                             'reversible': 'false'})

        if self.reactants:
            writer.beginElement('listOfReactants')
            for reactant in self.reactants:
                reactant.writeXml(writer)
            writer.endElement()

        if self.products:
            writer.beginElement('listOfProducts')
            for product in self.products:
                product.writeXml(writer)
            writer.endElement()

        # CONTINUE: I currently only support mass action kinetics with a 
        # numerical propensity factor.
        if self.massAction and self.propensityFactor:
            writer.beginElement('kineticLaw')
            writer.beginElement('math', 
                                {'xmlns': 'http://www.w3.org/1998/Math/MathML'})
            # If there are no reactants, the propensity function is a constant.
            if not self.reactants:
                # <cn> stands for Content Number.
                writer.writeElement('cn', {}, repr(self.propensityFactor))
            else:
                writer.beginElement('apply')
                writer.writeElement('times')
                # <cn> stands for Content Number.
                writer.writeElement('cn', {}, repr(self.propensityFactor))
                denominator = 1
                for reactant in self.reactants:
                    for n in range(reactant.stoichiometry):
                        denominator *= n + 1
                        if n == 0:
                            # <ci> stands for Content Identifier.
                            writer.writeElement('ci', {}, reactant.species)
                        else:
                            writer.beginElement('apply')
                            writer.writeElement('minus')
                            writer.writeElement('ci', {}, reactant.species)
                            writer.writeElement('cn', {}, str(n))
                            writer.endElement() # apply
                if denominator != 1:
                    writer.writeElement('cn', {},
                                        repr(1.0 / float(denominator)))
                writer.endElement() # apply
            writer.endElement() # math
            writer.endElement() # kineticLaw

        writer.endElement() # reaction

    def stringCmdlSpeciesReferenceList(self, species):
        """Make a CMDL string for the list of species references."""
        x = ''
        for speciesReference in species:
            for i in range(speciesReference.stoichiometry):
                x = x + speciesReference.species + '+'
        # Exclude the trailing '+'.
        if x:
            x = x[:-1]
        return x

    def stringCmdl(self):
        """Make a CMDL string for the reaction."""
        return self.stringCmdlSpeciesReferenceList(self.reactants) +\
            '->' +\
            self.stringCmdlSpeciesReferenceList(self.products)

def main():
    import sys

    sys.path.insert(1, '..')
    from io.XmlWriter import XmlWriter
    from ParameterEvaluation import evaluatePropensityFactors

    writer = XmlWriter()

    assert Reaction('r1', '', [], [], False, '').hasErrors([], True)

    # Mass action kinetics.

    identifiers = ['s1', 's2']
    a = SpeciesReference('s1', 1)
    x = Reaction('r1', '1', [a], [], True, '0')
    evaluatePropensityFactors([x], {})
    x.writeXml(writer)
    print(x.makeMassActionPropensityFunction('x', identifiers, True))
    print(x.makeMassActionPropensityFunction('x', identifiers, False))
    print(x.makeMassActionPropensityFunctionMathematica(identifiers))
    print('Mass action term:')
    print(x.makeMassActionTerm())
    print('')

    p = SpeciesReference('s2', 1)
    x = Reaction('r1', 'reaction', [a], [p], True, '1')
    evaluatePropensityFactors([x], {})
    x.writeXml(writer)
    print(x.makeMassActionPropensityFunction('x', identifiers, True))
    print(x.makeMassActionPropensityFunction('x', identifiers, False))
    print(x.makeMassActionPropensityFunctionMathematica(identifiers))
    print('')

    b = SpeciesReference('s2', 2)
    x = Reaction('r1', 'name', [a, b], [p], True, '3')
    evaluatePropensityFactors([x], {})
    x.writeXml(writer)
    print(x.makeMassActionPropensityFunction('x', identifiers, True))
    print(x.makeMassActionPropensityFunction('x', identifiers, False))
    print(x.makeMassActionPropensityFunctionMathematica(identifiers))
    print('')

    print('Time inhomogeneous')
    p = SpeciesReference('s2', 1)
    x = Reaction('r1', 'reaction', [a], [p], True, '2+sin(t)')
    x.writeXml(writer)
    print(x.makeInhomogeneousMassActionPropensityFunction
          ('x', identifiers, True))
    print(x.makeInhomogeneousMassActionPropensityFunction
          ('x', identifiers, False))
    print('')

    p = SpeciesReference('s2', 1)
    x = Reaction('r1', 'name', [a, b], [p], True, '2+sin(t)')
    x.writeXml(writer)
    print(x.makeInhomogeneousMassActionPropensityFunction
          ('x', identifiers, True))
    print(x.makeInhomogeneousMassActionPropensityFunction
          ('x', identifiers, False))
    print('')

    # Custom kinetics.

    x.massAction = False
    x.propensity = '5*s1*s2'
    x.writeXml(writer)
    print('')

    x.propensity = '5'
    x.writeXml(writer)
    print('')

    x.propensity = 's10'
    x.writeXml(writer)
    print('')

    identifiers.append('s3')
    x.propensity = 's3'
    x.writeXml(writer)
    print(x.hasErrors(identifiers, True))
    print('')

if __name__ == '__main__':
    main()
