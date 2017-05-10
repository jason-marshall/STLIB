"""Implements functions for parameter evaluation."""

# If we are running the unit tests.
if __name__ == '__main__':
    import sys
    sys.path.insert(1, '..')

from io.MathematicaWriter import mathematicaForm

import re
import string
import math
# Import the math functions for use in evaluating expressions.
from math import *

def getParameters(expression, identifiers):
    parameters = []
    while True:
        matchObject = re.search('([a-zA-z]|_)[a-zA-Z0-9_]*', expression)
        if not matchObject:
            break
        id = matchObject.group()
        if id in identifiers:
            parameters.append(id)
        expression = expression[matchObject.end():]
    return list(set(parameters))
        
def getIdentifiers(expression):
    identifiers = []
    while True:
        matchObject = re.search('([a-zA-z]|_)[a-zA-Z0-9_]*', expression)
        if not matchObject:
            break
        identifiers.append(matchObject.group())
        expression = expression[matchObject.end():]
    return identifiers
        
# CONTINUE REMOVE
def checkIdentifiers(identifiers):
    """Check the identifiers. If they are valid return None, otherwise return
    an error message. The identifiers may not be objects defined in math:
    ['pow', 'cosh', 'ldexp', 'hypot', 'tan', 'asin', 'log', 'fabs', 'floor', 'sqrt', 'frexp', 'degrees', 'pi', 'log10', '__doc__', 'fmod', 'atan', '__file__', 'ceil', 'sinh', '__name__', 'cos', 'e', 'tanh', 'radians', 'sin', 'atan2', 'modf', 'exp', 'acos']
    Parameters:
    - identifiers is a list of the identifiers."""
    # Check that each is a valid SBML identifier.
    reserved = math.__dict__.keys()
    for id in identifiers:
        matchObject = re.match('([a-zA-z]|_)[a-zA-Z0-9_]*', id)
        if not (matchObject and matchObject.group(0) == id):
            return '"%s" is not a valid identifier.' % id
        if id in reserved:
            return '"%s" is a reserved word. You cannot use it as an identifier.' % id
    # Indicate that there were no errors.
    return None

class Mangler:
    def __init__(self, prefix, identifiers):
        self.prefix = prefix
        self.identifiers = identifiers

    def __call__(self, matchObject):
        m = matchObject.group()
        if m in self.identifiers:
            return self.prefix + m
        else:
            return m

def mangle(expression, prefix, identifiers):
    """Mangle the identifiers in the expression. Return the mangled
    expression."""
    mangler = Mangler(prefix, identifiers)
    return re.sub('([a-zA-z]|_)[a-zA-Z0-9_]*', mangler, expression)

class KineticLawDecorator:
    """Mangle each parameter identifier by adding a prefix. Change species
    identifiers to array elements."""
    def __init__(self, parameterPrefix, parameterIdentifiers, speciesArrayName,
                 speciesIdentifiers):
        self.parameterPrefix = parameterPrefix
        self.parameterIdentifiers = parameterIdentifiers
        self.speciesArrayName = speciesArrayName
        self.speciesIdentifiers = speciesIdentifiers

    def __call__(self, expression):
        return re.sub('([a-zA-z]|_)[a-zA-Z0-9_]*', self.decorateIdentifier,
                      expression)

    def decorateIdentifier(self, matchObject):
        m = matchObject.group()
        # Special case: the e in 1e-5 is not a parameter.
        n = matchObject.start()
        if m == 'e' and n != 0 and matchObject.string[n-1] in\
               ['.'] + list(string.digits):
            return m
        if m in self.parameterIdentifiers:
            return self.parameterPrefix + m
        elif m in self.speciesIdentifiers:
            return self.speciesArrayName + '[' +\
                str(self.speciesIdentifiers.index(m)) + ']'
        else:
            return m

class KineticLawDecoratorMathematica:
    """Change species identifiers to function evaluations."""
    def __init__(self, speciesIdentifiers):
        self.speciesIdentifiers = speciesIdentifiers

    def __call__(self, expression):
        # Change species identifiers to function evaluations.
        expression = re.sub('([a-zA-z]|_)[a-zA-Z0-9_]*',
                            self.decorateIdentifier, expression)
        # Fix the floating-point numbers.
        return re.sub(r'[0-9]+\.?[0-9]*(e|E)(\+|-)?[0-9]+',
                      self.decorateNumber, expression)

    def decorateIdentifier(self, matchObject):
        m = matchObject.group()
        if m in self.speciesIdentifiers:
            return m + '[t]'
        else:
            return m

    def decorateNumber(self, matchObject):
        return mathematicaForm(float(matchObject.group()))

class KineticLawDecoratorSbml:
    """Change parameter identifiers to parameter values.
    - parameters is the dictionary of parameters"""
    def __init__(self, parameters):
        self.parameters = parameters

    def __call__(self, expression):
        # Change species identifiers to function evaluations.
        return re.sub('([a-zA-z]|_)[a-zA-Z0-9_]*', self.decorateIdentifier,
                      expression)

    def decorateIdentifier(self, matchObject):
        m = matchObject.group()
        if m in self.parameters:
            return self.parameters[m].expression
        else:
            return m

def evaluateValues(parameters):
    """Evaluate the expressions in the dictionary of parameters and
    compartments. Return True if the evaluation is successful. Store
    the parameter values in the 'value' member data field."""
    # Check the identifiers.
    for id in parameters:
        matchObject = re.match('([a-zA-z]|_)[a-zA-Z0-9_]*', id)
        if not (matchObject and matchObject.group() == id):
            return id + ' is not a valid identifier.'
    # Start with null values.
    for id in parameters:
        parameters[id].value = None
    # The prefix for mangling parameters.
    prefix = '__p_'
    # Make a list of identifiers and mangled expressions.
    remaining = [(id, mangle(parameters[id].expression, prefix, 
                             parameters.keys())) for id in parameters]
    passes = 0
    while remaining:
        passes += 1
        if passes > len(parameters):
            remainingIds = [x[0] for x in remaining]
            return 'Could not evaluate the expressions for: ' +\
                ', '.join(remainingIds) + '.'
        for id, expression in remaining:
            # Try to evaluate the expression.
            try:
                value = eval(expression)
            except:
                continue
            # Try to convert it to a floating point number.
            try:
                value = float(value)
            except:
                return 'Could not convert the expression for "%s" to a floating point number.' % id
            # Record the value.
            parameters[id].value = value
            exec(prefix + id + ' = value')
            remaining.remove((id, expression))
    # Indicate that there were no errors.
    return None

def evaluateParametersOld(__parameters):
    """Evaluate the expressions in the dictionary of parameters. Return True
    if the evaluation is successful. Store the parameter values in the 
    'value' member data field."""
    # I hide the variables in a class to avoid collision with the parameter
    # identifiers.
    class __Namespace:
        pass
    __names = __Namespace()
    __names.parameters = __parameters
    # Check the identifiers.
    __names.error = checkIdentifiers(__names.parameters)
    if __names.error:
        return __names.error
    # Start with null values.
    for id in __names.parameters:
        __names.parameters[id].value = None
    # Evaluate the expressions.
    __names.remaining = __names.parameters.keys()
    __names.passes = 0
    while __names.remaining:
        __names.passes += 1
        if __names.passes > len(__names.parameters):
            return 'Could not evaluate the expressions for ' +\
                ', '.join(__names.remaining) + '.'
        for id in __names.remaining:
            # Try to evaluate the expression.
            try:
                __names.value = eval(__names.parameters[id].expression)
            except:
                continue
            # Try to convert it to a floating point number.
            try:
                __names.value = float(__names.value)
            except:
                return 'Could not convert the expression for "%s" to a floating point number.' % id
            # Record the value.
            __names.parameters[id].value = __names.value
            exec(id + ' = __names.value')
            __names.remaining.remove(id)
    # Indicate that there were no errors.
    return None

def evaluateInitialAmounts(species, parameters):
    """Evaluate the propensity factors for the mass action kinetic laws.
    The parameter values must be evaluated before using this in this function.
    If the evaluation is successful return None. Otherwise return an error 
    message.
    Parameters:
    - species: A dictionary of species.
    - parameters: The dictionary of parameters."""
    # The prefix for mangling parameters.
    prefix = '__p_'
    # Evaluate the parameters.
    for id in parameters:
        assert parameters[id].value is not None
        exec(prefix + id + " = parameters['" + id + "'].value")
    parameterIdentifiers = parameters.keys()
    # Evaluate the initial amounts for the species.
    for id in species:
        # Start with a null value.
        species[id].initialAmountValue = None
        try:
            species[id].initialAmountValue =\
                eval(mangle(species[id].initialAmount, prefix, 
                            parameterIdentifiers))
        except:
            return 'Could not evaluate the initial amount for species '\
                + id + '.'
        # This form of the conditional checks for nan and other wierdness.
        if not species[id].initialAmountValue >= 0:
            return 'Species ' + id + ' does not have a non-negative initial amount.'
    # Indicate that there were no errors.
    return None

def evaluatePropensityFactors(reactions, parameters):
    """Evaluate the propensity factors for the mass action kinetic laws.
    The parameter values must be evaluated before using this in this function.
    If the evaluation is successful return None. Otherwise return an error 
    message.
    Parameters:
    - reactions: A list of reactions.
    - parameters: The dictionary of parameters."""
    # The prefix for mangling parameters.
    prefix = '__p_'
    # Evaluate the parameters.
    for id in parameters:
        assert parameters[id].value is not None
        exec(prefix + id + " = parameters['" + id + "'].value")
    parameterIdentifiers = parameters.keys()
    # Evaluate the propensity factors for the mass action kinetic laws.
    for r in reactions:
        # Start with a null value.
        r.propensityFactor = None
        if r.massAction:
            try:
                r.propensityFactor = eval(mangle(r.propensity, prefix, 
                                                 parameterIdentifiers))
            except:
                return 'Could not evaluate the propensity factor for reaction '\
                    + r.id + '.'
            # This form of the conditional checks for nan and other wierdness.
            if not r.propensityFactor >= 0:
                return 'Reaction ' + r.id + ' has a negative propensity factor.'
    # Indicate that there were no errors.
    return None

def makeValuesDictionary(model):
    values = {}
    for k, v in model.parameters.iteritems():
        values[k] = v
    for k, v in model.compartments.iteritems():
        values[k] = v
    return values

def evaluateModel(model):
    """Evaluate the parameters, the species initial amounts, and the
    reaction propensities for the mass action kinetic laws. Return
    None if successful. Otherwise return an error message."""
    values = makeValuesDictionary(model)
    return evaluateValues(values) or\
        evaluateInitialAmounts(model.species, values) or\
        evaluatePropensityFactors(model.reactions, values)

def evaluateModelInhomogeneous(model):
    """Evaluate the parameters and the species initial amounts. Return
    None if successful. Otherwise return an error message."""
    values = makeValuesDictionary(model)
    return evaluateValues(values) or\
        evaluateInitialAmounts(model.species, model.parameters)


def main():
    from Value import Value

    # The unit tests are in Species.py, Reaction.py, and Model.py.
    print mangle('a b c x y z', '__', ['a', 'b', 'c'])
    print mangle('a b c aa x y z', '__', ['a', 'b', 'c'])
    print mangle('(a+b-c)*aa**x(y)/z', '__', ['a', 'b', 'c'])

    print '\nKineticLawDecorator: s1 and s2 are species.'
    decorator = KineticLawDecorator('__p_', ['a', 'b', 'e'], 'x', ['s1', 's2'])

    expression = '6.42e-5'
    print expression
    print decorator(expression)

    print '\nKineticLawDecoratorMathematica: s1 and s2 are species.'
    decorator = KineticLawDecoratorMathematica(['s1', 's2'])

    expression = '0.5*s1'
    print expression
    print decorator(expression)

    expression = '1e-10*s1*s2'
    print expression
    print decorator(expression)

    expression = '1.23e-10'
    print expression
    print decorator(expression)

    expression = '1.23e10'
    print expression
    print decorator(expression)

    # CONTINUE: Make this work.
    expression = 'sqrt(s1)'
    print expression
    print decorator(expression)

    print '\nKineticLawDecoratorSbml: c1 and c2 are parameters.'
    decorator = KineticLawDecoratorSbml({'c1': Value('', '1'),
                                         'c2': Value('', '2')})

    expression = 'c1*s1'
    print expression
    print decorator(expression)

    expression = 'c1/c2*sqrt(c3)*s1'
    print expression
    print decorator(expression)


if __name__ == '__main__':
    main()
