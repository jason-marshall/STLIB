"""Implements the Species class."""

class Species:
    """A species is comprised of a compartment, a name and an initial
    amount.
    Member data:
    - self.compartment: Identifier of a compartment.
    - self.name: Optional descriptive name.
    - self.initialAmount: String expression for the initial amount. It may use
    the model parameters in a Python expression.
    - self.initialAmountValue: Floating point number that is the initial 
    amount. None indicates that it has not been calculated."""
    
    def __init__(self, compartment, name, initialAmount):
        """Construct from the compartment, name, and initial population."""
        self.compartment = compartment
        self.name = name
        self.initialAmount = initialAmount
        self.initialAmountValue = None

    def hasErrors(self, compartmentIdentifiers):
        """Return None if the species is valid and the initial population
        expression is not empty. Otherwise return an error message."""
        # If the compartment is named, it must be in the list of compartment
        # identifiers.
        if self.compartment and not self.compartment in compartmentIdentifiers:
            return 'Bad compartment.'
        if not self.initialAmount:
            return 'Empty initial amount.'
        return None

    def writeXmlCommon(self, writer, id, attributes):
        attributes['id'] = id
        if self.name:
            attributes['name'] = self.name
        writer.writeEmptyElement('species', attributes)

    def writeXml(self, writer, id):
        """The initialAmount attribute is an expression."""
        attributes = {'initialAmount':self.initialAmount}
        # Record the compartment if it is not the unnamed compartment.
        if self.compartment:
            attributes['compartment'] = self.compartment
        self.writeXmlCommon(writer, id, attributes)

    def writeSbml(self, writer, id, unnamedCompartment):
        """The initialAmount attribute is a floating point number. Thus the
        initial amount expression must be evaluated before calling
        this function."""
        attributes = {'hasOnlySubstanceUnits':'true'}
        # Check that the inital amount expression has been evaluated.
        if self.initialAmountValue is not None:
            attributes['initialAmount'] = str(self.initialAmountValue)
        if self.compartment:
            attributes['compartment'] = self.compartment
        else:
            attributes['compartment'] = unnamedCompartment
        self.writeXmlCommon(writer, id, attributes)

    def readXml(self, attributes):
        """Read from an attributes dictionary. No error checking is 
        necessary."""
        # The attribute "dictionary" may not work as it should. In particular
        # the test "x in attributes" may not work. Thus we need to directly 
        # use attributes.keys().
        keys = attributes.keys()
        # If the compartment is not specified, use the unnamed compartment.
        if 'compartment' in attributes.keys():
            self.compartment = attributes['compartment']
        else:
            self.compartment = ''
        if 'name' in attributes.keys():
            self.name = attributes['name']
        else:
            self.name = ''
        if 'initialAmount' in attributes.keys():
            self.initialAmount = attributes['initialAmount']
        else:
            self.initialAmount = ''
        self.initialAmountValue = None

def main():
    import math
    import sys
    sys.path.insert(1, '..')

    from Value import Value
    from ParameterEvaluation import evaluateInitialAmounts
    from io.XmlWriter import XmlWriter
    writer = XmlWriter()

    x = Species('C1', 'species 1', '0')
    x.writeXml(writer, 's1')

    x = Species('C1', 'species 1', '')
    print evaluateInitialAmounts({'s1':x}, {})

    x = Species('C1', 'species 2', '1')
    x.writeXml(writer, 's2')

    x = Species('C1', 'species 2', '-1')
    # I don't check the initial amount expression with hasErrors().
    print evaluateInitialAmounts({'s2':x}, {})

    x = Species('C1', 'species 2', 'p1')
    print evaluateInitialAmounts({'s1':x}, {})

    x = Species('C1', 'species 2', 'p2')
    p = Value('', '5.0')
    p.value = 5.0
    print evaluateInitialAmounts({'s1':x}, {'p1':p})

if __name__ == '__main__':
    main()
