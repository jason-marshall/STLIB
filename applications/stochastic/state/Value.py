"""Implements the Value class."""

# If we are running the unit tests.
if __name__ == '__main__':
    import sys
    sys.path.insert(1, '..')

class Value:
    """This is a base class for Parameter and Compartment. Both of these
    are stored in a dictionary with the 
    identifiers as the keys. Value has the following member data:
    - self.name is a string that gives a descriptive name.
    - self.expression is a string holding a Python expression that defines
    the value.
    - self.value is the floating point value of the expression. None indicates
    that it has not been calculated."""
    
    def __init__(self, name, expression):
        """Construct from the name and expression strings. The value is
        set to None to indicate that it has not been calculated."""
        self.name = name
        self.expression = expression
        self.value = None

    def hasErrors(self):
        """Return None if the parameter is valid. Otherwise return an error
        message."""
        # The name can be anything.
        # The expression must not be empty.
        if not self.expression:
            return 'Empty expression.'
        # Don't check the value.
        return None

    def writeXml(self, writer, tag, id):
        """Write in XML format. The expression is stored, but not the value."""
        assert tag in ('parameter', 'compartment')
        attributes = {'id':id, 'expression':self.expression}
        if self.name:
            attributes['name'] = self.name
        writer.writeEmptyElement(tag, attributes)

    def readXml(self, attributes):
        """Read from an attributes dictionary. Return any errors encountered."""
        # The attribute "dictionary" may not work as it should. In particular
        # the test "x in attributes" may not work. Thus we need to directly 
        # use attributes.keys().
        keys = attributes.keys()
        if not 'expression' in keys:
            return 'Missing expression attribute in parameter or compartment.\n'
        self.expression = attributes['expression']
        if 'name' in keys:
            self.name = attributes['name']
        else:
            self.name = ''
        self.value = None
        return ''

    def writeParameterXml(self, writer, id):
        self.writeXml(writer, 'parameter', id)

    def writeCompartmentXml(self, writer, id):
        self.writeXml(writer, 'compartment', id)

    def writeParameterSbml(self, writer, id):
        """Write the parameter in SBML format. In SBML format the value
        is stored, but not the expression. The constant field is required for
        SBML, but not used in Cain. Thus we set it to false."""
        assert self.value is not None
        attributes = {'id':id, 'value':repr(self.value), 'constant':'false'}
        if self.name:
            attributes['name'] = self.name
        writer.writeEmptyElement('parameter', attributes)

    def writeCompartmentSbml(self, writer, id):
        """Write the compartment in SBML format. In SBML format the value
        is stored, but not the expression. The constant field is required for
        SBML, but not used in Cain. Thus we set it to false."""
        assert self.value is not None
        attributes = {'id':id, 'size':repr(self.value), 'constant':'false'}
        if self.name:
            attributes['name'] = self.name
        writer.writeEmptyElement('compartment', attributes)

def main():
    import math
    from ParameterEvaluation import evaluateValues
    from io.XmlWriter import XmlWriter

    writer = XmlWriter()

    x = Value('', '')
    x.name = 'Parameter 1'
    x.expression = '3'
    x.value = 3.0
    x.writeParameterXml(writer, 'P1')
    x.writeParameterSbml(writer, 'P1')
    x.writeCompartmentXml(writer, 'C1')
    x.writeCompartmentSbml(writer, 'C1')

    # Invalid identifers.
    print('\nInvalid identifers:')
    x = Value('', '')
    result = evaluateValues({'':x})
    print(result)
    result = evaluateValues({' ':x})
    print(result)
    result = evaluateValues({'2x':x})
    print(result)
    result = evaluateValues({'a.txt':x})
    print(result)

    # Invalid expressions.
    print('\nInvalid expressions:')
    x.expression = ''
    result = evaluateValues({'a':x})
    print(result)
    x.expression = ' '
    result = evaluateValues({'a':x})
    print(result)
    x.expression = 'x'
    result = evaluateValues({'a':x})
    print(result)
    x.expression = '1 2'
    result = evaluateValues({'a':x})
    print(result)
    x.expression = 'a'
    result = evaluateValues({'a':x})
    print(result)

    # Invalid expressions for two parameters.
    print('\nInvalid expressions for two parameters:')
    y = Value('', '')
    x.expression = 'b'
    y.expression = 'a'
    result = evaluateValues({'a':x, 'b':y})
    print(result)

if __name__ == '__main__':
    main()
