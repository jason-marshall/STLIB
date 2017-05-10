"""Tests the Value class."""

import sys
if __name__ == '__main__':
    sys.path.insert(1, '..')
    sys.path.insert(1, '../..')
else:
    sys.path.insert(1, 'state')

import math
from StringIO import StringIO

from unittest import TestCase, main

from Value import Value
from io.XmlWriter import XmlWriter
from ParameterEvaluation import evaluateValues

class ValueTest(TestCase):
    def test(self):
        writer = XmlWriter(StringIO())

        x = Value('', '')
        assert x.hasErrors()

        x.name = 'Parameter 1'
        x.expression = '3'
        x.value = 3.0
        assert not x.hasErrors()
        x.writeParameterXml(writer, 'P1')
        x.writeParameterSbml(writer, 'P1')
        x.writeCompartmentXml(writer, 'C1')
        x.writeCompartmentSbml(writer, 'C1')

        # Invalid identifers.
        x = Value('', '')
        result = evaluateValues({'':x})
        assert result
        result = evaluateValues({' ':x})
        assert result
        result = evaluateValues({'2x':x})
        assert result
        result = evaluateValues({'a.txt':x})
        assert result

        # Invalid expressions.
        x.expression = ''
        result = evaluateValues({'a':x})
        assert result
        x.expression = ' '
        result = evaluateValues({'a':x})
        assert result
        x.expression = 'x'
        result = evaluateValues({'a':x})
        assert result
        x.expression = '1 2'
        result = evaluateValues({'a':x})
        assert result
        x.expression = 'a'
        result = evaluateValues({'a':x})
        assert result

        # Valid expressions for a single parameter.
        for key in math.__dict__.keys():
            x.expression = '1'
            assert not evaluateValues({key:x})
            assert x.value == 1

        x.expression = '1.0'
        result = evaluateValues({'a':x})
        assert not result
        assert x.value == 1

        x.expression = '1.0'
        result = evaluateValues({'lambda':x})
        assert not result
        assert x.value == 1

        x.expression = '1.0'
        result = evaluateValues({'e':x})
        assert not result
        assert x.value == 1

        x.expression = '1.0'
        result = evaluateValues({'pi':x})
        assert not result
        assert x.value == 1

        x.expression = '1.0'
        result = evaluateValues({'sin':x})
        assert not result
        assert x.value == 1

        x.expression = '1'
        result = evaluateValues({'a':x})
        assert not result
        assert x.value == 1

        x.expression = '2 * 3'
        result = evaluateValues({'a':x})
        assert not result
        assert x.value == 2 * 3

        x.expression = '2 - 3'
        result = evaluateValues({'a':x})
        assert not result
        assert x.value == 2 - 3

        x.expression = '2 ** 3'
        result = evaluateValues({'a':x})
        assert not result
        assert x.value == 2**3

        x.expression = '2./3'
        result = evaluateValues({'a':x})
        assert not result
        assert x.value == 2./3

        x.expression = 'pi'
        result = evaluateValues({'a':x})
        assert not result
        assert x.value == math.pi

        x.expression = 'e'
        result = evaluateValues({'a':x})
        assert not result
        assert x.value == math.e

        x.expression = 'sqrt(2)'
        result = evaluateValues({'a':x})
        assert not result
        assert x.value == math.sqrt(2)

        x.expression = 'log(2)'
        result = evaluateValues({'a':x})
        assert not result
        assert x.value == math.log(2)


        # Valid expressions for two parameters.
        y = Value('', '')
        x.expression = 'pi'
        y.expression = 'e'
        result = evaluateValues({'a':x, 'b':y})
        assert not result
        assert x.value == math.pi
        assert y.value == math.e

        x.expression = '1.0'
        y.expression = 'a'
        result = evaluateValues({'a':x, 'b':y})
        assert not result
        assert x.value == 1
        assert y.value == 1

        x.expression = 'b'
        y.expression = '1'
        result = evaluateValues({'a':x, 'b':y})
        assert not result
        assert x.value == 1
        assert y.value == 1

        x.expression = '2'
        y.expression = 'sqrt(a)'
        result = evaluateValues({'a':x, 'b':y})
        assert not result
        assert x.value == 2
        assert y.value == math.sqrt(2)

        x.expression = '2**3'
        y.expression = 'sqrt(a)'
        result = evaluateValues({'a':x, 'b':y})
        assert not result
        assert x.value == 2**3
        assert y.value == math.sqrt(2**3)

        # Invalid expressions for two parameters.
        y = Value('', '')
        x.expression = 'b'
        y.expression = 'a'
        result = evaluateValues({'a':x, 'b':y})
        assert result

        # Valid expressions for three parameters.
        z = Value('', '')
        x.expression = 'pi'
        y.expression = 'a**2'
        z.expression = 'sqrt(a + b)'
        result = evaluateValues({'a':x, 'b':y, 'c':z})
        assert not result
        assert x.value == math.pi
        assert y.value == x.value**2
        assert z.value == math.sqrt(x.value + y.value)

if __name__ == '__main__':
    main()
