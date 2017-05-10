"""Tests the Identifier module."""

import sys
if __name__ == '__main__':
    sys.path.insert(1, '..')
    sys.path.insert(1, '../..')
else:
    sys.path.insert(1, 'state')

from unittest import TestCase, main

from Identifier import *

class IdentifierTest(TestCase):
    def test(self):
        assert hasFormatError('')
        assert hasFormatError('2')
        assert hasFormatError('2a')
        assert hasFormatError('a b')
        assert hasFormatError('a+b')
        assert hasFormatError('a-b')

        assert not hasFormatError('_')
        assert not hasFormatError('__')
        assert not hasFormatError('__a')
        assert not hasFormatError('a1')
        assert not hasFormatError('a_1')

        for id in ['and', 'as', 'assert', 'break', 'class', 'continue', 'def',
                   'del', 'elif', 'else', 'except', 'exec', 'finally', 'for',
                   'from', 'global', 'if', 'import', 'in', 'is', 'lambda', 'not',
                   'or', 'pass', 'print', 'raise', 'return', 'try', 'while',
                   'with', 'yield']:
            assert hasReservedWordError(id)
        assert not hasReservedWordError('And')

        for id in ['False', 'True', 'None', 'NotImplemented', 'Ellipsis']:
            assert hasBuiltinConstantError(id)
        assert not hasBuiltinConstantError('false')

        assert hasSystemDefinedError('____')
        assert hasSystemDefinedError('__a__')
        assert hasSystemDefinedError('______')
        assert not hasSystemDefinedError('___')
        assert not hasSystemDefinedError('__a_')

        assert hasApplicationDefinedError('__')
        assert hasApplicationDefinedError('__a')
        assert hasApplicationDefinedError('______')
        assert not hasApplicationDefinedError('_')
        assert not hasApplicationDefinedError('_a__')

        assert hasBuiltinExceptionError('ArithmeticError')
        assert not hasBuiltinExceptionError('arithmeticError')

        assert hasBuiltinFunctionError('any')
        assert not hasBuiltinFunctionError('Any')

        assert hasMathError('exp')
        assert hasMathError('e')
        assert not hasMathError('Exp')
        assert not hasMathError('E')

if __name__ == '__main__':
    main()
