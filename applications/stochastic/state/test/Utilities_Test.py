"""Tests the Utilities module."""

import sys
if __name__ == '__main__':
    sys.path.insert(1, '..')
else:
    sys.path.insert(1, 'state')

from unittest import TestCase, main

from Utilities import isFloat, getNewIntegerString, getUniqueName

class UtilitiesTest(TestCase):
    def testIsFloat(self):
        assert isFloat('1')
        assert isFloat('-1')
        assert isFloat('1e4')
        assert not isFloat('(1)')
        assert not isFloat('2 * 3')

    def testGetNewIntegerString(self):
        assert getNewIntegerString([]) == '1'
        assert getNewIntegerString(['0']) == '1'
        assert getNewIntegerString(['a']) == '1'
        assert getNewIntegerString(['1']) == '2'
        assert getNewIntegerString(['2']) == '1'
        assert getNewIntegerString(['2', '1']) == '3'

    def testGetUniqueName(self):
        assert getUniqueName('a', []) == 'a'
        assert getUniqueName('a', ['a']) == 'a1'
        assert getUniqueName('a', ['a', 'a1']) == 'a2'

if __name__ == '__main__':
    main()
