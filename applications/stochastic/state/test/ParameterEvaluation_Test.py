"""Tests the ParameterEvaluation module."""

import sys
if __name__ == '__main__':
    sys.path.insert(1, '..')
    sys.path.insert(1, '../..')
else:
    sys.path.insert(1, 'state')

from unittest import TestCase, main

from ParameterEvaluation import getParameters

class ParameterEvaluationTest(TestCase):
    def test(self):
        assert len(getParameters('', [])) == 0
        assert len(getParameters('a', [])) == 0
        assert len(getParameters('a', ['a'])) == 1
        assert len(getParameters('a alpha', ['a'])) == 1
        assert len(getParameters('a alpha', ['alpha'])) == 1
        assert len(getParameters('a alpha _beta', ['alpha'])) == 1
        assert len(getParameters('a alpha _beta', ['alpha', '_b'])) == 1
        assert len(getParameters('a alpha _beta', ['alpha', '_beta'])) == 2

if __name__ == '__main__':
    main()
