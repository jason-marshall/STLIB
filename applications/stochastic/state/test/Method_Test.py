"""Tests the Method class."""

import sys
if __name__ == '__main__':
    sys.path.insert(1, '..')
    sys.path.insert(1, '../..')
else:
    sys.path.insert(1, 'state')

from StringIO import StringIO

from unittest import TestCase, main

from Method import Method, writeMethodXml

class MethodTest(TestCase):
    def test(self):
        method = Method()
        method.id = 'a'
        method.startTime = -1.
        method.equilibrationTime = 1.
        method.recordingTime = 2.
        method.maximumSteps = 1e9
        method.numberOfFrames = 100
        method.solverParameter = 0.01
        stream = StringIO()
        writeMethodXml(method, stream)

if __name__ == '__main__':
    main()
