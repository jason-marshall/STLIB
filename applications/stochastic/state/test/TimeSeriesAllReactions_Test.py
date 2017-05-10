"""Tests the TimeSeriesAllReactions class."""

import sys
if __name__ == '__main__':
    sys.path.insert(1, '..')
    sys.path.insert(1, '../..')
else:
    sys.path.insert(1, 'state')

from StringIO import StringIO

from unittest import TestCase, main

from TimeSeriesAllReactions import TimeSeriesAllReactions
from io.XmlWriter import XmlWriter

class TimeSeriesAllReactionsTest(TestCase):
    def test(self):
        # One species, two reactions.
        trajectory = TimeSeriesAllReactions([0], [0, 1], 1., 5.)
        trajectory.appendInitialPopulations([7.])
        trajectory.appendIndices([0, 1, 1, 0])
        trajectory.appendTimes([1, 2, 3, 4])
        assert not trajectory.hasErrors()

        writer = XmlWriter(StringIO())
        writer.beginDocument()
        trajectory.writeXml(writer, 'model', 'method')
        writer.endDocument()

if __name__ == '__main__':
    main()
