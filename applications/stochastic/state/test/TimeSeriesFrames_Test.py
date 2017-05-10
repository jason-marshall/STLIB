"""Tests the TimeSeriesFrames class."""

import sys
if __name__ == '__main__':
    sys.path.insert(1, '..')
    sys.path.insert(1, '../..')
else:
    sys.path.insert(1, 'state')

from StringIO import StringIO

from unittest import TestCase, main

from TimeSeriesFrames import TimeSeriesFrames
from io.XmlWriter import XmlWriter

class TimeSeriesFramesTest(TestCase):
    def test(self):
        trajectory = TimeSeriesFrames([0, 1])
        trajectory.recordedSpecies = [0, 1]
        trajectory.recordedReactions = [0, 1]
        trajectory.appendPopulations([2, 3, 7, 11])
        trajectory.appendReactionCounts([13, 17, 19, 23])
        assert not trajectory.hasErrors()

        writer = XmlWriter(StringIO())
        writer.beginDocument()
        trajectory.writeXml(writer, 'model', 'method')
        writer.endDocument()

if __name__ == '__main__':
    main()
