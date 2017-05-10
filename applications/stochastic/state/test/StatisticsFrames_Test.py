"""Tests the StatisticsFrames class."""

import sys
if __name__ == '__main__':
    sys.path.insert(1, '..')
    sys.path.insert(1, '../..')
else:
    sys.path.insert(1, 'state')

from StringIO import StringIO

from unittest import TestCase, main

from StatisticsFrames import StatisticsFrames
from io.XmlWriter import XmlWriter

class StatisticsFramesTest(TestCase):
    def test(self):
        frameTimes = [0, 1]
        recordedSpecies = [0, 1, 2]
        x = StatisticsFrames(recordedSpecies)
        x.setFrameTimes(frameTimes)
        x.setStatistics([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        assert x.statistics[0][0] == (1, 2)
        assert x.statistics[0][1] == (3, 4)
        assert x.statistics[0][2] == (5, 6)
        assert x.statistics[1][0] == (7, 8)
        assert x.statistics[1][1] == (9, 10)
        assert x.statistics[1][2] == (11, 12)
        assert not x.hasErrors()

        writer = XmlWriter(StringIO())
        writer.beginDocument()
        x.writeXml(writer, 'model', 'method')
        writer.endDocument()

if __name__ == '__main__':
    main()
