"""Tests the StatisticsAverage class."""

import sys
if __name__ == '__main__':
    sys.path.insert(1, '..')
    sys.path.insert(1, '../..')
else:
    sys.path.insert(1, 'state')

from StringIO import StringIO

from unittest import TestCase, main

from StatisticsAverage import StatisticsAverage
from io.XmlWriter import XmlWriter

class StatisticsAverageTest(TestCase):
    def test(self):
        recordedSpecies = [0, 1, 2]
        x = StatisticsAverage(recordedSpecies)
        x.setStatistics([1, 2, 3, 4, 5, 6])
        assert x.statistics[0] == (1, 2)
        assert x.statistics[1] == (3, 4)
        assert x.statistics[2] == (5, 6)
        assert not x.hasErrors()

        writer = XmlWriter(StringIO())
        writer.beginDocument()
        x.writeXml(writer, 'model', 'method')
        writer.endDocument()

if __name__ == '__main__':
    main()
