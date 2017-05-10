"""Tests the TimeEvent class."""

import sys
if __name__ == '__main__':
    sys.path.insert(1, '..')
    sys.path.insert(1, '../..')
else:
    sys.path.insert(1, 'state')

from StringIO import StringIO

from unittest import TestCase, main

from TimeEvent import TimeEvent
from io.XmlWriter import XmlWriter

class TimeEventTest(TestCase):
    def test(self):
        assert TimeEvent('e1', '', '', '').hasErrors()

        x = TimeEvent('e1', '', '[0]', 'p1=1; p2=2')
        assert not x.hasErrors()
        writer = XmlWriter(StringIO())
        x.writeXml(writer)

        assert TimeEvent('e1', '', '1', '').hasErrors()
        assert not TimeEvent('e1', '', 'range(10)', '').hasErrors()
        assert not TimeEvent('e1', '', '[0.1 * i for i in range(10)]', '').\
               hasErrors()

if __name__ == '__main__':
    main()
