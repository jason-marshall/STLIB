"""Tests the TriggerEvent class."""

import sys
if __name__ == '__main__':
    sys.path.insert(1, '..')
    sys.path.insert(1, '../..')
else:
    sys.path.insert(1, 'state')

from StringIO import StringIO

from unittest import TestCase, main

from TriggerEvent import TriggerEvent
from io.XmlWriter import XmlWriter

class TriggerEventTest(TestCase):
    def test(self):
        assert TriggerEvent('e1', '', '', '', 0, True).hasErrors()
        assert TriggerEvent('e1', '', '', '', '', True).hasErrors()
        assert TriggerEvent('e1', '', '', '', 0., 0).hasErrors()
        assert not TriggerEvent('e1', '', '', '', 0., True).hasErrors()

        # CONTINUE: '>' is converted to '&gt;'. is this OK?
        x = TriggerEvent('e1', '', 't>1', 'p1=1; p2=2', 1., True)
        assert not x.hasErrors()
        writer = XmlWriter(StringIO())
        x.writeXml(writer)

if __name__ == '__main__':
    main()
