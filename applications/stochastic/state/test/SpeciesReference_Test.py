"""Tests the SpeciesReference class."""

import sys
if __name__ == '__main__':
    sys.path.insert(1, '..')
    sys.path.insert(1, '../..')
else:
    sys.path.insert(1, 'state')

from StringIO import StringIO

from unittest import TestCase, main

from SpeciesReference import SpeciesReference
from io.XmlWriter import XmlWriter

class SpeciesReferenceTest(TestCase):
    def test(self):
        stream = StringIO()
        writer = XmlWriter(stream)

        identifiers = ['s1', 's2']
        x = SpeciesReference('s1')
        assert not x.hasErrors(identifiers)
        assert x.hasErrors([])
        x.writeXml(writer)

        x = SpeciesReference('s1', 1)
        assert not x.hasErrors(identifiers)
        assert x.hasErrors([])
        x.writeXml(writer)

        x = SpeciesReference('s2', 2)
        assert not x.hasErrors(identifiers)
        assert x.hasErrors([])
        x.writeXml(writer)

if __name__ == '__main__':
    main()
