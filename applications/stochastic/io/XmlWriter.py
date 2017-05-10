"""XML Writer."""

import sys

def escape(data):
    """Escape &, <, and > in a string of data."""
    data = data.replace("&", "&amp;")
    data = data.replace(">", "&gt;")
    return data.replace("<", "&lt;")

def unescape(data):
    """Unescape &amp;, &lt;, and &gt; in a string of data."""
    data = data.replace("&lt;", "<")
    data = data.replace("&gt;", ">")
    return data.replace("&amp;", "&")

class XmlWriter:
    """A simple XML writer.  

    Why use standard solutions when you can roll your own?"""
    
    def __init__(self, out=sys.stdout, encoding="utf-8", indent=u"  "):
        self.out = out
        self.encoding = encoding
        self.indent = indent
        self.openElements = []

    def _indent(self):
        """Use the stack of open elements to indent before a tag."""
        self.out.write(self.indent * len(self.openElements))

    def _write(self, text):
        self._indent()
        self.out.write(text)

    def beginDocument(self):
        assert self.openElements == []
        self.out.write(u'<?xml version="1.0" encoding="%s"?>\n' % 
                       self.encoding)

    def endDocument(self):
        assert self.openElements == []

    def beginElement(self, tag, attributes={}):
        self._write('<' + tag)
        for (name, value) in attributes.items():
            self.out.write(' %s="%s"' % (name, escape(value)))
        self.out.write('>\n')
        self.openElements.append(tag)

    def endElement(self):
        assert self.openElements != []
        tag = self.openElements[-1]
        del self.openElements[-1]
        self._write('</%s>\n' % tag)

    def writeElement(self, tag, attributes={}, data=None):
        if data:
            self.beginElement(tag, attributes)
            self.writeData(data)
            self.endElement()
        else:
            self.writeEmptyElement(tag, attributes)

    def writeEmptyElement(self, tag, attributes={}):
        self._write('<' + tag)
        for (name, value) in attributes.items():
            self.out.write(' %s="%s"' % (name, escape(value)))
        self.out.write('/>\n')

    def writeData(self, data):
        self._write('%s\n' % escape(data))

def main():
    writer = XmlWriter()
    writer.beginDocument()
    writer.beginElement('model')
    writer.beginElement('listOfSpecies')
    writer.writeElement('species', {}, 's1')
    writer.writeElement('species', {}, 's<2>')
    writer.writeEmptyElement('species')
    writer.writeEmptyElement('species', {'name':'s1'})
    writer.writeEmptyElement('species', {'name':'s<2>'})
    writer.endElement()
    writer.endElement()
    writer.endDocument()

if __name__ == '__main__':
    main()
