"""Mathematica Writer."""

import sys
import math

def mathematicaCoefficient(n):
    assert n != 0
    if n == 1:
        return '+'
    elif n == -1:
        return '-'
    else:
        return '%+d' % n

def mathematicaForm(x):
    """Return a string representation of the floating point number that
    Mathematica can understand."""
    if x == 0:
        return '0'
    elif abs(x) > 1e-4 and abs(x) < 1e16:
        return repr(float(x))
    else:
        return '%r 2^%r' % math.frexp(x)

class MathematicaWriter:
    """A simple Mathematica writer.

    This is a dirty little hack, not a general purpose writer."""
    
    def __init__(self, out=sys.stdout, indent='  '):
	self.out = out
	self.indent = indent
	self.openElements = []
        self.first = True

    def _indent(self):
	"""Use the stack of open elements to indent before a tag."""
	self.out.write(self.indent * (len(self.openElements) - 1))

    def _writeBegin(self, text):
        if not self.first:
            self.out.write(',\n\n')
	self._indent()
	self.out.write(text)

    def _writeEnd(self, text):
	self.out.write(text)

    def begin(self, id, content=''):
        self.openElements.append(id)
        if id == 'Notebook':
            self._writeBegin('Notebook[{\n')
            self.first = True
        elif id in ['Title', 'Subtitle', 'Subsubtitle', 'Section',
                    'Subsection', 'Subsubsection']:
            self._writeBegin('Cell[CellGroupData[{Cell["%s", "%s"]' % (content, id))
            self.first = False
        elif id == 'Text':
            self._writeBegin('Cell["%s", "Text"]' % content)
            self.first = False
        elif id == 'Input':
            self._writeBegin('Cell[ToBoxes["%s"], "Input"]' % content)
            self.first = False
        else:
            assert False

    def end(self):
        id = self.openElements[-1]
	del self.openElements[-1]
        if id == 'Notebook':
            self._writeEnd('''},
WindowSize->{640, 750},
WindowMargins->{{4, Automatic}, {Automatic, 4}},
ShowSelection->True,
StyleDefinitions->"Default.nb"
]
''')
        elif id in ['Title', 'Subtitle', 'Subsubtitle', 'Section',
                    'Subsection', 'Subsubsection']:
            self._writeEnd('}]]')
        elif id in ['Text', 'Input']:
            pass
        else:
            assert False

def main():
    assert mathematicaCoefficient(1) == '+'
    assert mathematicaCoefficient(-1) == '-'
    assert mathematicaCoefficient(2) == '+2'
    assert mathematicaCoefficient(-2) == '-2'

    writer = MathematicaWriter()
    writer.begin('Notebook')
    writer.begin('Title', 'The Title')
    writer.begin('Subtitle', 'The Subtitle')
    writer.begin('Subsubtitle', 'The Subsubtitle')
    writer.begin('Section', 'The Section')
    writer.begin('Subsection', 'The Subsection')
    writer.begin('Subsubsection', 'The Subsubsection')
    writer.begin('Text', 'Some text.')
    writer.begin('Input', r'lambda:=10;\nmu:=lambda+1;')
    writer.end()
    writer.end()
    writer.end()
    writer.end()
    writer.end()
    writer.end()
    writer.end()
    writer.end()
    writer.end()

if __name__ == '__main__':
    main()
