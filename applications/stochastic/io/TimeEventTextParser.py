"""Parses a text representation of a time event."""

if __name__ == '__main__':
    import sys
    sys.path.insert(1, '..')

from state.TimeEvent import TimeEvent

import string
import re

class TimeEventTextParser:
    """Parses a text representation of a time event."""
    
    def __init__(self):
        self.errorMessage = ''

    def parse(self, identifier, times, assignments, name, identifiers,
              label=''):
        """identifier, times, assignments, and
        name are strings. If these can be parsed, define 
        self.event and return True."""
        self.errorMessage = ''

        # No need to check the name. It can be anything.
        # Check that the ID string is valid.
        matchObject = re.match('([a-zA-z]|_)[a-zA-Z0-9_]*', identifier)
        if not (matchObject and matchObject.group(0) == identifier):
            self.errorMessage = 'Time event ' + label +\
                ' has a bad identifier: ' + identifier + '\n' +\
                'Identifiers must begin with a letter or underscore and\n'+\
                'be composed only of letters, digits, and underscores.'
            return False
        # Check that the ID is distinct.
        if identifier in identifiers:
            self.errorMessage = 'Time event ' + label +\
                ' has a duplicate identifier: ' + identifier + '\n' +\
                'The identifiers must be distinct.'
            return False
        # Check the times.
        try:
            timesList = eval(times)
            if not isinstance(timesList, list):
                times = '[' + times + ']'
                timesList = [timesList]
            for t in timesList:
                try:
                    float(t)
                except:
                    self.errorMessage = 'Could not convert the times to '\
                                        'numeric values in time event ' +\
                                        label + '.'
                    return False
        except:
            self.errorMessage = 'Could not evaluate the time(s) in time '\
                                'event ' + label + '.'
            return False
        # If the assignments string is empty.
        if not assignments:
            self.errorMessage = 'Time event ' + label +\
                ' has an empty assignment.'
            return False
        # Build the event.
        self.event = TimeEvent(identifier, name, times, assignments)
        return True

    def parseTable(self, table, identifiers):
        """Return a list of the time events."""
        self.errorMessage = ''

        events = []
        count = 1
        for row in table:
            assert len(row) == 4
            if not self.parse(row[0], row[1], row[2], row[3], identifiers,
                              str(count)):
                return []
            # Record the reaction.
            events.append(self.event)
            count = count + 1
            identifiers.append(row[0])

        return events

def main():
    parser = TimeEventTextParser()
    assert parser.parse('e1', '1', 's1=2', 'name', [])

    assert not parser.parse('1', '1', 's1=2', 'name', [])
    assert parser.errorMessage
    print parser.errorMessage
    print ''

    assert not parser.parse('e1', '1', 's1=2', 'name', ['e1'])
    assert parser.errorMessage
    print parser.errorMessage
    print ''

    assert not parser.parse('e1', 'a', 's1=2', 'name', [])
    assert parser.errorMessage
    print parser.errorMessage
    print ''

    assert not parser.parse('e1', '[a]', 's1=2', 'name', [])
    assert parser.errorMessage
    print parser.errorMessage
    print ''

    assert not parser.parse('e1', '1', '', 'name', [])
    assert parser.errorMessage
    print parser.errorMessage
    print ''

    identifiers = []
    assert parser.parseTable([['e1', '1', 's1=2', 'name']], identifiers)
    assert not parser.parseTable([['e1', '1', 's1=2', 'name']], identifiers)
    assert parser.errorMessage
    print parser.errorMessage
    print ''

if __name__ == '__main__':
    main()
