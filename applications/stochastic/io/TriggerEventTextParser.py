"""Parses a text representation of a trigger event."""

if __name__ == '__main__':
    import sys
    sys.path.insert(1, '..')

from state.TriggerEvent import TriggerEvent

import string
import re

class TriggerEventTextParser:
    """Parses a text representation of a trigger event."""
    
    def __init__(self):
        self.errorMessage = ''

    def parse(self, identifier, trigger, assignments, delay,
              useValuesFromTriggerTime, name, identifiers, label=''):
        """If the trigger event can be parsed, define self.event and return
        True."""
        self.errorMessage = ''

        # No need to check the name. It can be anything.
        # Check that the ID string is valid.
        matchObject = re.match('([a-zA-z]|_)[a-zA-Z0-9_]*', identifier)
        if not (matchObject and matchObject.group(0) == identifier):
            self.errorMessage = 'Trigger event ' + label +\
                ' has a bad identifier: ' + identifier + '\n' +\
                'Identifiers must begin with a letter or underscore and\n'+\
                'be composed only of letters, digits, and underscores.'
            return False
        # Check that the ID is distinct.
        if identifier in identifiers:
            self.errorMessage = 'Trigger event ' + label +\
                ' has a duplicate identifier: ' + identifier + '\n' +\
                'The identifiers must be distinct.'
            return False
        # If the trigger string is empty.
        if not trigger:
            self.errorMessage = 'Trigger event ' + label +\
                ' has an empty trigger.'
            return False
        # If the assignments string is empty.
        if not assignments:
            self.errorMessage = 'Trigger event ' + label +\
                ' has an empty assignment.'
            return False
        # Check that the delay is numeric.
        if not delay.strip():
            delay = 0.
        else:
            try:
                delay = float(delay)
            except:
                self.errorMessage = 'In trigger event ' + label +\
                    ': The delay is not a floating-point value.'
                return False
        # Build the event.
        self.event = TriggerEvent(identifier, name, trigger, assignments,
                                  delay, useValuesFromTriggerTime=='1')
        error = self.event.hasErrors()
        if error:
            self.errorMessage = 'In trigger event ' + label + ': ' + error
            return False
        return True

    def parseTable(self, table, identifiers):
        """Return a list of the trigger events."""
        self.errorMessage = ''

        events = []
        count = 1
        for row in table:
            assert len(row) == 6
            if not self.parse(row[0], row[1], row[2], row[3], row[4], row[5],
                              identifiers, str(count)):
                return []
            # Record the reaction.
            events.append(self.event)
            count = count + 1
            identifiers.append(row[0])

        return events

def main():
    parser = TriggerEventTextParser()
    assert parser.parse('e1', 't>1', 's1=2', '0', '', 'name', [])

    assert not parser.parse('1', 't>1', 's1=2', '0', '', 'name', [])
    assert parser.errorMessage
    print parser.errorMessage
    print ''

    assert not parser.parse('e1', '', 's1=2', '0', '', 'name', [])
    assert parser.errorMessage
    print parser.errorMessage
    print ''

    assert not parser.parse('e1', 't>1', '', '0', '', 'name', [])
    assert parser.errorMessage
    print parser.errorMessage
    print ''

    assert not parser.parse('e1', 't>1', 's1=2', 'a', '', 'name', [])
    assert parser.errorMessage
    print parser.errorMessage
    print ''

    assert not parser.parse('e1', 't>1', 's1=2', '-1', '', 'name', [])
    assert parser.errorMessage
    print parser.errorMessage
    print ''

    identifiers = []
    assert parser.parseTable([['e1', 't>1', 's1=2', '0', '', 'name']],
                             identifiers)
    assert not parser.parseTable([['e1', 't>1', 's1=2', '0', '', 'name']],
                                 identifiers)
    assert parser.errorMessage
    print parser.errorMessage
    print ''

if __name__ == '__main__':
    main()
