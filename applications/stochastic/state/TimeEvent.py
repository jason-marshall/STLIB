"""Implements the TimeEvent class."""

class TimeEvent:
    """Member data:
    - self.id: The time event identifier string.
    - self.name: Optional descriptive name for the time event.
    - self.times: A string that is a python expression, which evaluates to
    the list of event times.
    - self.assignments: A string that is a python expression, which evaluates to
    the assignments that occur when the event is executed."""
    
    def __init__(self, id, name, times, assignments):
        self.id = id
        self.name = name
        self.times = times
        self.assignments = assignments

    def hasErrors(self):
        """Return None if the time event is valid. Otherwise return an error
        message. Note that the assignments are not checked."""
        # The identifier must be non-null.
        if not self.id:
            return 'The identifier is empty.'
        try:
            times = eval(self.times)
            if not isinstance(times, list):
                return 'The times field is not a list.'
            for t in times:
                try:
                    float(t)
                except:
                    return 'Could not convert the time to a numeric value.'
        except:
            return 'Could not evaluate the time(s).'
        return None

    def writeXml(self, writer):
        attributes = {'id': self.id, 'times': self.times,
                      'assignments': self.assignments}
        if self.name:
            attributes['name'] = self.name
        writer.writeEmptyElement('timeEvent', attributes)

    def readXml(self, attributes):
        """Read from an attributes dictionary. Return any errors encountered."""
        # The attribute "dictionary" may not work as it should. In particular
        # the test "x in attributes" may not work. Thus we need to directly 
        # use attributes.keys().
        keys = attributes.keys()
        for x in ['id', 'times', 'assignments']:
            if not x in keys:
                return 'Missing ' + x + ' attribute in time event.\n'
        self.id = attributes['id']
        if 'name' in keys:
            self.name = attributes['name']
        else:
            self.name = ''
        self.times = attributes['times']
        self.assignments = attributes['assignments']
        return ''


if __name__ == '__main__':
    import sys
    sys.path.insert(1, '..')
    from io.XmlWriter import XmlWriter

    x = TimeEvent('e1', '', '[0]', 'p1=1; p2=2')
    writer = XmlWriter()
    x.writeXml(writer)
