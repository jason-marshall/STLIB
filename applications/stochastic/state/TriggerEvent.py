"""Implements the TriggerEvent class."""

class TriggerEvent:
    """Member data:
    - self.id: The time event identifier string.
    - self.name: Optional descriptive name for the time event.
    - self.trigger: A string that is a python expression. The event fires
    when this predicate changes from false to true.
    - self.assignments: A string that is a python expression, which evaluates to
    the assignments that occur when the event is executed.
    - self.delay: A numeric value that is the delay between triggering and
    execution.
    - self.useValuesFromTriggerTime: Whether to use model values from the
    trigger time when executing the event."""
    
    def __init__(self, id, name, trigger, assignments, delay,
                 useValuesFromTriggerTime):
        self.id = id
        self.name = name
        self.trigger = trigger
        self.assignments = assignments
        self.delay = delay
        self.useValuesFromTriggerTime = useValuesFromTriggerTime

    def hasErrors(self):
        """Return None if the trigger event is valid. Otherwise return an error
        message. Note that the trigger and assignments are not checked."""
        # The identifier must be non-null.
        if not self.id:
            return 'The identifier is empty.'
        if not isinstance(self.delay, float):
            return 'The delay in the trigger event is not a floating-point '\
                   'value.'
        if not self.delay >= 0:
            return 'The delay in the trigger event must be non-negative.'
        if not isinstance(self.useValuesFromTriggerTime, bool):
            return 'The useValuesFromTriggerTime parameter is not a boolean '\
                   'value.'
        return None

    def writeXml(self, writer):
        attributes = {'id': self.id, 'trigger': self.trigger,
                      'assignments': self.assignments}
        if self.delay:
            attributes['delay'] = repr(self.delay)
        if self.name:
            attributes['name'] = self.name
        if self.useValuesFromTriggerTime:
            attributes['useValuesFromTriggerTime'] = 'true'
        writer.writeEmptyElement('triggerEvent', attributes)

    def readXml(self, attributes):
        """Read from an attributes dictionary. Return any errors encountered."""
        # The attribute "dictionary" may not work as it should. In particular
        # the test "x in attributes" may not work. Thus we need to directly 
        # use attributes.keys().
        keys = attributes.keys()
        for x in ['id', 'trigger', 'assignments']:
            if not x in keys:
                return 'Missing ' + x + ' attribute in trigger event.\n'
        self.id = attributes['id']
        if 'name' in keys:
            self.name = attributes['name']
        else:
            self.name = ''
        if 'delay' in keys:
            self.delay = float(attributes['delay'])
        else:
            self.delay = 0.
        self.trigger = attributes['trigger']
        self.assignments = attributes['assignments']
        if 'useValuesFromTriggerTime' in keys:
            self.useValuesFromTriggerTime = bool(\
                attributes['useValuesFromTriggerTime'] == 'true')
        else:
            self.useValuesFromTriggerTime = False
        return ''


if __name__ == '__main__':
    import sys
    sys.path.insert(1, '..')
    from io.XmlWriter import XmlWriter

    # CONTINUE: '>' is converted to '&gt;'. is this OK?
    x = TriggerEvent('e1', '', 't>1', 'p1=1; p2=2', 1., True)
    assert not x.hasErrors()
    writer = XmlWriter()
    x.writeXml(writer)
    print('')
