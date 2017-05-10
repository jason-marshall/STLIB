"""Implements utility functions."""

def isFloat(x):
    try:
        float(x)
    except:
        return False
    return True

def getNewIntegerString(strings):
    n = 0
    while True:
        n += 1
        if not str(n) in strings:
            return str(n)

def getUniqueName(base, strings):
    if not base in strings:
        return base
    n = 0
    while True:
        n += 1
        result = base + str(n)
        if not result in strings:
            return result
