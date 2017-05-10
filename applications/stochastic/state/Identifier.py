"""Implements functions to check identifiers."""

import re
import math

builtinKeys = globals()['__builtins__'].keys()
builtinExceptions = []
for s in builtinKeys:
    if s[0].isupper():
        builtinExceptions.append(s)
builtinFunctions = []
for s in builtinKeys:
    if s[0].islower():
        builtinFunctions.append(s)
mathFunctions = math.__dict__.keys()

def hasFormatError(id, name='Object'):
    """Check if it is a valid Python identifier."""
    matchObject = re.match('([a-zA-z]|_)[a-zA-Z0-9_]*', id)
    if not (matchObject and matchObject.group(0) == id):
        return name + ' has a bad identifier: ' + id + '\n' +\
            'Identifiers must begin with a letter or underscore and\n'+\
            'be composed only of letters, digits, and underscores.'

def hasReservedWordError(id, name='Object'):
    """Ensure the identifier is not a Python reserved word."""
    if id in ['and', 'as', 'assert', 'break', 'class', 'continue', 'def',
              'del', 'elif', 'else', 'except', 'exec', 'finally', 'for',
              'from', 'global', 'if', 'import', 'in', 'is', 'lambda', 'not',
              'or', 'pass', 'print', 'raise', 'return', 'try', 'while',
              'with', 'yield']:
        return name + ' has a bad identifier: ' + id + '\n' +\
            'That is a Python reserved word.'

def hasBuiltinConstantError(id, name='Object'):
    """Ensure the identifier is not a Python built-in constant."""
    if id in ['False', 'True', 'None', 'NotImplemented', 'Ellipsis']:
        return name + ' has a bad identifier: ' + id + '\n' +\
            'That is a Python built-in constant.'

def hasBuiltinExceptionError(id, name='Object'):
    """Ensure the identifier is not a Python built-in exception."""
    if id in builtinExceptions:
        return name + ' has a bad identifier: ' + id + '\n' +\
            'That is a Python built-in exception.'

def hasBuiltinFunctionError(id, name='Object'):
    """Ensure the identifier is not a Python built-in function."""
    if id in builtinFunctions:
        return name + ' has a bad identifier: ' + id + '\n' +\
            'That is a Python built-in function.'

def hasMathError(id, name='Object'):
    """Ensure the identifier is not a Python standard math function or 
    constant."""
    if id in mathFunctions:
        return name + ' has a bad identifier: ' + id + '\n' +\
            'That is a Python standard math function or constant.'

def hasSystemDefinedError(id, name='Object'):
    """Identifiers that begin and end with __ are reserved for system-defined
    names."""
    if len(id) >= 4 and id[0:2] == '__' and id[-2:] == '__':
        return name + ' has a bad identifier: ' + id + '\n' +\
            'Identifiers that begin and end with __ are reserved for\n' +\
            'system-defined names.'

def hasApplicationDefinedError(id, name='Object'):
    """Identifiers that begin with __ are reserved for application-defined
    names."""
    if len(id) >= 2 and id[0:2] == '__':
        return name + ' has a bad identifier: ' + id + '\n' +\
            'Identifiers that begin and end with __ are reserved for\n' +\
            'application-defined names.'
