"""SbmlLevel1To2.py."""

import sys, re

if len(sys.argv) != 3:
    print("""Usage:
python SbmlLevel1To2.py input.xml output.xml
Bad command line arguments. Exiting...""")
    sys.exit()

def convert(input):
    return input.replace('level="1"', 'level="2"')\
        .replace('<specie ', '<species ')\
        .replace('specie=', 'species=')\
        .replace('<specieReference ', '<speciesReference ')

output = open(sys.argv[2], 'w')
output.write(convert(open(sys.argv[1], 'r').read()))
