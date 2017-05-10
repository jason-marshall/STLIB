"""Export to gnuplot files."""

import csv

class gnuplot(csv.Dialect):
    """Describe the usual properties of gnuplot data files."""
    delimiter = ' '
    quotechar = '"'
    doublequote = True
    skipinitialspace = False
    lineterminator = '\n'
    quoting = csv.QUOTE_MINIMAL
csv.register_dialect("gnuplot", gnuplot)

def main():
    pass

if __name__ == '__main__':
    main()
