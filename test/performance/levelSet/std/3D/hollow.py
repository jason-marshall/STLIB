# A hollow cubic block of atoms. Specify the extent and spacing.

import sys

extent = int(sys.argv[1])
assert extent > 0
spacing = float(sys.argv[2])
assert spacing > 0

interior = range(1, extent - 1)
for k in range(extent):
    for j in range(extent):
        for i in range(extent):
            if not (i in interior and j in interior and k in interior):
                print('%s %s %s 1' % (i * spacing, j * spacing, k * spacing))
    