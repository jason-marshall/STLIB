#
# Make queries for uniform random points on a unit sphere.
#

import random
import math

num_families = 22
queries_per_family = 100


# Open the file.
fout = open('sphere.queries', 'w')
    
# Write the number of families and queries per family.
fout.write('%i\n' % num_families)
fout.write('%i\n' % queries_per_family)

# Make the centers for the queries.
g = random.Random(42)
centers = []
i = 0
while i < queries_per_family:
    x, y, z = 2 * g.random() - 1, 2 * g.random() - 1, 2 * g.random() - 1
    a = x*x + y*y + z*z
    if a <= 1:
        a = math.sqrt(a)
        centers.append((x / a, y / a, z / a))
        i += 1

# Write the query ranges.
for i in range(num_families-3, -3, -1):
    radius = 0.5 * math.pow(math.pow(0.5, i), 0.5)
    for j in range(queries_per_family):
        fout.write('%f %f %f %f %f %f\n' % (centers[j][0] - radius,
                                              centers[j][1] - radius,
                                              centers[j][2] - radius,
                                              centers[j][0] + radius,
                                              centers[j][1] + radius,
                                              centers[j][2] + radius))
# Close the file.
fout.close()
