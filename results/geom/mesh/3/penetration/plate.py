"""Make points that lie on the plane z = -0.99."""

nx = 1000
ny = 1000
# The number of points.
print nx * ny

# The points.
for i in range(nx):
    for j in range(ny):
        print -1.0 + 2.0 * i / (nx - 1.0), -1.0 + 2.0 * j / (ny - 1.0), -0.99
