# cylinder.l2.amr.py

# Each grid has 5120 points.
# 1600 grids.

di = 16
dj = 16
dk = 20
for k in range( 0, 320, dk ):
    for j in range( 0, 160, dj ):
        for i in range( 0, 160, di ):
            print i, j, k, i + di, j + dj, k + dk
