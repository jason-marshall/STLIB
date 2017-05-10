# cylinder.l0.amr.py

# Each grid has 800 points.
# 160 grids.

di = 10
dj = 10
dk = 8
for k in range( 0, 80, dk ):
    for j in range( 0, 40, dj ):
        for i in range( 0, 40, di ):
            print i, j, k, i + di, j + dj, k + dk
