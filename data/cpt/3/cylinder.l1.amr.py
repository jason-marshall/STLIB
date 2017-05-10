# cylinder.l1.amr.py

# Each grid has 2560 points.
# 400 grids.

di = 16
dj = 16
dk = 10
for k in range( 0, 160, dk ):
    for j in range( 0, 80, dj ):
        for i in range( 0, 80, di ):
            print i, j, k, i + di, j + dj, k + dk
