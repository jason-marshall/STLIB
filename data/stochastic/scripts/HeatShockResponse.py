# HeatShockResponse.py
# Usage:
# python HeatShockResponse.py reactions.txt rates.txt populations.txt

import sys

errorMessage = \
"""Usage:
python HeatShockResponse.py reactions.txt rates.txt populations.txt"""

if len(sys.argv) != 4:
  print errorMessage
  raise "Wrong number of command line arguments.  Exiting..."

# Write the reactions.
print "Writing the reactions file..."
outFile = open(sys.argv[1], "w")
# The number of reactions.
outFile.write("%d\n\n" % 61)

# 0: s0 + s1 -> s2
outFile.write("2 0 1 1 1\n" + "1 2 1\n")

# 1: s2 -> s0 + s1
outFile.write("1 2 1\n" + "2 0 1 1 1\n")

# 2: s0 + s3 -> s4
outFile.write("2 0 1 3 1\n" + "1 4 1\n")

# 3: s4 -> s0 + s3
outFile.write("1 4 1\n" + "2 0 1 3 1\n")

# 4: s0 + s5 -> s6
outFile.write("2 0 1 5 1\n" + "1 6 1\n")

# 5: s6 -> s0 + s5
outFile.write("1 6 1\n" + "2 0 1 5 1\n")

# 6: s3 + s13 -> s14
outFile.write("2 3 1 13 1\n" + "1 14 1\n")

# 7: s14 -> s3 + s13
outFile.write("1 14 1\n" + "2 3 1 13 1\n")

# 8: s13 + s15 -> s16
outFile.write("2 13 1 15 1\n" + "1 16 1\n")

# 9: s16 -> s13 + s15
outFile.write("1 16 1\n" + "2 13 1 15 1\n")

# 10: s2 + s5 -> s7
outFile.write("2 2 1 5 1\n" + "1 7 1\n")

# 11: s7 -> s2 + s5
outFile.write("1 7 1\n" + "2 2 1 5 1\n")

# 12: s4 + s5 -> s8
outFile.write("2 4 1 5 1\n" + "1 8 1\n")

# 13: s8 -> s4 + s5
outFile.write("1 8 1\n" + "2 4 1 5 1\n")

# 14: s2 + s9 -> s11
outFile.write("2 2 1 9 1\n" + "1 11 1\n")

# 15: s11 -> s2 + s9
outFile.write("1 11 1\n" + "2 2 1 9 1\n")

# 16: s4 + s10 -> s12
outFile.write("2 4 1 10 1\n" + "1 12 1\n")

# 17: s12 -> s4 + s10
outFile.write("1 12 1\n" + "2 4 1 10 1\n")

# 18: s14 + s17 -> s18
outFile.write("2 14 1 17 1\n" + "1 18 1\n")

# 19: s18 -> s14 + s17
outFile.write("1 18 1\n" + "2 14 1 17 1\n")

# 20:  -> s21
outFile.write("0\n" + "1 21 1\n")

# 21: s21 -> 
outFile.write("1 21 1\n" + "0\n")

# 22:  -> s13
outFile.write("0\n" + "1 13 1\n")

# 23: s13 -> 
outFile.write("1 13 1\n" + "0\n")

# 24: s16 -> s15
outFile.write("1 16 1\n" + "1 15 1\n")

# 25: s14 -> s3
outFile.write("1 14 1\n" + "1 3 1\n")

# 26: s18 -> s3 + s17
outFile.write("1 18 1\n" + "2 3 1 17 1\n")

# 27: s27 -> s3 + s26
outFile.write("1 27 1\n" + "2 3 1 26 1\n")

# 28:  -> s22
outFile.write("0\n" + "1 22 1\n")

# 29: s22 -> 
outFile.write("1 22 1\n" + "0\n")

# 30:  -> s17
outFile.write("0\n" + "1 17 1\n")

# 31: s17 -> 
outFile.write("1 17 1\n" + "0\n")

# 32: s18 -> s14
outFile.write("1 18 1\n" + "1 14 1\n")

# 33:  -> s24
outFile.write("0\n" + "1 24 1\n")

# 34: s24 -> 
outFile.write("1 24 1\n" + "0\n")

# 35:  -> s3
outFile.write("0\n" + "1 3 1\n")

# 36: s3 -> 
outFile.write("1 3 1\n" + "0\n")

# 37: s18 -> s13 + s17
outFile.write("1 18 1\n" + "2 13 1 17 1\n")

# 38: s20 -> s19
outFile.write("1 20 1\n" + "1 19 1\n")

# 39: s27 -> s13 + s26
outFile.write("1 27 1\n" + "2 13 1 26 1\n")

# 40:  -> s23
outFile.write("0\n" + "1 23 1\n")

# 41: s23 -> 
outFile.write("1 23 1\n" + "0\n")

# 42:  -> s19
outFile.write("0\n" + "1 19 1\n")

# 43: s19 -> 
outFile.write("1 19 1\n" + "0\n")

# 44: s20 -> s3
outFile.write("1 20 1\n" + "1 3 1\n")

# 45: s3 + s19 -> s20
outFile.write("2 3 1 19 1\n" + "1 20 1\n")

# 46: s20 -> s3 + s19
outFile.write("1 20 1\n" + "2 3 1 19 1\n")

# 47:  -> s25
outFile.write("0\n" + "1 25 1\n")

# 48: s25 -> 
outFile.write("1 25 1\n" + "0\n")

# 49:  -> s26
outFile.write("0\n" + "1 26 1\n")

# 50: s26 -> 
outFile.write("1 26 1\n" + "0 \n")

# 51: s27 -> s14
outFile.write("1 27 1\n" + "1 14 1\n")

# 52: s14 + s26 -> s27
outFile.write("2 14 1 26 1\n" + "1 27 1\n")

# 53: s27 -> s14 + s26
outFile.write("1 27 1\n" + "2 14 1 26 1\n")

# 54: s4 -> s0
outFile.write("1 4 1\n" + "1 0 1\n")

# 55: s12 -> s0 + s10
outFile.write("1 12 1\n" + "2 0 1 10 1\n")

# 56: s8 -> s6
outFile.write("1 8 1\n" + "1 6 1\n")

# 57: s14 -> s13
outFile.write("1 14 1\n" + "1 13 1\n")

# 58: s18 -> s13 + s17
outFile.write("1 18 1\n" + "2 13 1 17 1\n")

# 59: s27 -> s13 + s26
outFile.write("1 27 1\n" + "2 13 1 26 1\n")

# 60: s20 -> s19
outFile.write("1 20 1\n" + "1 19 1\n")

outFile.close()
print "Done."


# Write the rate constants.
print "Writing the rate constants file..."
outFile = open(sys.argv[2], "w")
# The number of reactions.
outFile.write("%d\n" % 61)
# The rate constants.
outFile.write("""2.54
1
0.254
1
0.0254
10
254
10000
0.000254
0.01
0.000254
1
0.000254
1
2.54
1
2540
1000
0.0254
1
6.62
0.5
20
0.03
0.03
0.03
0.03
0.03
1.67
0.5
20
0.03
0.03
0.00625
0.5
7
0.03
3
0.7
0.5
1
0.5
20
0.03
0.03
2.54
10000
0.43333
0.5
20
0.03
0.03
2.54
10000
0.03
0.03
0.03
0.03
0.03
0.03
0.03
""")
outFile.close()
print "Done."


# Write the populations.
print "Writing the populations file..."
outFile = open(sys.argv[3], "w")
# The number of species.
outFile.write("%d\n" % 28)
outFile.write("""0
0
0
0
1
4.64567e+06
1324
80
16
3413
29
584
1
22
0
171440
9150
2280
6
596
0
13
3
3
7
0
260
0
""")
outFile.close()
print "Done."
