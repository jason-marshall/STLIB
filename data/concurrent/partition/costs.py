import sys
import random

errorMessage = \
"""Usage:
python costs.py function dimension extent0 extent1 ...
- function is one of unit, uniformRandom, firstIndex, or indexSum.
- dimension is the array dimension.
- The extents are the array extents."""


# Cost functions.

def unit(index):
  return 1

def uniformRandom(index):
  return random.random()

def firstIndex(index):
  return index[0] + 1

def indexSum(index):
  sum = 1
  for x in index:
    sum += x
  return sum

# Check the number of arguments.
if len(sys.argv) < 4:
  print errorMessage
  raise "Error: Bad arguments.  Exiting..."

# Get the cost function.
cost = eval(sys.argv[1])

# Get the dimension
dimension = eval(sys.argv[2])
if not (dimension == 2 or dimension == 3):
  print errorMessage
  raise "Error: Bad dimension.  Exiting..."

if dimension == 2:
  extents = (eval(sys.argv[3]), eval(sys.argv[4]))
  size = extents[0] * extents[1]
  print "%d %d" % extents
  for j in range(extents[1]):
    for i in range(extents[0]):
      print cost((i, j))
elif dimension == 3:
  extents = (eval(sys.argv[3]), eval(sys.argv[4]), eval(sys.argv[5]))
  size = extents[0] * extents[1] * extents[2]
  print "%d %d %d" % extents
  for k in range(extents[2]):
    for j in range(extents[1]):
      for i in range(extents[0]):
        print cost((i, j, k))
else:
  raise "Internal error.  Exiting..."

