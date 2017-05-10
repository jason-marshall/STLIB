# ExtractHistogram.py

import sys

if __name__ == '__main__':
  errorMessage = \
  """Usage:
  python ExtractHistogram.py speciesIndex"""

  if len(sys.argv) != 2:
    print errorMessage
    raise AssertionError, "Wrong number of command line arguments.  Exiting..."

  # Parse the arguments.
  speciesIndex = int(sys.argv[1])

  # "histogram final"
  assert sys.stdin.readline().rstrip() == 'histogram final'
  # number of species
  sys.stdin.readline()
  # number of reactions
  sys.stdin.readline()
  # number of species to record
  numberOfRecordedSpecies = int(sys.stdin.readline())
  # list of species to record
  recordedSpecies = [int(x) for x in sys.stdin.readline().split()]
  assert len(recordedSpecies) == numberOfRecordedSpecies
  assert speciesIndex in recordedSpecies
  index = recordedSpecies.index(speciesIndex)
  # number of bins
  sys.stdin.readline()
  # list of initial MT 19937 state
  sys.stdin.readline()
  # "success"
  assert sys.stdin.readline().rstrip() == 'success'
  # number of trajectories
  sys.stdin.readline()
  # Skip until we reach the desired histogram.
  for i in range(index):
    # lower bound
    sys.stdin.readline()
    # bin width
    sys.stdin.readline()
    # list of weighted probabilities
    sys.stdin.readline()
  # Write the desired histogram.
  # lower bound
  sys.stdout.write(sys.stdin.readline())
  # bin width
  sys.stdout.write(sys.stdin.readline())
  # list of weighted probabilities
  sys.stdout.write(sys.stdin.readline())
