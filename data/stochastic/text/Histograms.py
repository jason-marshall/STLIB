# Histograms.py
# Usage:

import MT19937
import sys

if __name__ == '__main__':
  errorMessage = \
  """Usage:
  python Histograms.py endTime maximumAllowedSteps numberOfBins s0 ... sn numberOfTrajectories"""

  if len(sys.argv) <= 5:
    print errorMessage
    raise AssertionError, "Wrong number of command line arguments.  Exiting..."

  # 
  # Parse the arguments.
  #

  # End time.
  endTime = float(sys.argv[1])
  # Maximum number of allowed steps.
  maximumAllowedSteps = int(sys.argv[2])
  # Number of trajectories.
  numberOfBins = int(sys.argv[3])
  # The recorded species.
  recordedSpecies = [int(x) for x in sys.argv[4:-1]]
  # The number of trajectories.
  numberOfTrajectories = int(sys.argv[-1])

  # 
  # Write the input file.
  #

  # Number of frames
  print '1'
  # List of frame times
  print endTime
  # Maximum allowed steps
  print maximumAllowedSteps
  # Number of species to record.
  print len(recordedSpecies)
  # List of species to record.
  print ('%s ' * len(recordedSpecies)) % tuple(recordedSpecies)
  # Number of bins.
  print numberOfBins
  # Number of parameters.
  print '0'
  # Empty line for the parameters.
  print ''
  # List of MT 19937 state
  sys.stdout.write(MT19937.output())
  # Number of trajectories
  print numberOfTrajectories
