# Exact.py
# Usage:

import MT19937
import sys

if __name__ == '__main__':
  errorMessage = \
  """Usage:
  python Exact.py endTime maximumAllowedSteps numberOfTrajectories"""

  if len(sys.argv) != 4:
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
  numberOfTrajectories = int(sys.argv[3])

  # 
  # Write the input file.
  #

  # Number of frames
  print '1'
  # List of frame times
  print endTime
  # Maximum allowed steps
  print maximumAllowedSteps
  # Number of parameters.
  print '0'
  # Empty line for the parameters.
  print ''
  # List of MT 19937 state
  sys.stdout.write(MT19937.output())
  # Number of trajectories
  print numberOfTrajectories
