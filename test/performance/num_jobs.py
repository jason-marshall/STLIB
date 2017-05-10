# -*- python -*-

from SCons.Script import *

def cpuCount():
    """Return the number of available cores."""
    try:
        # The multiprocessing module was introduced in Python 2.6. It has been
        # backported 2.5 and 2.4 and is included in some distributions of these.
        import multiprocessing
        return multiprocessing.cpu_count()
    except:
        return 1

# Set the number of concurrent jobs to the number of available cores.
SetOption('num_jobs', cpuCount())
