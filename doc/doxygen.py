# -*- python -*-

import os
from SCons.Script import *

doxygen = Environment(ENV=os.environ,
                      BUILDERS={'Doxyfile':
                                    Builder(action='cat $SOURCES >$TARGET'),
                                'Doxygen':Builder(action='doxygen $SOURCE')})
