"""Run "make win" to build the windows executable in the dist directory.
You will need to use the MinGW 4.2 compilers. I manually renamed the compiler
executables to make them the default.

Open installer.iss with ISTool. Hit the Compile Setup button. The installer is
in output. Rename it to CainSetup.exe.
"""

from distutils.core import setup
import py2exe

import matplotlib
setup(windows=[{'script':'Cain.py',
                'icon_resources':[(1, 'gui/icons/cain.ico')]}],
      options={'py2exe':{'packages':['matplotlib', 'pytz'],
                         'skip_archive':1,
                         'dll_excludes':['libgdk_pixbuf-2.0-0.dll',
                                         'libgobject-2.0-0.dll',
                                         'libgdk-win32-2.0-0.dll']}},
      data_files=list(matplotlib.get_py2exe_datafiles()))
#setup(console=['generateTrajectories.py'])

# Old method.
if False:
    setup(windows=[{'script':'Cain.py',
                    'icon_resources':[(1, 'gui/icons/cain.ico')]}])

# In this method directories are not copied.
if False:
    import glob
    data = glob.glob(r'c:\Python25\Lib\site-packages\matplotlib\*')
    data.append(r'matplotlibrc')

    setup(windows=[{'script':'Cain.py',
                    'icon_resources':[(1, 'gui/icons/cain.ico')]}],
          data_files=[('matplotlibdata', data)])

