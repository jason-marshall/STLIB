"""Compile using MSVS."""

import sys
import string
import subprocess
import os.path
import shutil


def main():
    msvsPath = r'c:\Program Files\Microsoft Visual Studio 9.0\VC\bin'
    f = open('compile.bat', 'w')
    f.write(r'"%s"' % os.path.join(msvsPath,  'vcvars32') + '\r\n')
    
    f.write(r'"%s"' % os.path.join(msvsPath,  'cl') + r' /Isrc /EHsc solvers\Direct2DSearch.cc' + '\r\n')
    f.close()
    #subprocess.Popen('compile.bat', shell=True)

if __name__ == '__main__':
    main()
