# Note: This is a repository for Sean Mauch's STLIB project (old and not under development here)#

# README #

This library collects many of my projects. The source code in the stlib directory is
primarily composed of C++ header files. To use a library just add the stlib directory to your include path and include the appropriate header files. There is usually no need to build a compiled library (.a or .so file).

The unit tests are in the test/unit directory. There are example command-line programs in the examples directory and the test/performance directory. I use [SCons](http://www.scons.org/) to build the programs. 

# Documentation #

The source code is documented using [Doxygen](http://www.stack.nl/~dimitri/doxygen/). To build the documentation go to the doc directory and execute "scons". I periodically post a snapshot of the built
[HTML documentation](http://www.cacr.caltech.edu/~sean/projects/stlib/html/index.html).

# License #

This is free and unencumbered software released into the public domain.

Anyone is free to copy, modify, publish, use, compile, sell, or
distribute this software, either in source code form or as a compiled
binary, for any purpose, commercial or non-commercial, and by any
means.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.
