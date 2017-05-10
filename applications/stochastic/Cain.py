#! /usr/bin/env python

"""The script that launches Cain."""

def main():
    import sys
    import os, os.path
    if not sys.platform in ('win32', 'win64'):
        directory = os.path.dirname(__file__)
        if directory:
            os.chdir(directory)

    errors = []
    # Check for wxPython.
    try:
        import wx
    except:
        errors.append('Error: Unable to load wx. You must install wxPython.')
    # Check for numpy.
    try:
        import numpy
    except:
        errors.append('The python package numpy is not installed. Cain will not work correctly without it. Consult documentation for information on installing numpy.')
    # Check for matplotlib.
    try:
        import matplotlib
    except:
        errors.append('The python package matplotlib is not installed. Plotting will not work without it. Consult documentation for information on installing matplotlib.')

    # If necessary software is missing, show an error message and exit.
    if errors:
        errors.append('Consult the documentation available at http://cain.sourceforge.net/.')
        hasTk = True
        try:
            import Tkinter
        except:
            hasTk = False
        if hasTk:
            import tkMessageBox
            root = Tkinter.Tk()
            #Tkinter.Label(root, text='\n'.join(errors)).pack()
            #root.title('Error')
            tkMessageBox.showwarning('Error', '\n'.join(errors))
            root.mainloop()
        else:
            print('\n'.join(errors))
        sys.exit(1)

    # Launch the application.
    from gui.Application import Application
    if len(sys.argv) > 2:
        print "Warning: Unmatched command line arguments."
    app = Application()
    if len(sys.argv) == 2:
        #app.readInitialFile(sys.argv[1])
        wx.CallAfter(app.readInitialFile, *(sys.argv[1],))
    app.MainLoop()

if __name__ == '__main__':
    main()
