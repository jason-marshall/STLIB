"""Implements the preferences."""

from PreferencesDialog import PreferencesDialog
from copy import copy
import wx
import sys

class Preferences:
    """The preferences."""
    
    def __init__(self):
	"""."""
        # CONTINUE: It would be nice to add the -m64 flag for 64 bit
        # architectures.
        #if sys.platform in ('win32', 'win64'):
        #    bitsFlag = '-m32 '
        #else:
        #    bitsFlag = '-m64 '
        self.default = [['Compilation',
                         ['Compiler', 'g++', 80],
                         ['Flags', '-O3 -funroll-loops -fstrict-aliasing -finline-limit=6000', 500]],
                        ['SBML',
                         ['Version', ['3', '2', '1'], None]],
                        ['gnuplot',
                         ['X Scale', '1', 80],
                         ['Y Scale', '1', 80],
                         ['Style', ['lines', 'linespoints'], None]]]
        # Reset to the default.
        self.reset()

    def reset(self):
        # Clear the data.
        self.data = {}
        for category in self.default:
            # The first element is the category name.
            d = {}
            self.data[category[0]] = d
            for (field, value, width) in category[1:]:
                # If the value field is a list.
                if type(value) == type([]):
                    # The default is the first element of the list.
                    d[field] = value[0]
                else:
                    # Otherwise it is simply the value.
                    d[field] = value

    def openDialog(self):
        dialog = PreferencesDialog(self)
        if dialog.ShowModal() == wx.ID_OK:
            # CONTINUE: Validate.
            for subject in self.default:
                category = subject[0]
                for (field, value, width) in subject[1:]:
                    if type(value) == type([]):
                        self.data[category][field] = \
                            value[dialog.controls[category][field].\
                                      GetSelection()]
                    else:
                        self.data[category][field] = \
                            dialog.controls[category][field].GetValue()
        dialog.Destroy()
            
        

def main():
    app = wx.PySimpleApp()
    preferences = Preferences()
    preferences.openDialog()
    app.MainLoop()
    for category in preferences.data:
        print category
        for field in preferences.data[category]:
            print '    ', preferences.data[category][field]

if __name__ == '__main__':
    main()
