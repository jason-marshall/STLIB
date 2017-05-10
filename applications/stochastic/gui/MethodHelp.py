"""Help message for the selected method. This class is no longer used."""

# If we are running the unit tests.
if __name__ == '__main__':
    import sys
    sys.path.insert(1, '..')

import re
import wx
from HtmlWindow import HtmlWindow
from state.simulationMethods import timeDependence, timeDependenceDocStrings,\
    categories, categoryDocStrings, methods, methodDocStrings, options,\
    optionDocStrings, bibliography

class MethodHelp(wx.Frame):
    """The help window."""
    
    def __init__(self, time, category, method, option, parent=None):
	"""Render the HTML help file."""
	wx.Frame.__init__(self, parent, -1, 'Method information',
                          size=(600,400))
	self.html = HtmlWindow(self)
	if "gtk2" in wx.PlatformInfo:
	    self.html.SetStandardFonts()
        source =\
            [''.join(['<!DOCTYPE HTML PUBLIC "-//IETF//DTD HTML//EN"><html><body>',
                      '<p><tt>', timeDependence[time], '</tt> - ',
                      timeDependenceDocStrings[time], '</p>',
                      '<p><tt>', categories[time][category], '</tt> - ',
                      categoryDocStrings[time][category], '</p>',
                      '<p><tt>', methods[time][category][method], '</tt> - ',
                      methodDocStrings[time][category][method], '</p>',
                      '<p><tt>', options[time][category][method][option],
                      '</tt> - ',
                      optionDocStrings[time][category][method][option], '</p>',
                      '<p><ul>'])]
        for key in bibliography:
            if re.search(key, source[0]):
                source.append('<li>' + bibliography[key])
        source.append('</ul></p></body></html>')
	self.html.SetPage(''.join(source))

def main():
    app = wx.PySimpleApp()
    frame = MethodHelp(0,0, 0, 0)
    frame.Show()
    app.MainLoop()

if __name__ == '__main__':
    main()
