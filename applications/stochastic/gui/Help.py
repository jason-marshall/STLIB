"""Implements the help window."""

# If we are running the unit tests.
if __name__ == '__main__':
    import sys
    sys.path.insert(1, '..')
    resourcePath = '../'
else:
    from resourcePath import resourcePath

import os.path
import wx
from HtmlWindow import HtmlWindow

data = ['Acknowledgments',
        'License',
        'Introduction',
        'Installation',
        'Platform-Specific Information',
        ["User's Guide",
         ['Overview',
          'Quick Start',
          'Model List',
          'Method Editor',
          'Recorder',
          'Launcher',
          'Simulation Output',
          'Species Editor',
          'Reaction Editor',
          'Parameter Editor',
          'Compartment Editor',
          'Tool Bar',
          'Mathematical Expressions']],
        ['Examples',
         ['Birth-Death',
          'Immigration-Death']],
        ['Simulation Methods',
         ['Discrete Stochastic Simulations',
          'Direct Method',
          'First Reaction Method',
          'Next Reaction Method',
          'Tau-Leaping',
          'Direct Method with Time-Dependent Propensities',
          'Hybrid Direct/Tau-Leaping',
          'ODE Integration']],
        ['Performance',
         ['Exact Methods']],
        ["Developer's Guide",
         ['Command Line Solvers',
          'File Formats',
          'Adding New Solvers']],
        ['Cain XML File Format',
         ['Top-Level Elements',
          'Models',
          'Methods',
          'Output',
          'Random Numbers']],
        'FAQ',
        'Links',
        'Bibliography',
        'Known Issues',
        'To Do']

class Help(wx.Frame):
    """The help window."""
    
    def __init__(self, parent=None, title='Cain help'):
        """Read and render the HTML help file."""
        wx.Frame.__init__(self, parent, -1, title, size=(1100,750))
        splitter = wx.SplitterWindow(self)

        self.contents = wx.TreeCtrl(splitter,
                                    style=wx.TR_HIDE_ROOT|wx.TR_DEFAULT_STYLE)
        root = self.contents.AddRoot('Cain Help')
        self.addTreeNodes(root, data)
        self.Bind(wx.EVT_TREE_SEL_CHANGED, self.onActivated, self.contents)

        self.html = HtmlWindow(splitter)
        if "gtk2" in wx.PlatformInfo:
            self.html.SetStandardFonts()
        # The window title.
        self.html.SetRelatedFrame(self, self.GetTitle() + " -- %s")
        # Status bar.
        self.CreateStatusBar()
        self.html.SetRelatedStatusBar(0)

        splitter.SetMinimumPaneSize(10)
        splitter.SplitVertically(self.contents, self.html, 200)

        self.html.LoadPage(os.path.join(resourcePath, 'gui/help.html'))

    def addTreeNodes(self, parent, items):
        for item in items:
            if type(item) == str:
                self.contents.AppendItem(parent, item)
            else:
                newItem = self.contents.AppendItem(parent, item[0])
                self.addTreeNodes(newItem, item[1])

    def onActivated(self, event):
        label = self.contents.GetItemText(event.GetItem())
        self.html.OnLinkClicked(wx.html.HtmlLinkInfo('#' + label))

def main():
    app = wx.PySimpleApp()
    frame = Help()
    frame.Show()
    app.MainLoop()

if __name__ == '__main__':
    main()
