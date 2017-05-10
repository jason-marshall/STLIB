"""Implements the about box."""

import wx
from HtmlWindow import HtmlWindow

class About(wx.Dialog):
    """The about box."""
    
    text = '''
    <html>
    <body>
    <h1 align="center">
    Cain
    </h1>
    
    <p>Version 1.6.</p>

    <p>
    Developed at 
    <a href="http://www.cacr.caltech.edu/">CACR</a> at the
    <a href="http://www.caltech.edu/">California Institute of Technology</a>
    by 
    <a href="http://www.its.caltech.edu/~sean/">Sean Mauch</a>.
    </p>

    <p>
    Available from
    <a href="http://cain.sourceforge.net/">http://cain.sourceforge.net/</a>.
    </p>
    </body>
    </html>'''

    def __init__(self, parent=None):
        """."""
        wx.Dialog.__init__(self, parent, -1, 'About Cain.', size=(400,400))
        html = HtmlWindow(self)
        html.SetPage(self.text)
        button = wx.Button(self, wx.ID_OK, "Close")

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(html, 1, wx.EXPAND | wx.ALL, 5)
        sizer.Add(button, 0, wx.ALIGN_CENTER | wx.ALL, 5)

        self.SetSizer(sizer)
        self.Layout()

def main():
    app = wx.PySimpleApp()
    frame = About()
    frame.ShowModal()
    frame.Destroy()
    app.MainLoop()

if __name__ == '__main__':
    main()
