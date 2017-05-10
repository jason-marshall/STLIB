"""Implements the HTML window."""

import wx
import wx.html

class HtmlWindow(wx.html.HtmlWindow):
    """The HTML display window.

    Links to local files are opened in this browser.  External files are
    opened in an external browser."""

    def __init__(self, parent):
        """Just construct the base class."""
        wx.html.HtmlWindow.__init__(self, parent)

    def OnLinkClicked(self, link):
        """Decide whether to open the linked document in this browser or 
        to open in an external browser."""
        # If this is a local link.
        if link.GetHref()[0] == '#':
            # Open the link in this window.
            return wx.html.HtmlWindow.OnLinkClicked(self, link)
        else:
            # Open the page in an external browser.
            import webbrowser
            # Check the version.
            import platform
            version = platform.python_version_tuple()
            if 10 * int(version[0]) + int(version[1]) >= 25:
                # Open the page in a new tab.
                # This function is new in python 2.5.
                webbrowser.open_new_tab(link.GetHref())
            else:
                # Open the page in a new window.
                webbrowser.open_new(link.GetHref())
                
class TestHtmlWindow(wx.Frame):
    """Test the HTML window."""
    
    def __init__(self, parent=None):
        """Read and render the HTML help file."""
        wx.Frame.__init__(self, parent, -1, 'Test', size=(800,600))
        html = HtmlWindow(self)
        html.LoadPage('help.html')

def main():
    app = wx.PySimpleApp()
    frame = TestHtmlWindow()
    frame.Show()
    app.MainLoop()

if __name__ == '__main__':
    main()
