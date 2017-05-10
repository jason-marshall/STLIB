"""Message windows."""

# If we are running the unit tests.
import sys
if __name__ == '__main__':
    sys.path.insert(1, '..')

import wx
from HtmlWindow import HtmlWindow

class UpdateVersionFrame(wx.Frame):
    def __init__(self, parent):
        wx.Frame.__init__(self, parent, -1, 'Software update.')
        html = HtmlWindow(self)
        content = """A new version of Cain is available from
        <a href="http://cain.sourceforge.net/">http://cain.sourceforge.net/</a>."""
        html.SetPage(content)

class ScrolledMessagePanel(wx.Panel):
    def __init__(self, parent, message, title, size):
        wx.Panel.__init__(self, parent)
        self.parent = parent

        sizer = wx.BoxSizer(wx.VERTICAL)

        scroll = wx.ScrolledWindow(self)
        text = wx.StaticText(scroll, -1, message)
        bestSize = scroll.GetBestSize()
        scroll.SetScrollbars(1, 1, bestSize.GetWidth(), bestSize.GetHeight())
        sizer.Add(scroll, 1, wx.ALIGN_TOP | wx.EXPAND, 5)

        self.button = wx.Button(self, -1, 'OK')
        self.button.SetDefault()
        self.Bind(wx.EVT_BUTTON, self.onOk, self.button)
        sizer.Add(self.button, 0, wx.ALIGN_RIGHT | wx.ALIGN_TOP, 5)

        self.SetSizer(sizer)
        self.Fit()

    def onOk(self, event):
        self.parent.Destroy()

class ScrolledMessageFrame(wx.Frame):
    """Show message in a scrolled window."""
    def __init__(self, message, title, size=(800,600), parent=None):
        wx.Frame.__init__(self, parent, -1, title, size=size)
        ScrolledMessagePanel(self, message, title, size)

class CompilationError(wx.Frame):
    """Show compilation errors."""
    
    def __init__(self, errors, parent=None, title='Compilation Errors'):
        wx.Frame.__init__(self, parent, -1, title, size=(800,600))

        scroll = wx.ScrolledWindow(self, -1)
        # Replace left and right single quotes.
        errors = errors.replace('\xe2\x80\x98', "'")
        errors = errors.replace('\xe2\x80\x99', "'")
        text = wx.StaticText(scroll, -1, errors)
        size = text.GetBestSize()
        scroll.SetScrollbars(1, 1, size[0], size[1])

class CompilingMessage(wx.Frame):
    """Show that compilation is proceeding. This window disables closing. The
    application destroys the window when compilation is finished."""
    
    def __init__(self, parent=None, title='Compilation in progress.'):
        wx.Frame.__init__(self, parent, -1, title, size=(300,50))
        wx.StaticText(self, -1, 'Compiling...')
        self.Bind(wx.EVT_CLOSE, self.onClose)

    def onClose(self, event):
        if event.CanVeto():
            event.Veto()
        else:
            self.Destroy()

def truncatedMessageBox(message, caption='Message', style=wx.OK, size=1000):
    """Show a message box. Truncate the message if it is too long."""
    wx.MessageBox(message[0:min(size,len(message))], caption, style)

def truncatedErrorBox(message, size=1000):
    """Show an error box. Truncate the message if it is too long."""
    truncatedMessageBox(message, 'Error!', wx.OK|wx.ICON_EXCLAMATION, size)

def openWrite(filename):
    """Try to open the file in write mode. Return the file if it can be opened.
    Otherwise show an error message return None."""
    try:
        return open(filename, 'w')
    except:
        wx.MessageBox('Unable to write to the file ' + filename
                      + '. Save in a directory in which you have write '
                      + 'permissions.',
                      'Error!', style=wx.OK|wx.ICON_EXCLAMATION)
    return None

def main():
    app = wx.PySimpleApp()
    message = '\n'.join(['Long error message-------------------------------------------------------------------------number ' + str(n) + '.' for n in range(1,61)])
    messageFrame = ScrolledMessageFrame(message, 'Errors', (600, 300))
    messageFrame.Show()
    app.MainLoop()

if __name__ == '__main__':
    main()
