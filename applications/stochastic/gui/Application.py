#! /usr/bin/env python

"""The script that launches Cain."""

import sys
import os.path
import wx
from MainFrame import MainFrame
from resourcePath import resourcePath

class Application(wx.App):
    """The application class for Cain."""

    def __init__(self):
        """Construct the base class and redirect the output."""
        if not sys.platform in ('win32', 'win64'):
            wx.App.__init__(self, redirect=True, filename="ErrorLog.txt")
        else:
            # On windows we might not have write permission for the log file.
            wx.App.__init__(self)

    def OnInit(self):
        # Splash screen.
        image = wx.Image(os.path.join(resourcePath, "gui/splash.png"),
                         wx.BITMAP_TYPE_PNG)
        bmp = image.ConvertToBitmap()
        wx.SplashScreen(bmp, wx.SPLASH_CENTRE_ON_SCREEN | wx.SPLASH_TIMEOUT,
                        2000, None, -1)
        wx.Yield()
        # Main window.
        self.frame = MainFrame()
        self.frame.Show()
        self.SetTopWindow(self.frame)
        return True

    def readInitialFile(self, inputFileName):
        self.frame.readFile(inputFileName, False)
