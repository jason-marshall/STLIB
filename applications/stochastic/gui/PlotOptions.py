"""Implements plotting of time series data."""

# If we are running the unit tests.
import sys
if __name__ == '__main__':
    sys.path.insert(1, '..')

import wx
from colorsys import hsv_to_rgb

# CONTINUE
# It is better to use the WXAgg renderer than the WX renderer.
# However, the WXAgg renderer crashes on Mac OS X with some versions of
# matplotlib. Since it works with EPD 6, I will use WXAgg.

# CONTINUE
#if sys.platform != 'darwin':
import matplotlib
matplotlib.interactive(True)
matplotlib.use('WXAgg')

from pylab import gca, legend, title, xlabel, ylabel, xlim, ylim

# Legend locations.
legendLocations = (
    ('upper right', 1),
    ('upper left', 2),
    ('lower left', 3),
    ('lower right', 4),
    ('right', 5),
    ('center left', 6),
    ('center right', 7),
    ('lower center', 8),
    ('upper center', 9),
    ('center', 10)
    )
legendLocationNames = [x[0] for x in list(legendLocations)]
legendLocationValues = [x[1] for x in list(legendLocations)]

def wxToMatplotlibColor(color):
    return (color.Red() / 255.0, color.Green() / 255.0, color.Blue() / 255.0)

def makeBitmap(color):
    """Make a bitmap of the specified color."""
    bmp = wx.EmptyBitmap(16, 16)
    dc = wx.MemoryDC()
    dc.SelectObject(bmp)
    dc.SetBackground(wx.Brush(color))
    dc.Clear()
    dc.SelectObject(wx.NullBitmap)
    return bmp

class ColorButton(wx.Panel):
    def __init__(self, parent, color = wx.BLACK):
        wx.Panel.__init__(self, parent, -1)
        self.color = wx.BLACK
        self.colorButton = wx.BitmapButton(self, -1, makeBitmap(self.color))
        self.Bind(wx.EVT_BUTTON, self.onClick, self.colorButton)

    def onClick(self, event):
        # Get a color.
        color = wx.GetColourFromUser(None)
        if color.IsOk():
            self.color = color
            self.colorButton.SetBitmapLabel(makeBitmap(color))
        event.Skip()

    def setColor(self, color = wx.BLACK):
        """Set the color."""
        self.color = color
        self.colorButton.SetBitmapLabel(makeBitmap(self.color))
        
    def setColorHue(self, hue):
        """Set the color using the specified hue. The hue must be in the range 
        [0..1]. The saturation and value are unity."""
        assert 0 <= hue and hue <= 1
        rgb = hsv_to_rgb(hue, 1, 1)
        self.color = wx.Colour(rgb[0] * 255, rgb[1] * 255, rgb[2] * 255)
        self.colorButton.SetBitmapLabel(makeBitmap(self.color))
        
textSizes = ['xx-small', 'x-small', 'small', 'medium', 'large', 'x-large',
             'xx-large']
defaultTextSizeIndex = 3

class TextStyle(wx.Panel):
    def __init__(self, parent, color = wx.BLACK):
        wx.Panel.__init__(self, parent, -1)
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.sizeChoices = ['xx-small', 'x-small', 'small', 'medium', 'large',
                            'x-large', 'xx-large']

        self.colorButton = ColorButton(self, color)
        sizer.Add(self.colorButton)

        self.size = wx.Choice(self, choices=textSizes)
        self.size.SetSelection(defaultTextSizeIndex)
        sizer.Add(self.size)

        self.text = wx.TextCtrl(self, -1, '', size=(400, 12))
        sizer.Add(self.text, 1, wx.EXPAND)

        self.SetSizer(sizer)
        self.Layout()

    def reset(self):
        self.colorButton.setColor()
        self.size.SetSelection(defaultTextSizeIndex)
        self.text.SetValue('')

    def getSize(self):
        return textSizes[self.size.GetSelection()]

class PlotOptions(wx.Panel):
    
    def __init__(self, parent):
        wx.Panel.__init__(self, parent, -1)
        sizer = wx.BoxSizer(wx.VERTICAL)

        # Legend
        box = wx.BoxSizer(wx.HORIZONTAL)
        self.legend = wx.CheckBox(self, -1, 'Legend')
        self.legend.SetValue(True)
        box.Add(self.legend, 0, wx.ALIGN_CENTER_VERTICAL)
        # Text size.
        self.legendSize = wx.Choice(self, choices=textSizes)
        self.legendSize.SetSelection(defaultTextSizeIndex)
        box.Add(self.legendSize, 0, wx.ALIGN_CENTER_VERTICAL)
        # Location.
        self.legendLocation = wx.Choice(self, choices=legendLocationNames)
        self.legendLocation.SetSelection(0)
        box.Add(self.legendLocation, 0, wx.ALIGN_CENTER_VERTICAL)
        sizer.Add(box, 0, wx.ALIGN_TOP, 5)

        # Axes tick labels.
        box = wx.BoxSizer(wx.HORIZONTAL)
        box.Add(wx.StaticText(self, -1, 'Tick labels'), 0,
                wx.ALIGN_CENTER_VERTICAL)
        self.tickLabelsSize = wx.Choice(self, choices=textSizes)
        self.tickLabelsSize.SetSelection(defaultTextSizeIndex)
        box.Add(self.tickLabelsSize, 0, wx.ALIGN_CENTER_VERTICAL)
        sizer.Add(box, 0, wx.ALIGN_TOP, 5)

        # Labels.
        grid = wx.FlexGridSizer(3, 2)
        grid.AddGrowableCol(1, 0)
        # Title.
        grid.Add(wx.StaticText(self, -1, 'Title'), 0, wx.ALIGN_CENTER_VERTICAL)
        self.title = TextStyle(self)
        grid.Add(self.title, 0, wx.EXPAND, border=5)
        # X label.
        grid.Add(wx.StaticText(self, -1, 'X label'), 0, 
                 wx.ALIGN_CENTER_VERTICAL)
        self.xLabel = TextStyle(self)
        grid.Add(self.xLabel, 0, wx.EXPAND, border=5)
        # Y label.
        grid.Add(wx.StaticText(self, -1, 'Y label'), 0, 
                 wx.ALIGN_CENTER_VERTICAL)
        self.yLabel = TextStyle(self)
        grid.Add(self.yLabel, 0, wx.EXPAND, border=5)
        sizer.Add(grid, 0, wx.EXPAND, 5)

        # Domain.
        box = wx.BoxSizer(wx.HORIZONTAL)
        box.Add(wx.StaticText(self, -1, 'X axis limits ('), 0,
                wx.ALIGN_CENTER_VERTICAL)
        self.xmin = wx.TextCtrl(self, -1, '', size=(50, -1))
        box.Add(self.xmin, 0, wx.ALIGN_CENTER_VERTICAL, border=5)
        box.Add(wx.StaticText(self, -1, ','), 0, wx.ALIGN_CENTER_VERTICAL)
        self.xmax = wx.TextCtrl(self, -1, '', size=(50, -1))
        box.Add(self.xmax, 0, wx.ALIGN_CENTER_VERTICAL, border=5)
        box.Add(wx.StaticText(self, -1, '), Y axis limits ('), 0,
                wx.ALIGN_CENTER_VERTICAL)
        self.ymin = wx.TextCtrl(self, -1, '', size=(50, -1))
        box.Add(self.ymin, 0, wx.ALIGN_CENTER_VERTICAL, border=5)
        box.Add(wx.StaticText(self, -1, ','), 0, wx.ALIGN_CENTER_VERTICAL)
        self.ymax = wx.TextCtrl(self, -1, '', size=(50, -1))
        box.Add(self.ymax, 0, wx.ALIGN_CENTER_VERTICAL, border=5)
        box.Add(wx.StaticText(self, -1, ')'), 0, wx.ALIGN_CENTER_VERTICAL)
        sizer.Add(box, 0, wx.ALIGN_TOP, 5)

        # Custom figure size.
        box = wx.BoxSizer(wx.HORIZONTAL)
        self.customFigureSize = wx.CheckBox(self, -1,
                                            'Custom figure size in inches (')
        self.customFigureSize.SetValue(False)
        box.Add(self.customFigureSize, 0, wx.ALIGN_CENTER_VERTICAL)
        self.figureSizeX = wx.TextCtrl(self, -1, '', size=(50, -1))
        box.Add(self.figureSizeX, 0, wx.ALIGN_CENTER_VERTICAL, border=5)
        box.Add(wx.StaticText(self, -1, ','), 0, wx.ALIGN_CENTER_VERTICAL)
        self.figureSizeY = wx.TextCtrl(self, -1, '', size=(50, -1))
        box.Add(self.figureSizeY, 0, wx.ALIGN_CENTER_VERTICAL, border=5)
        box.Add(wx.StaticText(self, -1, ')'), 0,
                wx.ALIGN_CENTER_VERTICAL)
        sizer.Add(box, 0, wx.ALIGN_TOP, 5)

        self.SetSizer(sizer)
        self.Fit()

    def showLegendAndLabels(self):
        # Before calling this function, make empty plots to register the
        # labels for the legend.
        self.showLegend()
        self.showLabels()

    def showLegend(self):
        if self.legend.IsChecked():
            legend(loc=legendLocationValues[self.legendLocation.GetSelection()],
                   prop={'size':textSizes[self.legendSize.GetSelection()]})

    def showLabels(self):
        # Tick labels.
        for label in gca().xaxis.get_majorticklabels() +\
                gca().yaxis.get_majorticklabels():
            label.set_fontsize(textSizes[self.tickLabelsSize.GetSelection()])
        # Title.
        t = self.title.text.GetValue()
        if t:
            title(t, color=wxToMatplotlibColor(self.title.colorButton.color),
                  size=self.title.getSize())
        # Axes labels.
        t = self.xLabel.text.GetValue()
        if t:
            xlabel(t, color=wxToMatplotlibColor(self.xLabel.colorButton.color),
                  size=self.xLabel.getSize())
        t = self.yLabel.text.GetValue()
        if t:
            ylabel(t, color=wxToMatplotlibColor(self.yLabel.colorButton.color),
                  size=self.yLabel.getSize())

    def getCustomFigureSize(self):
        """Return a tuple of the figure size in inches if the custom figure
        size box is checked. Otherwise return None. If the size is invalid
        show a warning message and return None."""
        if self.customFigureSize.IsChecked():
            x, y = 0, 0
            try:
                x = float(self.figureSizeX.GetValue())
                y = float(self.figureSizeY.GetValue())
            except:
                pass
            if x > 0 and y > 0:
                return (x, y)
            else:
                wx.MessageBox('The custom figure size is not valid. ' +
                              'The default size will be used instead.',
                              'Error!', style=wx.OK|wx.ICON_EXCLAMATION)
        return None

    def _parseLimit(self, textControl):
        """Get the user specified limit. An unspecified value is indicated
        with None. Show an error message if the limit cannot be parsed."""
        t = textControl.GetValue()
        if t:
            try:
                return float(t)
            except:
                wx.MessageBox('The limit "' + t + '" is not valid. ' +
                              'The default limit will be used instead.',
                              'Error!', style=wx.OK|wx.ICON_EXCLAMATION)
        return None

    def setLimits(self):
        """Set any user specified plot range limits."""
        v = self._parseLimit(self.xmin)
        if v is not None:
            xlim(xmin=v)
        v = self._parseLimit(self.xmax)
        if v is not None:
            xlim(xmax=v)
        v = self._parseLimit(self.ymin)
        if v is not None:
            ylim(ymin=v)
        v = self._parseLimit(self.ymax)
        if v is not None:
            ylim(ymax=v)
                
        
def main():

    class TestPlotOptions(wx.Frame):
        """Test the PlotOptions panel."""

        def __init__(self, parent, title):
            wx.Frame.__init__(self, parent, -1, title)
            panel = PlotOptions(self)

            #size = self.GetBestSize()
            #self.SetSize(size)
            self.Fit()

    app = wx.PySimpleApp()
    TestPlotOptions(None, 'Options.').Show()
    app.MainLoop()

if __name__ == '__main__':
    main()

