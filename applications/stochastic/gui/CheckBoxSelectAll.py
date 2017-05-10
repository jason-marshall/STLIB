"""Check box select or deselect all of the items."""

import wx
import os, os.path
import sys

# If we are running the test code.
if os.path.split(os.getcwd())[1] == 'gui':
    resourcePath = '../'
else:
    from resourcePath import resourcePath

class CheckBoxSelectAll(wx.Panel):
    """Check box that can select or deselect all of the items."""
    def __init__(self, parent, title, size=(180, 100), choices=[]):
        wx.Panel.__init__(self, parent)

        self.isSingleSelectionMode = False

        bmp = wx.Image(os.path.join(resourcePath, 'gui/icons/16x16/add.png'),
                       wx.BITMAP_TYPE_PNG).ConvertToBitmap()
        self.checkButton = wx.BitmapButton(self, -1, bmp)
        if sys.platform in ('win32', 'win64'):
            bmp = wx.Image(os.path.join(resourcePath,
                                        'gui/icons/16x16/addDisabled.png'),
                           wx.BITMAP_TYPE_PNG).ConvertToBitmap()
            self.checkButton.SetBitmapDisabled(bmp)
        self.checkButton.SetToolTip(wx.ToolTip('Check each item.'))
        self.Bind(wx.EVT_BUTTON, self.onCheck, self.checkButton)

        bmp = wx.Image(os.path.join(resourcePath,
                                    'gui/icons/16x16/list-remove.png'),
                       wx.BITMAP_TYPE_PNG).ConvertToBitmap()
        self.unCheckButton = wx.BitmapButton(self, -1, bmp)
        if sys.platform in ('win32', 'win64'):
            bmp = wx.Image(os.path.join(resourcePath,
                                        'gui/icons/16x16/list-removeDisabled.png'),
                           wx.BITMAP_TYPE_PNG).ConvertToBitmap()
            self.unCheckButton.SetBitmapDisabled(bmp)
        self.unCheckButton.SetToolTip(wx.ToolTip('Un-check each item.'))
        self.Bind(wx.EVT_BUTTON, self.onUnCheck, self.unCheckButton)
        
        buttonsSizer = wx.BoxSizer(wx.HORIZONTAL)
        buttonsSizer.Add(self.checkButton)
        buttonsSizer.Add(self.unCheckButton)

        self.items = wx.CheckListBox(self, size=size, choices=choices)
        self.Bind(wx.EVT_CHECKLISTBOX, self.onSelected, self.items)

        sizer = wx.BoxSizer(wx.VERTICAL)
        row = wx.BoxSizer(wx.HORIZONTAL)
        row.Add(buttonsSizer)
        if title:
            row.Add(wx.StaticText(self, -1, ' ' + title))
        sizer.Add(row)
        sizer.Add(self.items, 1, wx.EXPAND)
        self.SetSizer(sizer)

    def set(self, choices):
        self.items.Set(choices)

    def get(self):
        result = []
        for i in range(self.items.GetCount()):
            if self.items.IsChecked(i):
                result.append(i)
        return result

    def areEqual(self, choices):
        """Return true if the choices are the same as those specified. Pass
        None for the choices to avoid this check."""
        if choices == None:
            return True
        if len(choices) != self.items.GetCount():
            return False
        for i in range(len(choices)):
            if choices[i] != self.items.GetString(i):
                return False
        return True

    def onCheck(self, event):
        self.checkAll(True)

    def onUnCheck(self, event):
        self.checkAll(False)

    def onSelected(self, event):
        if self.isSingleSelectionMode:
            self.checkAll(False)
            self.selection = event.GetSelection()
            wx.CallAfter(self.checkSingle)
        else:
            event.Skip()

    def checkSingle(self):
        self.items.Check(self.selection, True)

    def checkAll(self, value):
        for i in range(self.items.GetCount()):
            self.items.Check(i, value)

    def checkList(self, indices):
        for i in indices:
            self.items.Check(i, True)

    def setSingle(self):
        """Set the list to single-selection mode."""
        self.isSingleSelectionMode = True
        self.checkButton.Disable()
        self.unCheckButton.Disable()
        self.checkAll(False)
        if self.items.GetCount() != 0:
            self.items.Check(0, True)

    def setMultiple(self):
        """Set the list to multiple-selection mode."""
        self.isSingleSelectionMode = False
        self.checkButton.Enable()
        self.unCheckButton.Enable()
        self.checkAll(True)

    def enable(self):
        self.checkButton.Enable()
        self.unCheckButton.Enable()
        self.items.Enable()

    def disable(self):
        self.checkButton.Disable()
        self.unCheckButton.Disable()
        self.items.Disable()

def main():
    class TestFrame(wx.Frame):
        def __init__(self, parent=None):
            wx.Frame.__init__(self, parent, title='Check box',
                              size=(200,150))
            self.panel = CheckBoxSelectAll(self, 'Species', size=(180,100),
                                           choices=['s1', 's2', 's3'])
            assert self.panel.areEqual(['s1', 's2', 's3'])
            assert not self.panel.areEqual(['s1', 's2', 's4'])
            assert not self.panel.areEqual(['s1', 's2'])
            assert not self.panel.areEqual(['s1', 's2', 's3', 's4'])
            assert not self.panel.areEqual([])
        def setSingle(self):
            self.panel.setSingle()

    app = wx.PySimpleApp()
    # Multiple
    frame = TestFrame()
    frame.Show()
    # Single selection.
    frame = TestFrame()
    frame.setSingle()
    frame.Show()
    app.MainLoop()

if __name__ == '__main__':
    main()
