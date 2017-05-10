"""Implements the volume editor."""

import wx

class VolumeDialog(wx.Dialog):
    """Get the new volume."""
    
    def __init__(self, parent):
        """Construct."""
        wx.Dialog.__init__(self, parent, -1, 'Edit volume.')

        # Buttons.
        okButton = wx.Button(self, wx.ID_OK, 'OK')
        okButton.SetDefault()
        cancelButton = wx.Button(self, wx.ID_CANCEL, 'Cancel')
        buttons = wx.BoxSizer(wx.HORIZONTAL)
        buttons.Add(okButton, 0, wx.ALIGN_RIGHT, 5)
        buttons.Add(cancelButton, 0, wx.ALIGN_RIGHT, 5)

        sizer = wx.BoxSizer(wx.VERTICAL)
        # Volume.
        sizer.Add(wx.StaticText(self, -1, 'Enter the new volume in liters.'),
                  0, wx.ALL, border=5)
        self.volume = wx.TextCtrl(self)
        sizer.Add(self.volume, 0, wx.ALL, border=5)
        # Update mass action propensities.
        self.update = wx.CheckBox(self, -1, 'Update mass action propensities.')
        sizer.Add(self.update, 0, wx.ALL, border=5)
        sizer.Add(buttons, 0, wx.ALIGN_RIGHT | wx.ALIGN_BOTTOM, 5)

        self.SetSizer(sizer)
        self.Layout()
        self.Fit()

class VolumeEditor(wx.Panel):
    """The volume editor."""
    
    def __init__(self, parent, application):
        """."""
        wx.Panel.__init__(self, parent)
        self.application = application

        button = wx.Button(self, label='Volume')
        self.volume = wx.StaticText(self, -1, '')
       
        # Layout with sizers.
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(button, 0, wx.ALL, 5)
        sizer.Add(self.volume, 0, wx.ALL, 5)
        # Add a little extra space to the volume string is not clipped.
        sizer.Add(wx.StaticText(self, -1, ' '), 0, wx.ALL, 5)
        
        self.Bind(wx.EVT_BUTTON, self.onEdit, button)

        self.SetSizer(sizer)
        sizer.Fit(self)

    def setModel(self, model):
        self.model = model
        if self.model:
            self.volume.SetLabel(str(self.model.volume))
        else:
            self.volume.SetLabel('')

    def onEdit(self, event):
        # Make sure that a model is selected.
        if not self.model:
            wx.MessageBox('Error: No model is selected.', 'Error')
            return
        # Volume dialog.
        dialog = VolumeDialog(self)
        if dialog.ShowModal() != wx.ID_OK:
            return
        volumeString = dialog.volume.GetValue()
        doUpdate = dialog.update.IsChecked()
        dialog.Destroy()
        # Check the volume.
        try:
            volume = float(volumeString)
        except:
            wx.MessageBox(volumeString + ' is not a floating-point number.',
                          'Error!', style=wx.OK|wx.ICON_EXCLAMATION)
            return
        if volume <= 0:
            wx.MessageBox(volumeString + ' is not a positive volume.',
                          'Error!', style=wx.OK|wx.ICON_EXCLAMATION)
            return
        # If we are updating the mass action propensities we need to parse
        # the model first.
        if doUpdate:
            self.application.syncModel(self.model.id)
            self.application.parseModel(self.model.id)
        # Set the volume.
        self.model.setVolume(volume, doUpdate)
        # Update the display.
        self.volume.SetLabel(str(self.model.volume))
        self.application.updateVolume(self.model.id)

# Test code.

class VolumeEditorModel:
    def __init__(self):
        self.id = None
        self.volume = 1.

    def setVolume(self, volume, doUpdate):
        self.volume = volume

class VolumeEditorFrame(wx.Frame):
    def __init__(self, parent=None):
        wx.Frame.__init__(self, parent, title='Volume Editor',
                          size=(200,100))
        editor = VolumeEditor(self, self)
        editor.setModel(VolumeEditorModel())

    def updateVolume(self, id):
        pass

    def syncModel(self, id):
        pass

    def parseModel(self, id):
        pass

def main():
    app = wx.PySimpleApp()
    frame = VolumeEditorFrame()
    frame.Show()
    app.MainLoop()

if __name__ == '__main__':
    main()
