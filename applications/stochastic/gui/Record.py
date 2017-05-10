"""The species and reactions to record."""

import wx
import sys
import os, os.path

from CheckBoxSelectAll import CheckBoxSelectAll

class Record(wx.Notebook):
    """The species and reactions to record."""
    
    def __init__(self, parent):
        wx.Notebook.__init__(self, parent)

        # CONTINUE: I want to display this information for the tabs, not on
        # the whole window. Currently this does not work for linux.
        if sys.platform in ('darwin', 'win32', 'win64'):
            self.SetToolTip(wx.ToolTip('Before you launch a simulation, select the species and reactions to record. For histogram output you can record only species populations. If you change a model, hit the refresh button to update the lists.'))
        self.species = CheckBoxSelectAll(self, None, size=(150, 200))
        self.reactions = CheckBoxSelectAll(self, None, size=(150, 200))
        self.AddPage(self.species, 'Species', True)
        self.AddPage(self.reactions, 'Reactions', False)

    def set(self, species, reactions):
        self.species.set(species)
        self.reactions.set(reactions)

    def get(self):
        """Get the checked species and the checked reactions."""
        return (self.species.get(), self.reactions.get())

    def isValid(self, species, reactions):
        """Return true if the lists of the species and reactions are the same
        as those specified."""
        return self.species.areEqual(species) and\
            self.reactions.areEqual(reactions)

    def checkAll(self, value):
        self.species.checkAll(value)
        self.reactions.checkAll(value)

    def checkSpecies(self, value):
        self.species.checkAll(value)

    def checkList(self, species, reactions):
        self.species.checkList(species)
        self.reactions.checkList(reactions)

    def enable(self):
        self.species.enable()
        self.reactions.enable()

    def disable(self):
        self.species.disable()
        self.reactions.disable()


if __name__ == '__main__':
    class TestFrame(wx.Frame):
        def __init__(self, parent=None):
            wx.Frame.__init__(self, parent, title='Record',
                              size=(300,400))
            panel = Record(self)
            panel.set(['s1', 's2', 's3'],
                      ['TheFirstReaction', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7',
                       'r8'])
            assert panel.isValid(['s1', 's2', 's3'],
                                 ['TheFirstReaction', 'r2', 'r3', 'r4', 'r5',
                                  'r6', 'r7', 'r8'])
            assert not panel.isValid(['s1', 's2', 's4'],
                                     ['TheFirstReaction', 'r2', 'r3', 'r4',
                                      'r5', 'r6', 'r7', 'r8'])

    app = wx.PySimpleApp()
    frame = TestFrame()
    frame.Show()
    app.MainLoop()
