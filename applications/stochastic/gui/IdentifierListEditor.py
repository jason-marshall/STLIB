"""Identifier list editor."""

import sys
# If we are running the unit tests.
if __name__ == '__main__':
    sys.path.insert(1, '..')
    resourcePath = '../'
    from state.Utilities import getNewIntegerString
else:
    from resourcePath import resourcePath

import os.path
import wx

from StateModified import StateModified

class IdentifierListEditorButtons(wx.Panel):
    def __init__(self, parent, useDuplicate):
        wx.Panel.__init__(self, parent)

        bmp = wx.Image(os.path.join(resourcePath, 'gui/icons/16x16/add.png'),
                       wx.BITMAP_TYPE_PNG).ConvertToBitmap()
        self.insert = wx.BitmapButton(self, -1, bmp)
        if sys.platform in ('win32', 'win64'):
            bmp = wx.Image(os.path.join(resourcePath,
                                        'gui/icons/16x16/addDisabled.png'),
                           wx.BITMAP_TYPE_PNG).ConvertToBitmap()
            self.insert.SetBitmapDisabled(bmp)
        self.insert.SetToolTip(wx.ToolTip('Insert.'))

        bmp = wx.Image(os.path.join(resourcePath,
                                    'gui/icons/16x16/editcopy.png'),
                       wx.BITMAP_TYPE_PNG).ConvertToBitmap()
        self.clone = wx.BitmapButton(self, -1, bmp)
        if sys.platform in ('win32', 'win64'):
            bmp = wx.Image(os.path.join(resourcePath,
                                        'gui/icons/16x16/editcopyDisabled.png'),
                           wx.BITMAP_TYPE_PNG).ConvertToBitmap()
            self.clone.SetBitmapDisabled(bmp)
        self.clone.SetToolTip(wx.ToolTip('Clone.'))

        if useDuplicate:
            bmp = wx.Image(os.path.join(resourcePath,
                                        'gui/icons/16x16/chess-board.png'),
                           wx.BITMAP_TYPE_PNG).ConvertToBitmap()
            self.duplicate = wx.BitmapButton(self, -1, bmp)
            if sys.platform in ('win32', 'win64'):
                bmp = wx.Image(os.path.join(resourcePath,
                                            'gui/icons/16x16/chess-boardDisabled.png'),
                               wx.BITMAP_TYPE_PNG).ConvertToBitmap()
                self.duplicate.SetBitmapDisabled(bmp)
            self.duplicate.SetToolTip(wx.ToolTip('Duplicate.'))
        else:
            self.duplicate = None

        bmp = wx.Image(os.path.join(resourcePath, 'gui/icons/16x16/cancel.png'),
                       wx.BITMAP_TYPE_PNG).ConvertToBitmap()
        self.delete = wx.BitmapButton(self, -1, bmp)
        if sys.platform in ('win32', 'win64'):
            bmp = wx.Image(os.path.join(resourcePath,
                                        'gui/icons/16x16/cancelDisabled.png'),
                           wx.BITMAP_TYPE_PNG).ConvertToBitmap()
            self.delete.SetBitmapDisabled(bmp)
        self.delete.SetToolTip(wx.ToolTip('Delete.'))

        bmp = wx.Image(os.path.join(resourcePath, 'gui/icons/16x16/up.png'),
                       wx.BITMAP_TYPE_PNG).ConvertToBitmap()
        self.moveUp = wx.BitmapButton(self, -1, bmp)
        if sys.platform in ('win32', 'win64'):
            bmp = wx.Image(os.path.join(resourcePath,
                                        'gui/icons/16x16/upDisabled.png'),
                           wx.BITMAP_TYPE_PNG).ConvertToBitmap()
            self.moveUp.SetBitmapDisabled(bmp)
        self.moveUp.SetToolTip(wx.ToolTip('Move up.'))

        bmp = wx.Image(os.path.join(resourcePath, 'gui/icons/16x16/down.png'),
                       wx.BITMAP_TYPE_PNG).ConvertToBitmap()
        self.moveDown = wx.BitmapButton(self, -1, bmp)
        if sys.platform in ('win32', 'win64'):
            bmp = wx.Image(os.path.join(resourcePath,
                                        'gui/icons/16x16/downDisabled.png'),
                           wx.BITMAP_TYPE_PNG).ConvertToBitmap()
            self.moveDown.SetBitmapDisabled(bmp)
        self.moveDown.SetToolTip(wx.ToolTip('Move down.'))

        sizer = wx.BoxSizer(wx.VERTICAL)

        row = wx.BoxSizer(wx.HORIZONTAL)
        row.Add(self.insert)
        row.Add(self.clone)
        if self.duplicate:
            row.Add(self.duplicate)
        sizer.Add(row)

        row = wx.BoxSizer(wx.HORIZONTAL)
        row.Add(self.delete)
        row.Add(self.moveUp)
        row.Add(self.moveDown)
        sizer.Add(row)
        self.SetSizer(sizer)
        self.disable()

    def enable(self, up, down):
        """Enable the clone, duplicate and delete buttons. Enable the move 
        up button if up is true. Likewise for the move down button."""
        self.clone.Enable()
        if self.duplicate:
            self.duplicate.Enable()
        self.delete.Enable()
        self.moveUp.Enable(up)
        self.moveDown.Enable(down)

    def disable(self):
        """Disable the clone, duplicate, delete, move up and mode down
        buttons."""
        self.clone.Disable()
        if self.duplicate:
            self.duplicate.Disable()
        self.delete.Disable()
        self.moveUp.Disable()
        self.moveDown.Disable()

class IdentifierListEditor(wx.Panel, StateModified):
    def __init__(self, parent, title, insert, clone, duplicate, edit, delete,
                 toolTip=None):
        """Parameters:
        parent: The parent widget.
        title: The title for the identifiers. Displayed in a large font over
        the list.
        insert: A function that returns an identifier.
        clone: A function that takes an identifier and returns an identifier.
        duplicate: A function that takes an identifier and returns an
        identifier.
        edit: A function that takes the old identifier and the new identifier
        and returns true if the change is acceptable.
        delete: A function that deletes the item.
        toolTip: A tool tip string to attach to the title.
        """
        wx.Panel.__init__(self, parent)

        self.title = wx.StaticText(self, -1, title)
        if toolTip:
            self.title.SetToolTip(wx.ToolTip(toolTip))
        self.insert = insert
        self.clone = clone
        self.duplicate = duplicate
        self.edit = edit
        self.delete = delete

        self.buttons = IdentifierListEditorButtons(self, bool(self.duplicate))
        self.Bind(wx.EVT_BUTTON, self.onInsert, self.buttons.insert)
        self.Bind(wx.EVT_BUTTON, self.onClone, self.buttons.clone)
        if self.duplicate:
            self.Bind(wx.EVT_BUTTON, self.onDuplicate, self.buttons.duplicate)
        self.Bind(wx.EVT_BUTTON, self.onDelete, self.buttons.delete)
        self.Bind(wx.EVT_BUTTON, self.onMoveUp, self.buttons.moveUp)
        self.Bind(wx.EVT_BUTTON, self.onMoveDown, self.buttons.moveDown)

        # List with single selection.
        self.list = wx.ListCtrl(self, -1, style=wx.LC_REPORT | 
                                wx.LC_SINGLE_SEL | wx.LC_NO_HEADER |
                                wx.LC_EDIT_LABELS)
        self.list.InsertColumn(0, '')

        # Item selection.
        self.Bind(wx.EVT_LIST_ITEM_SELECTED, self.onItemSelected, self.list)
        self.Bind(wx.EVT_LIST_ITEM_DESELECTED, self.onItemDeselected, self.list)

        # Label editing.
        self.Bind(wx.EVT_LIST_END_LABEL_EDIT, self.onEndLabelEdit, self.list)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.title, 0, wx.ALL, 0)
        sizer.Add(wx.StaticLine(self), 0, wx.EXPAND|wx.ALL, 1)
        sizer.Add(self.buttons, 0)
        sizer.Add(self.list, 1, wx.EXPAND)
        self.SetSizer(sizer)
        self.Fit()

    def clear(self):
        self.list.DeleteAllItems()

    def select(self):
        if self.list.GetItemCount():
            self.list.SetItemState(0, wx.LIST_STATE_SELECTED, 
                                   wx.LIST_STATE_SELECTED)

    def selectLast(self):
        size = self.list.GetItemCount()
        if size:
            self.list.SetItemState(size - 1, wx.LIST_STATE_SELECTED, 
                                   wx.LIST_STATE_SELECTED)

    def getText(self, index):
        return self.list.GetItemText(index)

    def insertItem(self, id):
        self.list.InsertStringItem(self.list.GetItemCount(), id)
        self.list.SetColumnWidth(0, wx.LIST_AUTOSIZE)

    def getSelectedIndex(self):
        """Return the index of the selected item, or -1 if no item is 
        selected."""
        for item in range(self.list.GetItemCount()):
            if self.list.GetItemState(item, wx.LIST_STATE_SELECTED):
                return item
        return -1

    def getSelectedText(self):
        """Return the text of the selected item, or None if no item is 
        selected."""
        index = self.getSelectedIndex()
        if index == -1:
            return None
        return self.list.GetItemText(index)

    def enableDelete(self):
        self.buttons.delete.Enable()

    def disableDelete(self):
        self.buttons.delete.Disable()

    def onInsert(self, event):
        index = self.list.GetItemCount()
        self.list.InsertStringItem(index, self.insert())
        self.list.SetItemState(index, wx.LIST_STATE_SELECTED, 
                               wx.LIST_STATE_SELECTED)
        self.list.SetColumnWidth(0, wx.LIST_AUTOSIZE)
        event.Skip()
        self.processEventStateModified()

    def onClone(self, event):
        index = self.getSelectedIndex()
        # If an item is selected.
        if index != -1:
            # This will be a valid identifier if the item can be cloned.
            cloneId = self.clone(self.list.GetItemText(index))
            if cloneId:
                self.list.InsertStringItem(index + 1, cloneId)
                self.list.SetItemState(index + 1, wx.LIST_STATE_SELECTED, 
                                       wx.LIST_STATE_SELECTED)
                self.list.SetColumnWidth(0, wx.LIST_AUTOSIZE)
        event.Skip()
        self.processEventStateModified()

    def onDuplicate(self, event):
        index = self.getSelectedIndex()
        # If an item is selected.
        if index != -1:
            # This will be a valid identifier if the item can be duplicated.
            newId = self.duplicate(self.list.GetItemText(index))
            if newId:
                self.list.InsertStringItem(index + 1, newId)
                self.list.SetItemState(index + 1, wx.LIST_STATE_SELECTED, 
                                       wx.LIST_STATE_SELECTED)
                self.list.SetColumnWidth(0, wx.LIST_AUTOSIZE)
        event.Skip()
        self.processEventStateModified()

    def onDelete(self, event):
        item = self.getSelectedIndex()
        # If an item is selected.
        if item != -1:
            self.delete(self.list.GetItemText(item))
            self.list.DeleteItem(item)
            self.list.SetColumnWidth(0, wx.LIST_AUTOSIZE)
        event.Skip()
        self.processEventStateModified()

    def onMoveUp(self, event):
        item = self.getSelectedIndex()
        # If an item is selected and we can move it up.
        if item != -1 and item != 0:
            tmp = self.list.GetItemText(item - 1)
            self.list.SetItemText(item - 1, self.list.GetItemText(item))
            self.list.SetItemState(item - 1, wx.LIST_STATE_SELECTED, 
                                   wx.LIST_STATE_SELECTED)
            self.list.SetItemText(item, tmp)
        event.Skip()
        # Note: Since the order is not stored, this does not modify the state.

    def onMoveDown(self, event):
        item = self.getSelectedIndex()
        # If an item is selected and we can move it up.
        if item != -1 and item != self.list.GetItemCount() - 1:
            tmp = self.list.GetItemText(item + 1)
            self.list.SetItemText(item + 1, self.list.GetItemText(item))
            self.list.SetItemState(item + 1, wx.LIST_STATE_SELECTED, 
                                   wx.LIST_STATE_SELECTED)
            self.list.SetItemText(item, tmp)
        event.Skip()
        # Note: Since the order is not stored, this does not modify the state.

    def onEndLabelEdit(self, event):
        if not event.IsEditCancelled():
            old = self.list.GetItemText(event.GetIndex())
            new = event.GetLabel()
            if old == new or not self.edit(old, new):
                event.Veto()
            else:
                # Manually change the width if necessary. Autosizing will 
                # not work because the text is not changed until the event
                # processing is finished. Manually changing the text caused
                # bus errors.
                extent = self.list.GetTextExtent(new)[0] + 10
                if extent > self.list.GetColumnWidth(0):
                    self.list.SetColumnWidth(0, extent)
                self.processEventStateModified()

    def onItemSelected(self, event):
        """Enable the clone, delete, move up and mode down buttons."""
        index = event.GetIndex()
        self.buttons.enable(index != 0, index != self.list.GetItemCount() - 1)
        event.Skip()

    def onItemDeselected(self, event):
        """Disable the clone, delete, move up and mode down buttons."""
        self.buttons.disable()
        event.Skip()

#
# Test code.
#

class State:
    def __init__(self):
        self.identifiers = []

    def insert(self):
        id = getNewIntegerString(self.identifiers)
        self.identifiers.append(id)
        return id

    def clone(self, id):
        return self.insert()

    def duplicate(self, id):
        return self.insert()

    def edit(self, old, new):
        return not new in self.identifiers

    def delete(self, id):
        return True

class IdentifierListEditorFrame(wx.Frame):
    def __init__(self, parent=None):
        wx.Frame.__init__(self, parent, title='Identifier List Editor',
                          size=(500,300))
        self.state = State()
        editor = IdentifierListEditor(self, 'Identifiers',
                                      insert=self.state.insert, 
                                      clone=self.state.clone,
                                      duplicate=self.state.duplicate,
                                      edit=self.state.edit,
                                      delete=self.state.delete)

def main():
    app = wx.PySimpleApp()
    frame = IdentifierListEditorFrame()
    frame.Show()
    app.MainLoop()

if __name__ == '__main__':
    main()
