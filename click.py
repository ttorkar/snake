# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 12:02:59 2017

@author: trent
"""
import win32api, win32con
def click(x, y):
    win32api.SetCursorPos((x,y))
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,x,y,0,0)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,x,y,0,0)
    