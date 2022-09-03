import numpy as np
import cv2
from mss import mss
from PIL import Image
import json
import mqtt
import time
from PIL import ImageGrab, Image
import win32gui
import win32con
import subprocess
import threading

class BackgroundTasks(threading.Thread):
    def run(self,*args,**kwargs):
        subprocess.call(['C:\\Program Files (x86)\\Steam\\steamapps\\common\\VZX Player\\vzx_store_player.exe'])

t = BackgroundTasks()
t.start()

time.sleep(3)

toplist, winlist = [], []
def enum_cb(hwnd, results):
    winlist.append((hwnd, win32gui.GetWindowText(hwnd)))
win32gui.EnumWindows(enum_cb, toplist)

vovoid = [(hwnd, title) for hwnd, title in winlist if 'vovoid' in title.lower()][0][0]
win32gui.SetWindowPos(vovoid, win32con.HWND_TOPMOST, 5080, 1362, 40, 30, 0) 
bbox = win32gui.GetWindowRect(vovoid)

bounding_box = {'top': bbox[0], 'left': bbox[1], 'width': bbox[2], 'height': bbox[3]}

sct = mss()
messager = mqtt.PiMessager()
messager.connect()

while True:
    #px = ImageGrab.grab(bbox).resize((40,30), resample=0)
    sct_img = sct.grab(bounding_box)
    sct_array = np.array(sct_img)
    #cv2.imshow('screen', sct_array)
    pixels = []

    for x in range(40):
        for y in range(30):
            x_sample = x * int(bounding_box['width'] / 40)
            y_sample = y * int(bounding_box['height'] / 30)
            pixels += [[(x, 30 - y - 1), (int(sct_array[y_sample][x_sample][2]), int(sct_array[y_sample][x_sample][1]), int(sct_array[y_sample][x_sample][0]))]]
            #pixels += [[(x, 30 - y - 1), (int(px.getpixel((x, y))[0]), int(px.getpixel((x, y))[1]), int(px.getpixel((x, y))[2]))]]
    
    messager.send_message(json.dumps({"type": "rgb", "pixels": pixels}))
    win32gui.SetWindowPos(vovoid, win32con.HWND_TOPMOST, 5080, 1362, 40, 30, 0) 
    time.sleep(20 / 1000)

    #if (cv2.waitKey(1) & 0xFF) == ord('q'):
    #    cv2.destroyAllWindows()
    #    break
