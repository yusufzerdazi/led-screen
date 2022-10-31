from turtle import width
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

visualisers = [
    {
        "path": "C:\\Program Files (x86)\\Steam\\steamapps\\common\\VZX Player\\vzx_store_player.exe",
        "name": "vovoid"
    },
    {
        "path": "C:\\Program Files (x86)\\Steam\\steamapps\\common\\projectM\\projectM.exe",
        "name": "projectm"
    }
]

window_width = 160
window_height = 120

width_buffer = 8
top_buffer = 31
border_buffer = 4

class BackgroundTasks(threading.Thread):
    def run(self,*args,**kwargs):
        subprocess.call([visualisers[0]['path']])

t = BackgroundTasks()
t.start()

time.sleep(2)

toplist, winlist = [], []
def enum_cb(hwnd, results):
    winlist.append((hwnd, win32gui.GetWindowText(hwnd)))
win32gui.EnumWindows(enum_cb, toplist)

visualiser_window = [(hwnd, title) for hwnd, title in winlist if visualisers[0]['name'] in title.lower()][0][0]
win32gui.SetWindowPos(visualiser_window, win32con.HWND_TOPMOST, 0, 0, window_width + 2 * width_buffer, window_height + width_buffer + top_buffer, 0) 
time.sleep(2)
bbox = win32gui.GetWindowRect(visualiser_window)
print(bbox)

bounding_box = {'top': bbox[1] + top_buffer, 'left': bbox[0] + width_buffer, 'width': bbox[2] - 2 * width_buffer, 'height': bbox[3] - top_buffer - width_buffer - border_buffer}

sct = mss()
messager = mqtt.PiMessager()
messager.connect()

px = ImageGrab.grab(bbox)
px.save("tst.png")

try:
    while True:
        sct_img = sct.grab(bounding_box)
        sct_array = np.array(sct_img)
        #cv2.imshow('screen', sct_array)
        pixels = []
        x_step = bounding_box['width'] / 40
        y_step = bounding_box['height'] / 30

        for x in range(40):
            for y in range(30):
                x_sample = int(x * x_step + x_step / 2)
                y_sample = int(y * y_step + y_step / 2)
                pixels += [[(x, 30 - y - 1), (int(sct_array[y_sample][x_sample][2]), int(sct_array[y_sample][x_sample][1]), int(sct_array[y_sample][x_sample][0]))]]
                #pixels += [[(x, 30 - y - 1), (int(px.getpixel((x, y))[0]), int(px.getpixel((x, y))[1]), int(px.getpixel((x, y))[2]))]]
        
        messager.send_message(json.dumps({"type": "rgb", "pixels": pixels}))
        win32gui.SetWindowPos(visualiser_window, win32con.HWND_TOPMOST, 0, 0, window_width + 2 * width_buffer, window_height + width_buffer + top_buffer, 0) 
        time.sleep(40 / 1000)

        #if (cv2.waitKey(1) & 0xFF) == ord('q'):
        #    cv2.destroyAllWindows()
        #    break
except KeyboardInterrupt:
    print("Exiting LED server")
    messager.send_message(json.dumps({"type": "blackout"}))
