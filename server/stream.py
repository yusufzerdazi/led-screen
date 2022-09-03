import numpy as np
import cv2
from mss import mss
from PIL import Image
import json
import mqtt
import time
from PIL import ImageGrab
import win32gui

def get_pixel_colour(i_x, i_y):
	import win32gui
	i_desktop_window_id = win32gui.GetDesktopWindow()
	i_desktop_window_dc = win32gui.GetWindowDC(i_desktop_window_id)
	long_colour = win32gui.GetPixel(i_desktop_window_dc, i_x, i_y)
	i_colour = int(long_colour)
	win32gui.ReleaseDC(i_desktop_window_id,i_desktop_window_dc)
	return (i_colour & 0xff), ((i_colour >> 8) & 0xff), ((i_colour >> 16) & 0xff)

#px = ImageGrab.grab().load()
#bounding_box = {'top': 0, 'left': 0, 'width': 2560, 'height': 1440}

#sct = mss()
messager = mqtt.PiMessager()
messager.connect()

while True:
    #sct_img = sct.grab(bounding_box)
    #sct_array = np.array(sct_img)
    #cv2.imshow('screen', sct_array)
    #px = ImageGrab.grab().load()
    pixels = []

    for x in range(40):
        for y in range(30):
            x_sample = x * 64
            y_sample = y * 48
            pixel = get_pixel_colour(x_sample, y_sample)
            pixels += [[(x, 30 - y - 1), (int(pixel[0]), int(pixel[1]), int(pixel[2]))]]
            
    messager.send_message(json.dumps({"type": "rgb", "pixels": pixels}))
    time.sleep(50 / 1000)

    #if (cv2.waitKey(1) & 0xFF) == ord('q'):
    #    cv2.destroyAllWindows()
    #    break
