import numpy as np
import cv2
from mss import mss
from PIL import Image
import json
import mqtt

bounding_box = {'top': 100, 'left': 0, 'width': 40, 'height': 30}

sct = mss()
messager = mqtt.PiMessager()
messager.connect()

while True:
    sct_img = sct.grab(bounding_box)
    sct_array = np.array(sct_img)
    cv2.imshow('screen', sct_array)
    pixels = []

    for x in range(len(sct_array)):
        for y in range(len(sct_array[x])):
            pixels += [(x, y), (int(sct_array[x][y][0]), int(sct_array[x][y][1]), int(sct_array[x][y][2]))]

    messager.send_message(json.dumps({"type": "pixels", "pixels": pixels}))

    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        cv2.destroyAllWindows()
        break
