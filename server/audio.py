# Server #
import pyaudio
import numpy
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import json

CHUNK = 512
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
SAMPLE_RATE = 10

class Audio():
    def __init__(self, callback):
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.channelcount = None
        self.rolling = []
        self.callback = callback

    def initialise_stream(self):
        #Set default to first in list or ask Windows
        try:
            default_device_index = self.audio.get_default_input_device_info()
        except IOError:
            default_device_index = -1

        #Select Device
        print ("Available devices:\n")
        for i in range(0, self.audio.get_device_count()):
            info = self.audio.get_device_info_by_index(i)
            print (str(info["index"]) + ": \t %s \n \t %s \n" % (info["name"], self.audio.get_host_api_info_by_index(info["hostApi"])["name"]))

            if(info["name"] == "OUT 1-2 (BEHRINGER UMC 404HD 192k)"):
                device_id = info["index"]

        print (device_id)

        #Get device info
        try:
            device_info = self.audio.get_device_info_by_index(device_id)
        except IOError:
            device_info = self.audio.get_device_info_by_index(default_device_index)
            print ("Selection not available, using default.")

        self.channelcount = device_info["maxInputChannels"] if (device_info["maxOutputChannels"] < device_info["maxInputChannels"]) else device_info["maxOutputChannels"]
        self.stream = self.audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=int(device_info["defaultSampleRate"]),
                        input = True,
                        frames_per_buffer=CHUNK,
                        as_loopback=True,
                        input_device_index=device_info["index"])
    
    def animate(self, i):
        data = numpy.frombuffer(self.stream.read(CHUNK), dtype=numpy.int16)
        p = numpy.log10(numpy.abs(numpy.fft.rfft(data)))
        p = numpy.mean([p[:256], p[::-1][:256]], axis=0)

        self.rolling = self.rolling + [p]
        self.rolling = self.rolling[-1:]
        rolling_mean = numpy.mean(self.rolling, axis=0)
        mean = rolling_mean[:240].reshape(-1, 6).max(axis=1)

        plt.cla()
        plt.bar([i for i in range(len(mean))], [int(j ** 3) for j in mean])

        self.callback(json.dumps({"frequencies": mean.tolist()}))
    
    def start_listening(self):
        ani = FuncAnimation(plt.gcf(), self.animate, interval=20)
        plt.show()