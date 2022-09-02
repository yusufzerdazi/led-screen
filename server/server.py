
import time
import mqtt
import audio

messager = mqtt.PiMessager()
messager.connect()

audio = audio.Audio(messager.send_message)
audio.initialise_stream()
audio.start_listening()