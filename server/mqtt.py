import paho.mqtt.client as mqttClient

class PiMessager:
    def __init__(self):
        self.broker_address= "raspberrypi.local"
        self.port = 1883
        self.connected = False

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print("Connected to broker")
            self.connected = True
        else:
            print("Connection failed")

    def connect(self):
        self.client = mqttClient.Client("server")
        self.client.on_connect=self.on_connect
        self.client.connect(self.broker_address, port=self.port)
        self.client.loop_start()
    
    def send_message(self, message):
        if self.connected:
            self.client.publish("mqtt/led-screen", message)