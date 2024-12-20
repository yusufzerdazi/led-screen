import paho.mqtt.client as mqttClient

class Mqtt:
    def __init__(self, on_message):
        self.broker_address= "raspberrypi2.local"
        self.topic = "mqtt/led-screen"
        self.port = 1883
        self.connected = False
        self.on_message = on_message

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print("Connected to broker")
            self.connected = True
            self.client.subscribe(self.topic)
            self.client.on_message = self.on_message
        else:
            print("Connection failed")

    def connect(self):
        self.client = mqttClient.Client(mqttClient.CallbackAPIVersion.VERSION1, "client")
        self.client.on_connect=self.on_connect
        self.client.connect(self.broker_address, port=self.port)
        self.client.loop_start()