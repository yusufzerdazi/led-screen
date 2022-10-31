import bottle
import paho.mqtt.client as mqttClient

LAST_MESSAGE = ""

def on_message(client, userdata, msg):
    global LAST_MESSAGE
    LAST_MESSAGE = msg.payload.decode()

client = mqttClient.Client("xblinds2")
client.connect("raspberrypi.local", port=1883)
client.loop_start()

client.subscribe("xblinds/all/status")
client.on_message = on_message

@bottle.route('/blinds/open')
def open_blinds():
    client.publish("xblinds/all", "open")
    return bottle.HTTPResponse(status=200, body="Blinds opened")
    
@bottle.route('/blinds/close')
def open_blinds():
    client.publish("xblinds/all", "close")
    return bottle.HTTPResponse(status=200, body="Blinds closed")

@bottle.route('/blinds/status')
def open_blinds():
    print(LAST_MESSAGE)
    return bottle.HTTPResponse(status=200, body=LAST_MESSAGE)
    
@bottle.route('/dashboard')
def dashboard():
    return bottle.static_file("index.html", root='.')

bottle.run(host='0.0.0.0', port=8081)