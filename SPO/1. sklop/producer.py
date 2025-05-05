import paho.mqtt.client as mqtt
import time
from datetime import datetime

broker = "mosquitto"
port = 1883
topic = "/project/data"

def on_connect(client, userdata, flags, reasonCode, properties=None):
    print(f"Connected to MQTT Broker: {reasonCode}")

producer = mqtt.Client(client_id="project_producer", callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
producer.on_connect = on_connect
producer.connect(broker, port, 60)

while True:
    message = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ret = producer.publish(topic, message, qos=1, retain=False)
    print(f"Sending: {message} [rc: {ret.rc}]")
    time.sleep(2)