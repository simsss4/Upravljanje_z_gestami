import paho.mqtt.client as mqtt

broker = "172.25.70.243"
port = 1883
topic = "/project/data"

def on_connect(client, userdata, flags, reasonCode, properties=None):
    print(f"Connected to MQTT Broker: {reasonCode}")

def on_subscribe(client, userdata, mid, granted_qos, properties=None):
    print("Subscribed to topic")

def on_message(client, userdata, msg):
    print(f"Received: {msg.payload.decode()}")

client = mqtt.Client(client_id="project_consumer", clean_session=False, callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
client.on_connect = on_connect
client.on_subscribe = on_subscribe
client.on_message = on_message
client.connect(broker, port, 60)
client.subscribe(topic, qos=1)
client.loop_forever()