// src/app/dashboard/mqtt.service.ts

import { Injectable } from '@angular/core';
// @ts-ignore
import mqtt from 'mqtt'; // brez '/dist'

@Injectable({
  providedIn: 'root',
})
export class MqttService {
  private client!: mqtt.MqttClient;

  connect() {
    this.client = mqtt.connect('wss://test.mosquitto.org:8081');

    this.client.on('connect', () => {
      console.log('✅ MQTT connected');
      this.client.subscribe('model/utrujenost');
    });

    this.client.on('message', (topic: string, message: Buffer) => {
      const payload = JSON.parse(message.toString());
      console.log(`📩 [${topic}]:`, payload);
    });

    this.client.on('error', (err: any) => {
      console.error('❌ MQTT Error', err);
    });
  }

  publish(topic: string, payload: any) {
    if (this.client && this.client.connected) {
      this.client.publish(topic, JSON.stringify(payload));
      console.log(`📤 Sent to [${topic}]`, payload);
    } else {
      console.warn('⚠️ MQTT client not connected!');
    }
  }
}
