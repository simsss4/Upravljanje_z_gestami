import { Injectable } from '@angular/core';
import { Subject, Observable } from 'rxjs';
import mqtt from 'mqtt';

@Injectable({
  providedIn: 'root',
})
export class MqttService {
  private client!: mqtt.MqttClient;
  private messageSubject = new Subject<{ topic: string; payload: string }>();
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 5000; // 5 seconds

  constructor() {
    this.connect();
  }

  connect() {
    this.client = mqtt.connect('mqtt://10.147.20.65:9001', {
      reconnectPeriod: this.reconnectDelay,
      clientId: `angular_client_${Math.random().toString(16).slice(3)}`,
    });

    this.client.on('connect', () => {
      console.log('✅ MQTT connected');
      this.reconnectAttempts = 0;
      
      this.client.subscribe(['environment/#', 'drowsiness/#', 'gestures/#'], (err) => {
        if (err) {
          console.error('Subscription error:', err);
        } else {
          console.log('Subscribed to environment/#, drowsiness/#, gestures/#');
        }
      });
    });

    this.client.on('message', (topic: string, message: Buffer) => {
      try {
        const payload = message.toString();
        console.log(`[${topic}]: ${payload}`);
        this.messageSubject.next({ topic, payload });
      } catch (e) {
        console.error('Failed to parse MQTT message:', e);
      }
    });

    this.client.on('error', (err: any) => {
      console.error('MQTT Error:', err);
    });

    this.client.on('close', () => {
      console.warn('MQTT connection closed');
      this.reconnectAttempts++;
      if (this.reconnectAttempts >= this.maxReconnectAttempts) {
        console.error('Max reconnect attempts reached. Please check broker.');
      }
    });

    this.client.on('reconnect', () => {
      console.log(`Attempting to reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
    });
  }

  getMessages(): Observable<{ topic: string; payload: string }> {
    return this.messageSubject.asObservable();
  }

  publish(topic: string, payload: any) {
    if (this.client && this.client.connected) {
      const message = typeof payload === 'string' ? payload : JSON.stringify(payload);
      this.client.publish(topic, message);
      console.log(`📤 Sent to [${topic}]: ${message}`);
    } else {
      console.warn('MQTT client not connected!');
    }
  }

  disconnect() {
    if (this.client && this.client.connected) {
      this.client.end();
      console.log('MQTT client disconnected');
    }
  }
}