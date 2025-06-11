import { CommonModule } from '@angular/common';
import { Component, OnInit, OnDestroy, ChangeDetectorRef } from '@angular/core';
import { RouterLink } from '@angular/router';
import { MqttService } from '../../app/Services/mqtt.service';
import { Subscription } from 'rxjs';

type WeatherType = 'dan' | 'noč' | 'jasno' | 'deževno' | 'megleno';
type GestureType =
  | 'dvig_roke'
  | 'spust_roke'
  | 'horizontalno_desno'
  | 'horizontalno_levo'
  | 'stisnjena_pest'
  | 'stisnjena_pest_izklop'
  | 'fan_gor'
  | 'fan_dol'
  | 'temp_gor'
  | 'temp_dol';

type GestureFunction =
  | 'zapri_levo_okno_spredaj'
  | 'odpri_levo_okno_spredaj'
  | 'zapri_desno_okno_spredaj'
  | 'odpri_desno_okno_spredaj'
  | 'zapri_levo_okno_zadaj'
  | 'odpri_levo_okno_zadaj'
  | 'zapri_desno_okno_zadaj'
  | 'odpri_desno_okno_zadaj'
  | 'glasnost_gor'
  | 'glasnost_dol'
  | 'radio_postaja_prev'
  | 'radio_postaja_next'
  | 'vklop_radio'
  | 'izklop_radio'
  | 'zapiranje_vzvratnega_ogledala'
  | 'odpiranje_vzvratnega_ogledala'
  | 'premik_kota_vzvratnega_ogledala_v_levo'
  | 'premik_kota_vzvratnega_ogledala_v_desno'
  | 'premik_kota_vzvratnega_ogledala_navzgor'
  | 'premik_kota_vzvratnega_ogledala_navzdol'
  | 'zviševanje_moči_pihanja_klimatske_naprave'
  | 'zniževanje_moči_pihanja_klimatske_naprave'
  | 'zviševanje_nastavitve_temperature'
  | 'zniževanje_nastavitve_temperature';

interface GestureData {
  type: GestureType;
  functionality: GestureFunction;
}

interface RadioStation {
  frequency: string;
  name: string;
}

@Component({
  selector: 'app-dashboard',
  imports: [RouterLink, CommonModule],
  providers: [MqttService],
  templateUrl: './dashboard.html',
  styleUrl: './dashboard.css',
})
export class Dashboard implements OnInit, OnDestroy {
  gesture?: GestureData;
  isModalOpen = false;
  alerts: string[] = [];
  warnings: string[] = [];
  dashboardTheme = 'normal';
  panelTheme = 'normal';

  radioOn = true;
  volumeLevel: number = 40;
  currentStationIndex: number = 4;
  fanSpeed: number = 3;
  temperature: number = 21.5;

  showAlertIcon = false;
  showWarningIcon = false;

  private mqttSubscription!: Subscription;

  private topSlotIcons: string[] = ['alert', 'warning'];
  private bottomSlotIcons: string[] = ['dnevne', 'kratke', 'meglenkle'];

  private isDriverExhausted: boolean = false;

  private gestureDataMap: GestureData[] = [
    { type: 'dvig_roke', functionality: 'glasnost_gor' },
    { type: 'spust_roke', functionality: 'glasnost_dol' },
    { type: 'horizontalno_desno', functionality: 'radio_postaja_next' },
    { type: 'horizontalno_levo', functionality: 'radio_postaja_prev' },
    { type: 'stisnjena_pest', functionality: 'vklop_radio' },
    { type: 'stisnjena_pest_izklop', functionality: 'izklop_radio' },
    {
      type: 'fan_gor',
      functionality: 'zviševanje_moči_pihanja_klimatske_naprave',
    },
    {
      type: 'fan_dol',
      functionality: 'zniževanje_moči_pihanja_klimatske_naprave',
    },
    { type: 'temp_gor', functionality: 'zviševanje_nastavitve_temperature' },
    { type: 'temp_dol', functionality: 'zniževanje_nastavitve_temperature' },
  ];

  private stations: RadioStation[] = [
    { frequency: '89.3 MHz', name: 'Val 202' },
    { frequency: '90.5 MHz', name: 'Radio 1' },
    { frequency: '92.5 MHz', name: 'Radio Aktual' },
    { frequency: '93.7 MHz', name: 'Radio Antena' },
    { frequency: '96.7 MHz', name: 'Radio Veseljak' },
    { frequency: '97.7 MHz', name: 'Radio Celje' },
    { frequency: '98.9 MHz', name: 'Radio Center' },
    { frequency: '101.3 MHz', name: 'Radio Ekspres' },
    { frequency: '104.9 MHz', name: 'Radio Maribor' },
    { frequency: '88.8 MHz', name: 'Radio Si' },
  ];

  private weatherResponses: Record<
    string,
    { alerts: string[]; warnings: string[] }
  > = {
    dan: {
      alerts: ['Vklop dnevnih luči!'],
      warnings: [
        'Zaznan: Dan, priporočena uporaba sončnih očal. Vozite previdno!',
        'Svetloba ekrana povečana ++',
      ],
    },
    noč: {
      alerts: ['Vklop kratkih luči!'],
      warnings: [
        'Zaznana: Noč, držite varnostno razdaljo in vozite previdno!',
        'Svetloba ekrana zmanjšana --',
      ],
    },
    jasno: {
      alerts: [],
      warnings: ['Normalna svetloba ekrana', 'Vozite varno!'],
    },
    megleno: {
      alerts: ['Vklop meglenk! Držite varnostno razdaljo!'],
      warnings: ['Pomanjšana vidljivost!', 'Vozite previdno!'],
    },
    deževno: {
      alerts: ['Vklop brisalcev', 'Priporočena vožnja: 60 km/h'],
      warnings: ['Mokro cestišče'],
    },
  };

  private functionalityHandlers: Record<GestureFunction, () => void> = {
    glasnost_gor: () => this.increaseVolume(),
    glasnost_dol: () => this.decreaseVolume(),
    radio_postaja_prev: () => this.previousRadioStation(),
    radio_postaja_next: () => this.nextRadioStation(),
    vklop_radio: () => this.turnOnRadio(),
    izklop_radio: () => this.turnOffRadio(),
    zviševanje_moči_pihanja_klimatske_naprave: () => this.increaseFanSpeed(),
    zniževanje_moči_pihanja_klimatske_naprave: () => this.decreaseFanSpeed(),
    zviševanje_nastavitve_temperature: () => this.increaseTemperature(),
    zniževanje_nastavitve_temperature: () => this.decreaseTemperature(),
    // Prazne izvedbe funkcij, za tiste geste ki mapirajo samo na izpise
    odpri_levo_okno_spredaj: () => {},
    zapri_levo_okno_spredaj: () => {},
    odpri_desno_okno_spredaj: () => {},
    zapri_desno_okno_spredaj: () => {},
    odpri_levo_okno_zadaj: () => {},
    zapri_levo_okno_zadaj: () => {},
    odpri_desno_okno_zadaj: () => {},
    zapri_desno_okno_zadaj: () => {},
    zapiranje_vzvratnega_ogledala: () => {},
    odpiranje_vzvratnega_ogledala: () => {},
    premik_kota_vzvratnega_ogledala_v_levo: () => {},
    premik_kota_vzvratnega_ogledala_v_desno: () => {},
    premik_kota_vzvratnega_ogledala_navzgor: () => {},
    premik_kota_vzvratnega_ogledala_navzdol: () => {},
  };

  constructor(
    private mqttService: MqttService,
    private cdr: ChangeDetectorRef
  ) {}

  ngOnInit() {
    this.mqttSubscription = this.mqttService.getMessages().subscribe(
      ({ topic, payload }) => this.handleMqttMessage(topic, payload),
      (error) => console.error('❌ MQTT subscription error', error)
    );
  }

  ngOnDestroy() {
    if (this.mqttSubscription) {
      this.mqttSubscription.unsubscribe();
    }
  }

  private handleMqttMessage(topic: string, payload: any) {
    console.log(`Processing MQTT message: ${topic} -> ${payload}`);
    const parts = topic.split('/');
    const mainTopic = parts[0];

    switch (mainTopic) {
      case 'environment':
        this.handleEnvironmentMessage(parts[1], payload);
        break;
      case 'drowsiness':
        this.handleDrowsinessMessage(parts[1], payload);
        break;
      case 'gestures':
        this.handleGestureMessage(parts[1], payload);
        break;
      default:
        console.warn(`⚠️ Unhandled MQTT topic: ${topic}`);
    }
  }

  private envMessageBuffer: { timeofday?: string; weather?: string } = {};
  private handleEnvironmentMessage(subTopic: string, payload: string) {
    if (subTopic === 'timeofday' || subTopic === 'weather') {
      this.envMessageBuffer[subTopic] = payload;

      if (this.envMessageBuffer.timeofday && this.envMessageBuffer.weather) {
        this.processEnvironment(
          this.envMessageBuffer.timeofday,
          this.envMessageBuffer.weather
        );

        this.envMessageBuffer = {};
      }
    } else if (subTopic === 'alerts') {
      this.alerts = [payload];
      this.flashTopIcon('alert', 5000);
      this.cdr.detectChanges();
    } else if (subTopic === 'metrics') {
      console.log(`Environment metrics: ${payload}`);
    }
  }
  private processEnvironment(timePayload: string, weatherPayload: string) {
    const timeLabel = this.parsePayloadLabel(timePayload);
    const weatherLabel = this.parsePayloadLabel(weatherPayload);

    const timeType = this.mapToWeatherType(timeLabel);
    const weatherType = this.mapToWeatherType(weatherLabel);

    console.log(
      `Processing combined payloads. [Time type: ${timeType}, Weather type: ${weatherType}]`
    );

    if (weatherType) {
      const weatherData = this.weatherResponses[weatherType];

      this.alerts = [];
      this.warnings = [];

      this.alerts = [...weatherData.alerts];
      this.warnings = [...weatherData.warnings];

      console.log('Updated alerts:', this.alerts);
      console.log('Updated warnings:', this.warnings);
      this.panelTheme = '';

      if (this.alerts.length > 0) {
        this.flashTopIcon('alert', 5000);
      }
      if (this.warnings.length > 0) {
        this.flashTopIcon('warning', 3000);
      }

      this.setNormalTheme();

      if (timeType === 'noč') {
        this.setDimTheme();
        this.panelTheme = 'dim';
        this.showIconInSlot('kratke', 'bottom');
      } else if (timeType === 'dan') {
        this.showIconInSlot('dnevne', 'bottom');
      }

      if (weatherType === 'megleno') {
        this.showIconInSlot('meglenkle', 'bottom');
      }
      if (weatherType === 'jasno') {
        this.setBrightTheme();
      } else if (!['noč', 'dan', 'megleno'].includes(weatherType)) {
        this.clearDashboardIcons();
      }

      this.cdr.detectChanges();
    } else {
      console.warn(`⚠️ Invalid weather type: ${weatherLabel}`);
    }
  }

  private handleGestureMessage(subTopic: string, payload: string) {
    const gesture = this.parsePayloadLabel(payload);
    const functionality = this.mapToGestureFunction(subTopic, gesture);
    if (functionality) {
      this.clearDashboardIcons();
      this.warnings = [];
      const alertMessage = this.translateFunctionalityToAlert(functionality);
      this.alerts = [alertMessage];
      this.flashTopIcon('alert', 5000);
      this.cdr.detectChanges();

      const specifiedGestures: GestureFunction[] = [
        'glasnost_gor',
        'glasnost_dol',
        'radio_postaja_prev',
        'radio_postaja_next',
        'vklop_radio',
        'izklop_radio',
        'zviševanje_moči_pihanja_klimatske_naprave',
        'zniževanje_moči_pihanja_klimatske_naprave',
        'zviševanje_nastavitve_temperature',
        'zniževanje_nastavitve_temperature',
      ];

      if (specifiedGestures.includes(functionality)) {
        const gestureType = this.gestureDataMap.find(
          (g) => g.functionality === functionality
        )?.type;
        if (gestureType) {
          this.gesture = { type: gestureType, functionality };
          console.log(`Gesture: ${gestureType}, Function: ${functionality}`);

          const handler = this.functionalityHandlers[functionality];
          if (handler) {
            handler();
          } else {
            console.warn(
              `No handler defined for functionality: ${functionality}`
            );
          }
        } else {
          console.warn(
            `No gesture type mapped for functionality: ${functionality}`
          );
        }
      } else {
        console.log(`Alert-only gesture: ${functionality}`);
      }
    } else {
      console.warn(`Invalid gesture: ${gesture} for topic ${subTopic}`);
    }
  }

  private parsePayloadLabel(payload: string): string {
    return payload.split(' ')[0];
  }

  private mapToWeatherType(label: string): WeatherType | null {
    const mapping: Record<string, WeatherType> = {
      day: 'dan',
      night: 'noč',
      clear: 'jasno',
      foggy: 'megleno',
      rainy: 'deževno',
    };
    return mapping[label] || null;
  }

  private mapToGestureFunction(
    subTopic: string,
    gesture: string
  ): GestureFunction | null {
    const mapping: Record<string, Record<string, GestureFunction>> = {
      okna: {
        open_front_left_window: 'odpri_levo_okno_spredaj',
        close_front_left_window: 'zapri_levo_okno_spredaj',
        open_front_right_window: 'odpri_desno_okno_spredaj',
        close_front_right_window: 'zapri_desno_okno_spredaj',
        open_back_left_window: 'odpri_levo_okno_zadaj',
        close_back_left_window: 'zapri_levo_okno_zadaj',
        open_back_right_window: 'odpri_desno_okno_zadaj',
        close_back_right_window: 'zapri_desno_okno_zadaj',
      },
      radio: {
        turn_on_radio: 'vklop_radio',
        turn_off_radio: 'izklop_radio',
        next_station: 'radio_postaja_next',
        previous_station: 'radio_postaja_prev',
        volume_up: 'glasnost_gor',
        volume_down: 'glasnost_dol',
      },
      vzvratna_ogledala: {
        close_rm: 'zapiranje_vzvratnega_ogledala',
        open_rm: 'odpiranje_vzvratnega_ogledala',
        left_rm: 'premik_kota_vzvratnega_ogledala_v_levo',
        right_rm: 'premik_kota_vzvratnega_ogledala_v_desno',
        up_rm: 'premik_kota_vzvratnega_ogledala_navzgor',
        down_rm: 'premik_kota_vzvratnega_ogledala_navzdol',
      },
      klimatska_naprava: {
        climate_warmer: 'zviševanje_nastavitve_temperature',
        climate_colder: 'zniževanje_nastavitve_temperature',
        fan_stronger: 'zviševanje_moči_pihanja_klimatske_naprave',
        fan_weaker: 'zniževanje_moči_pihanja_klimatske_naprave',
      },
    };
    return mapping[subTopic]?.[gesture] || null;
  }

  private handleDrowsinessMessage(subTopic: string, payload: string) {
    if (subTopic === 'status' && payload === 'Utrujen') {
      this.isDriverExhausted = true;
      this.alerts = ['Zaznana utrujenost voznika! Priporočamo odmor.'];
      this.flashTopIcon('alert', 5000);
      this.cdr.detectChanges();
    } else if (subTopic === 'status' && payload === 'Buden') {
      this.isDriverExhausted = false;
      this.clearDashboardIcons();
      this.alerts = [];
    } else if (subTopic === 'metrics') {
      console.log(`Drowsiness metrics: ${payload}`);
    }
    this.cdr.detectChanges();
  }

  get volumeBars(): string {
    return '|'.repeat(Math.floor(this.volumeLevel / 2));
  }
  private increaseVolume() {
    if (this.volumeLevel < 100) {
      this.volumeLevel += 10;
    }
  }

  private decreaseVolume() {
    if (this.volumeLevel > 0) {
      this.volumeLevel -= 5;
    }
  }

  get currentStation(): RadioStation {
    return this.stations[this.currentStationIndex];
  }

  private nextRadioStation() {
    if (this.currentStationIndex < this.stations.length - 1) {
      this.currentStationIndex++;
    }
  }

  private previousRadioStation() {
    if (this.currentStationIndex > 0) {
      this.currentStationIndex--;
    }
  }

  private turnOnRadio() {
    this.radioOn = true;
  }

  private turnOffRadio() {
    this.radioOn = false;
  }

  private increaseFanSpeed() {
    if (this.fanSpeed < 5) {
      this.fanSpeed++;
    }
  }

  private decreaseFanSpeed() {
    if (this.fanSpeed > 1) {
      this.fanSpeed--;
    }
  }

  private increaseTemperature() {
    this.temperature += 0.5;
  }

  private decreaseTemperature() {
    this.temperature -= 0.5;
  }

  getRandomStation(): RadioStation {
    const randomIndex = Math.floor(Math.random() * this.stations.length);
    return this.stations[randomIndex];
  }

  openModal() {
    this.isModalOpen = true;
  }

  closeModal() {
    this.isModalOpen = false;
  }

  setBrightTheme(): void {
    this.dashboardTheme = 'bright';
  }

  setDimTheme(): void {
    this.dashboardTheme = 'dim';
  }

  setNormalTheme(): void {
    this.dashboardTheme = 'normal';
  }

  setPanelsTheme(): void {
    this.panelTheme = 'dim';
  }

  private translateFunctionalityToAlert(func: GestureFunction): string {
    const map: Record<GestureFunction, string> = {
      odpri_levo_okno_spredaj: 'Odpiranje sprednjega levega okna!',
      zapri_levo_okno_spredaj: 'Zapiranje sprednjega levega okna!',
      odpri_desno_okno_spredaj: 'Odpiranje sprednjega desnega okna!',
      zapri_desno_okno_spredaj: 'Zapiranje sprednjega desnega okna!',
      odpri_levo_okno_zadaj: 'Odpiranje zadnjega levega okna!',
      zapri_levo_okno_zadaj: 'Zapiranje zadnjega levega okna!',
      odpri_desno_okno_zadaj: 'Odpiranje zadnjega desnega okna!',
      zapri_desno_okno_zadaj: 'Zapiranje zadnjega desnega okna!',
      glasnost_gor: 'Povečevanje glasnosti!',
      glasnost_dol: 'Zmanjševanje glasnosti!',
      radio_postaja_prev: 'Preklop na prejšnjo radijsko postajo!',
      radio_postaja_next: 'Preklop na naslednjo radijsko postajo!',
      vklop_radio: 'Vklop radia!',
      izklop_radio: 'Izklop radia!',
      zapiranje_vzvratnega_ogledala: 'Zapiranje vzvratnega ogledala!',
      odpiranje_vzvratnega_ogledala: 'Odpiranje vzvratnega ogledala!',
      premik_kota_vzvratnega_ogledala_v_levo:
        'Premik kota vzvratnega ogledala v levo!',
      premik_kota_vzvratnega_ogledala_v_desno:
        'Premik kota vzvratnega ogledala v desno!',
      premik_kota_vzvratnega_ogledala_navzgor:
        'Premik kota vzvratnega ogledala navzgor!',
      premik_kota_vzvratnega_ogledala_navzdol:
        'Premik kota vzvratnega ogledala navzdol!',
      zviševanje_moči_pihanja_klimatske_naprave:
        'Zviševanje moči pihanja klimatske naprave!',
      zniževanje_moči_pihanja_klimatske_naprave:
        'Zniževanje moči pihanja klimatske naprave!',
      zviševanje_nastavitve_temperature: 'Zviševanje nastavitve temperature!',
      zniževanje_nastavitve_temperature: 'Zniževanje nastavitve temperature!',
    };
    return map[func];
  }

  private showIconInSlot(iconId: string, slot: 'top' | 'bottom') {
    const validIds = slot === 'top' ? this.topSlotIcons : this.bottomSlotIcons;

    validIds.forEach((id) => {
      const el = document.getElementById(id);
      if (el) {
        el.classList.remove('lit');
        el.style.display = 'none';
      }
    });

    if (validIds.includes(iconId)) {
      const target = document.getElementById(iconId);
      if (target) {
        target.style.display = 'block';
        target.classList.add('lit');
      }
    }
  }

  private clearDashboardIcons() {
    [...this.topSlotIcons, ...this.bottomSlotIcons].forEach((id) => {
      const el = document.getElementById(id);
      if (el) {
        el.style.display = 'none';
        el.classList.remove('lit');
      }
    });
  }

  private flashTopIcon(type: 'alert' | 'warning', duration: number) {
    if (type === 'alert') {
      this.showAlertIcon = true;
      setTimeout(() => {
        this.showAlertIcon = false;
      }, duration);
    } else if (type === 'warning') {
      this.showWarningIcon = true;
      setTimeout(() => {
        this.showWarningIcon = false;
      }, duration);
    }
  }
}
