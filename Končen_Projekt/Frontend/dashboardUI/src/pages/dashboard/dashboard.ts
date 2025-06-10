import { CommonModule } from '@angular/common';
import { Component } from '@angular/core';
import { RouterLink } from '@angular/router';
import { MqttService } from '../../app/Services/mqtt.service';

type WeatherType = 'dan' | 'noč' | 'jasno' | 'deževno' | 'megleno';
type GestureType =
  | 'leva_roka_gor_odprto'
  | 'leva_roka_dol_odprto'
  | 'desna_roka_gor_odprto'
  | 'desna_roka_dol_odprto'
  | 'leva_roka_gor_zaprto'
  | 'leva_roka_dol_zaprto'
  | 'desna_roka_gor_zaprto'
  | 'desna_roka_dol_zaprto'
  | 'dvig_roke'
  | 'spust_roke'
  | 'horizontalno_desno'
  | 'horizontalno_levo'
  | 'stisnjena_pest'
  | 'gib_prstov_levo'
  | 'gib_prstov_desno';

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
  | 'vklop_radio';
// | 'izklop_radio'
// | 'zapiranje vzvratnega ogledala'
// | 'odpiranje vzvratnega ogledala'
// | 'premik kota vzvratnega ogledala v levo'
// | 'premik kota vzvratnega ogledala v desno'
// | 'premik kota vzvratnega ogledala navzgor'
// | 'premik kota vzvratnega ogledala v navzdol'
// | 'zviševanje moči pihanja klimatske naprave'
// | 'zniževanje moči pihanja klimatske naprave'
// | 'zviševanje nastavitve temperature'
// | 'zniževanje nastavitve temperature';

interface WeatherData {
  type: WeatherType;
  imageUrl: string;
}
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
export class Dashboard {
  weather?: WeatherData;
  gesture?: GestureData;
  isModalOpen = false;
  alerts: string[] = [];
  warnings: string[] = [];
  dashboardTheme = 'normal';
  panelTheme = 'normal';

  showAlertIcon = false;
  showWarningIcon = false;

  private topSlotIcons: string[] = ['alert', 'warning'];
  private bottomSlotIcons: string[] = ['dnevne', 'kratke', 'meglenkle'];

  private mockWeatherData: WeatherData[] = [
    { type: 'dan', imageUrl: 'MockWeather/mock_day.jpg' },
    { type: 'noč', imageUrl: 'MockWeather/mock_night.jpg' },
    { type: 'jasno', imageUrl: 'MockWeather/mock_clear.jpg' },
    { type: 'deževno', imageUrl: 'MockWeather/mock_rainy.jpg' },
    { type: 'megleno', imageUrl: 'MockWeather/mock_foggy.png' },
  ];

  private isDriverExhausted: boolean = false;

  private gestureDataMap: GestureData[] = [
    { type: 'leva_roka_gor_odprto', functionality: 'zapri_levo_okno_spredaj' },
    { type: 'leva_roka_dol_odprto', functionality: 'odpri_levo_okno_spredaj' },
    {
      type: 'desna_roka_gor_odprto',
      functionality: 'zapri_desno_okno_spredaj',
    },
    {
      type: 'desna_roka_dol_odprto',
      functionality: 'odpri_desno_okno_spredaj',
    },
    { type: 'leva_roka_gor_zaprto', functionality: 'zapri_levo_okno_zadaj' },
    { type: 'leva_roka_dol_zaprto', functionality: 'odpri_levo_okno_zadaj' },
    { type: 'desna_roka_gor_zaprto', functionality: 'zapri_desno_okno_zadaj' },
    { type: 'desna_roka_dol_zaprto', functionality: 'odpri_desno_okno_zadaj' },
    { type: 'dvig_roke', functionality: 'glasnost_gor' },
    { type: 'spust_roke', functionality: 'glasnost_dol' },
    { type: 'horizontalno_desno', functionality: 'radio_postaja_next' },
    { type: 'horizontalno_levo', functionality: 'radio_postaja_prev' },
    { type: 'stisnjena_pest', functionality: 'vklop_radio' },
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

  getRandomStation(): RadioStation {
    const randomIndex = Math.floor(Math.random() * this.stations.length);
    return this.stations[randomIndex];
  }

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

  loadDemoImage(): void {
    const random = Math.floor(Math.random() * this.mockWeatherData.length);
    this.weather = this.mockWeatherData[random];

    const weatherType = this.weather.type;
    const weatherData = this.weatherResponses[weatherType];

    this.alerts = weatherData.alerts;
    this.warnings = weatherData.warnings;
    this.panelTheme = '';

    if (this.alerts.length > 0) {
      this.flashTopIcon('alert', 5);
    }
    if (this.warnings.length > 0) {
      this.flashTopIcon('warning', 3);
    }

    if (weatherType === 'noč') {
      this.setDimTheme();
      this.panelTheme = 'dim';
      this.showIconInSlot('kratke', 'bottom');
    } else if (weatherType === 'dan') {
      this.setBrightTheme();
      this.showIconInSlot('dnevne', 'bottom');
    } else if (weatherType === 'megleno') {
      this.setNormalTheme();
      this.showIconInSlot('meglenkle', 'bottom');
    } else {
      this.setNormalTheme();
      this.clearDashboardIcons();
    }
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
  public triggerRandomGesture(): void {
    this.clearDashboardIcons();
    this.warnings.length = 0;
    const randomIndex = Math.floor(Math.random() * this.gestureDataMap.length);
    const selectedGesture = this.gestureDataMap[randomIndex];
    const alertMessage = this.translateFunctionalityToAlert(
      selectedGesture.functionality
    );

    console.log(
      'Selected Gesture:\n' +
        selectedGesture.type +
        '\nFunction:\n' +
        selectedGesture.functionality
    );

    this.alerts.length = 0;
    this.alerts.push(alertMessage);
    this.flashTopIcon('alert', 5000);
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
    };

    return map[func];
  }

  // private functionalityHandlers: Record<GestureFunction, () => void> = {
  //   glasnost_gor: () => this.povecajGlasnost(),
  //   glasnost_dol: () => this.zmanjsajGlasnost(),
  //   radio_postaja_prev: () => this.prejsnjaPostaja(),
  //   radio_postaja_next: () => this.naslednjaPostaja(),
  //   vklop_radio: () => this.vklopiRadio(),
  //   nastavi_kot_levo_ogledalo: () => this.nastaviKotLevoOgledalo(),
  //   nastavi_kot_desno_ogledalo: () => this.nastaviKotDesnoOgledalo(),

  //   odpri_levo_okno_spredaj: () => {},
  //   zapri_levo_okno_spredaj: () => {},
  //   odpri_desno_okno_spredaj: () => {},
  //   zapri_desno_okno_spredaj: () => {},
  //   odpri_levo_okno_zadaj: () => {},
  //   zapri_levo_okno_zadaj: () => {},
  //   odpri_desno_okno_zadaj: () => {},
  //   zapri_desno_okno_zadaj: () => {},
  // };

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

  constructor(private mqttService: MqttService) {
    this.mqttService.connect(); // vzpostavi povezavo takoj
  }
  public triggerUtrujenostDemo(): void {
    const mockPayload = {
      model: 'model2',
      status: 'drowsy',
      message: 'Utrujenost zaznana – priporočamo odmor!',
      timestamp: new Date().toISOString(),
    };

    // Prikaži sporočilo na zaslonu
    this.alerts = ['Zaznana utrujenost voznika! Priporočamo odmor.'];
    this.flashTopIcon('alert', 5000);

    this.mqttService.publish('model/utrujenost', mockPayload);
  }
}
