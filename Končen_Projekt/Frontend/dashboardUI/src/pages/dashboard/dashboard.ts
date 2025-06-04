import { CommonModule } from '@angular/common';
import { Component } from '@angular/core';
import { RouterLink } from '@angular/router';

type WeatherType = 'dan' | 'noč' | 'jasno' | 'deževno' | 'megleno';
type GestureType = 'leva_roka_gor_odprto' | 'leva_roka_dol_odprto' | 'desna_roka_gor_odprto' | 'desna_roka_dol_odprto' |
                    'leva_roka_gor_zaprto' | 'leva_roka_dol_zaprto' | 'desna_roka_gor_zaprto' | 'desna_roka_dol_zaprto';

type GestureFunction = 'zapri_levo_okno_spredaj' | 'odpri_levo_okno_spredaj' | 'zapri_desno_okno_spredaj' | 'odpri_desno_okno_spredaj' |
                    'zapri_levo_okno_zadaj' | 'odpri_levo_okno_zadaj' | 'zapri_desno_okno_zadaj' | 'odpri_desno_okno_zadaj';

interface WeatherData {
  type: WeatherType;
  imageUrl: string;
}
interface GestureData {
  type: GestureType;
  functionality: GestureFunction;
}

@Component({
  selector: 'app-dashboard',
  imports: [RouterLink, CommonModule],
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
  { type: 'desna_roka_gor_odprto', functionality: 'zapri_desno_okno_spredaj' },
  { type: 'desna_roka_dol_odprto', functionality: 'odpri_desno_okno_spredaj' },
  { type: 'leva_roka_gor_zaprto', functionality: 'zapri_levo_okno_zadaj' },
  { type: 'leva_roka_dol_zaprto', functionality: 'odpri_levo_okno_zadaj' },
  { type: 'desna_roka_gor_zaprto', functionality: 'zapri_desno_okno_zadaj' },
  { type: 'desna_roka_dol_zaprto', functionality: 'odpri_desno_okno_zadaj' },
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

  loadDemoImage(): void {
    const random = Math.floor(Math.random() * this.mockWeatherData.length);
    this.weather = this.mockWeatherData[random];

    const weatherType = this.weather.type;
    const weatherData = this.weatherResponses[weatherType];

    this.alerts = weatherData.alerts;
    this.warnings = weatherData.warnings;

    if (weatherType === 'noč') {
      this.setDimTheme();
    } else if (weatherType === 'dan') {
      this.setBrightTheme();
    } else {
      this.setNormalTheme();
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
}
