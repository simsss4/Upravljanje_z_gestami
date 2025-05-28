import { CommonModule } from '@angular/common';
import { Component } from '@angular/core';
import { RouterLink } from '@angular/router';

type WeatherType = 'dan' | 'noč' | 'jasno' | 'deževno' | 'megleno';

interface WeatherData {
  type: WeatherType;
  imageUrl: string;
}

@Component({
  selector: 'app-dashboard',
  imports: [RouterLink, CommonModule],
  templateUrl: './dashboard.html',
  styleUrl: './dashboard.css',
})
export class Dashboard {
  weather?: WeatherData;
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
