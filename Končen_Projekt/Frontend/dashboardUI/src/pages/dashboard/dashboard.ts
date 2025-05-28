import { CommonModule } from '@angular/common';
import { Component } from '@angular/core';
import { RouterLink } from '@angular/router';

type WeatherType = 'day' | 'night' | 'clear' | 'rainy' | 'foggy';

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

  private mockWeatherData: WeatherData[] = [
    { type: 'day', imageUrl: 'MockWeather/mock_day.jpg' },
    { type: 'night', imageUrl: 'MockWeather/mock_night.jpg' },
    { type: 'clear', imageUrl: 'MockWeather/mock_clear.jpg' },
    { type: 'rainy', imageUrl: 'MockWeather/mock_foggy.png' },
    { type: 'foggy', imageUrl: 'MockWeather/mock_rainy.jpg' },
  ];

  loadDemoImage(): void {
    const random = Math.floor(Math.random() * this.mockWeatherData.length);
    this.weather = this.mockWeatherData[random];
  }

  openModal() {
    this.isModalOpen = true;
  }

  closeModal() {
    this.isModalOpen = false;
  }
}
