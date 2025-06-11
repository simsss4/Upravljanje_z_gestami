import { ComponentFixture, TestBed } from '@angular/core/testing';

import { StatistikaModelov } from './statistika-modelov';

describe('StatistikaModelov', () => {
  let component: StatistikaModelov;
  let fixture: ComponentFixture<StatistikaModelov>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [StatistikaModelov]
    })
    .compileComponents();

    fixture = TestBed.createComponent(StatistikaModelov);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
