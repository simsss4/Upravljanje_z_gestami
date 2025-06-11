import { ComponentFixture, TestBed } from '@angular/core/testing';

import { DiagnostikaSistema } from './diagnostika-sistema';

describe('DiagnostikaSistema', () => {
  let component: DiagnostikaSistema;
  let fixture: ComponentFixture<DiagnostikaSistema>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [DiagnostikaSistema]
    })
    .compileComponents();

    fixture = TestBed.createComponent(DiagnostikaSistema);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
