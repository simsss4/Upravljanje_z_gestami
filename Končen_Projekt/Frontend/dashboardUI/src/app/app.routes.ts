import { Routes } from '@angular/router';
import { Dashboard } from '../pages/dashboard/dashboard';
import { DiagnostikaSistema } from '../pages/diagnostika-sistema/diagnostika-sistema';
import { StatistikaModelov } from '../pages/statistika-modelov/statistika-modelov';

export const routes: Routes = [
    {
        path: '',
        component: Dashboard,
        title: 'Dashboard'
    },
    {
        path: 'diagnostika',
        component: DiagnostikaSistema,
        title: 'Diagnostika sistema'
    },
    {
        path: 'statistika',
        component: StatistikaModelov,
        title: 'Statistika modelov'
    },
    {
        path: '**',
        redirectTo: ''
    }
];
