# Upravljanje z gestami pri avtonomni vožnji

## Specifikacije projekta
- trije naučeni modeli umetne inteligence (detekcija gest, okolja, analiza voznika)
- modeli delujejo kot backend za analizo vnosa
- rezultati se prikažejo na ustvarjeni armaturni plošči
- uporabljene tehnologije Python, Eclipse mosquitto, ZeroTierVPN in Angular 

## Namestitev

### Predpogoji
- Python <= 3.12
- Angular v20
- Node.js >= 20
- npm <= 10.9.1

### Nastavitev Repozitorija
- git clone https://github.com/simsss4/Upravljanje_z_gestami.git
- cd Upravljanje_z_gestami

### Nastavitev Frontend-a
- cd Frontend/DashboardUI
- npm install
- zagon: ng serve -o

### Nastavitev Backend-a
- cd Backend
- python -m venv venv
- .\venv\Scripts\activate
- pip install -r requirements.txt
- zagon: python merging_gui.py

### Nastavitev Broker-ja
- ustvariti zerotier omrežje: https://my.zerotier.com/
- povezava na dano omrežje npr. test network id: 41d49af6c2dcb5ce
- menjava na IP ustvarjenega omrežja v Backend skriptah (detect_env_data, drowsiness_detector, gesture_control_gui)
- menjava na IP ustvarjenega omrežja v Frontend (mqtt.serice.ts)

### Zagon MQTT Broker-ja
- zagon terminala kot administrator nato zagon s ustvarjeno konfiugracijo:
- mosquitto -c "C:\Users\tergl\Desktop\School Repos\UI-projekt---Upravljanje-z-gestami\Končen_Projekt\Backend\mosquitto\config\mosquitto.conf" -v
