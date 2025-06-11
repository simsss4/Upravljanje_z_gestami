CONFIGURE FRONTEND
cd Frontend/DashboardUI
npm install

zagon: ng serve -o

CONFIGURE BACKEND:
cd Backend
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
python 'merging_gui'

ZeroTier VPN
test network id: 41d49af6c2dcb5ce
host IP: 10.147.20.65

RUN mosquitto
 mosquitto -c "C:\Users\tergl\Desktop\School Repos\UI-projekt---Upravljanje-z-gestami\Končen_Projekt\Backend\mosquitto\config\mosquitto.conf" -v
