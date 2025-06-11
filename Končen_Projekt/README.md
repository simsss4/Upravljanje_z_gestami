MODEL 1 - Gestna kontrola (upravljanje radia, klime, oknov in vzratnih ogledal)

Funkcije:

----------------------------------------------------------------------------------------------------------

MODEL 2 - Model za analizo utrujenosti voznika

vhod: slika proti vozniku

izhodi/funkcionalnost:

voznik je utrujen ->    Display "Zaznana je bila utrujenost, prosimo vzamite si odmor od vožnje!" alert

voznik ni utrujen ->    Ni izpisa opozorila

----------------------------------------------------------------------------------------------------------

MODEL 3 - Model za analizo okolje-specifičnih podatkov

vhod: slika navzven proti vetrobranskemu steklu

izhodi/funkcionalnost:

day ->      Bright dashboard Theme
            Sunglasses suggestion

night ->    Auto-enable headlights
            Dim dashboard

clear ->    Normal dashboard brightness
            Display "Normal weather, drive safe!" alert

foggy ->    Auto-activate fog lights
            Display "Low Visibility" warning
            Suggest keeping distance

rainy ->    Auto-enable wipers (animated visualization)
            Display "Wet Road" warning
            Suggest slower speed (e.g., "Recommended: 60 km/h")


----------------------------------------------------------------------------------------------------------

RUN BACKEND:
cd Backend
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
python 'merging_gui'

ZeroTier VPN
test network id: 41d49af6c2dcb5ce
host IP: 10.147.20.65
