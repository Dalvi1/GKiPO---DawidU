import urllib.request
import numpy as np
import cv2
import matplotlib.pyplot as plt

# --- 1. FUNKCJA WCZYTUJĄCA ZDJĘCIE ZE ZDALNEGO URL (Krok 1) ---

def wczytaj_zdjecie_ze_zdalnego_url(url):
    """Pobiera i dekoduje obraz z URL za pomocą urllib i cv2."""
    try:
        print(f"Pobieranie zdjęcia z: {url}...")
        # Dodanie nagłówka User-Agent, aby uniknąć blokady
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        
        with urllib.request.urlopen(req) as response:
            dane_binarne = response.read()

        # Konwersja danych binarnych na tablicę NumPy
        tablica_bajtów = np.asarray(bytearray(dane_binarne), dtype=np.uint8)

        # Dekodowanie obrazu (cv2.IMREAD_COLOR dla 3 kanałów)
        img = cv2.imdecode(tablica_bajtów, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Błąd dekodowania obrazu. Sprawdź, czy URL jest poprawny.")
        
        print("Zdjęcie wczytane pomyślnie.")
        return img

    except Exception as e:
        print(f"Błąd wczytywania zdjęcia z URL: {e}")
        return None


# --- 2. FUNKCJA WIZUALIZUJĄCA HISTOGRAMY (Krok 2) ---

def wyswietl_histogramy(img):
    """Wyświetla histogramy w skali szarości i dla kanałów BGR."""
    
    plt.figure(figsize=(15, 6))

    # 1. Histogram Skali Szarości
    szary = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist_szary = cv2.calcHist([szary], [0], None, [256], [0, 256])

    plt.subplot(1, 2, 1)
    plt.plot(hist_szary, color='gray')
    plt.title('Histogram Skali Szarości')
    plt.xlabel('Intensywność (0-255)')
    plt.ylabel('Liczba pikseli')
    
    # 2. Histogramy Kanałów Koloru
    kolory = ('b', 'g', 'r')
    plt.subplot(1, 2, 2)
    
    for i, kolor in enumerate(kolory):
        hist = cv2.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(hist, color=kolor)
        
    plt.title('Histogramy Kanałów Koloru (B, G, R)')
    plt.xlabel('Intensywność (0-255)')
    plt.ylabel('Liczba pikseli')
    plt.legend(['Niebieski', 'Zielony', 'Czerwony'])

    plt.tight_layout()
    plt.show() 



# --- 3. FUNKCJA OCENIAJĄCA JAKOŚĆ NA PODSTAWIE HISTOGRAMU (Krok 3) ---

def oszacuj_jakosc_przez_histogram(hist_dane, calkowita_liczba_pikseli):
    """Oszacowuje kontrast i ekspozycję na podstawie danych histogramu."""
    
    # Analiza Krańców (Clipping)
    procent_czarnych = (hist_dane[0][0] / calkowita_liczba_pikseli) * 100
    procent_bialych = (hist_dane[255][0] / calkowita_liczba_pikseli) * 100
    
    # Analiza Kontrastu (Pokrycie Tonalne)
    pokrycie_tonalne = (np.sum(hist_dane > 0) / 256) * 100
    
    # Analiza Ekspozycji (Średnia Jasność)
    poziomy_jasnosci = np.arange(256)
    srednia_intensywnosc = np.sum(hist_dane * poziomy_jasnosci) / calkowita_liczba_pikseli
    
    # --- PROSTA OCENA ---
    problemy = []
    
    if procent_czarnych > 0.5:
        problemy.append(f"Utrata detali w cieniach ({procent_czarnych:.2f}% czarnych).")
    if procent_bialych > 0.5:
        problemy.append(f"Utrata detali w światłach ({procent_bialych:.2f}% białych).")
    if pokrycie_tonalne < 60:
        problemy.append(f"Niski kontrast ({pokrycie_tonalne:.1f}% pokrycia).")
    if abs(srednia_intensywnosc - 128) > 30:
        problemy.append(f"Ekspozycja przesunięta (Średnia: {srednia_intensywnosc:.1f}).")
        
    ocena = "Bardzo dobra jakość tonalna i ekspozycja." if not problemy else "Wykryto problemy tonalne/ekspozycyjne."

    print("\n--- 📊 Raport Jakości Zdjęcia ---")
    print(f"**Ocena Końcowa:** {ocena}")
    if problemy:
        print("\nWykryte Problemy:")
        for p in problemy:
            print(f"- {p}")
    
    print("\nSzczegóły Numeryczne:")
    print(f"  Średnia Intensywność: {srednia_intensywnosc:.1f}")
    print(f"  Pokrycie Tonalne: {pokrycie_tonalne:.1f}%")
    print(f"  Przycięcie (Cienie/Światła): {procent_czarnych:.2f}% / {procent_bialych:.2f}%")
    print("-----------------------------------")


# --- 4. WYKONANIE ANALIZY GŁÓWNEJ ---

# Adres URL do przetestowania (Nissan Skyline - URL wybrany przez pana)
URL_ZDJECIA = "https://upload.wikimedia.org/wikipedia/commons/thumb/a/ac/Nissan_Skyline_R34_tuned.jpg/640px-Nissan_Skyline_R34_tuned.jpg"

print("\n--- ROZPOCZĘCIE ANALIZY ---")
obraz = wczytaj_zdjecie_ze_zdalnego_url(URL_ZDJECIA)

if obraz is not None:
    
    # Krok 2: Wyświetlenie Histogramów
    wyswietl_histogramy(obraz)
    
    # Przygotowanie danych do oceny jakości (histogram w skali szarości)
    szary = cv2.cvtColor(obraz, cv2.COLOR_BGR2GRAY)
    hist_szary_dane = cv2.calcHist([szary], [0], None, [256], [0, 256])
    
    # Krok 3: Oszacowanie Jakości
    liczba_pikseli = obraz.shape[0] * obraz.shape[1]
    oszacuj_jakosc_przez_histogram(hist_szary_dane, liczba_pikseli)
