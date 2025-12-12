import urllib.request
import numpy as np
import cv2
import matplotlib.pyplot as plt

# WCZYTANIE ZDJĘCIA ZE ZDALNEGO URL

def load_image_from_remote_url(url):
    """Pobiera i dekoduje obraz z URL za pomocą urllib i cv2."""
    try:
        print(f"Pobieranie zdjęcia z: {url}...")
        # Nagłówek User-Agent, aby uniknąć blokady
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})

        with urllib.request.urlopen(req) as response:
            binary_data = response.read()

        # Konwersja danych binarnych na tablicę NumPy
        byte_array = np.asarray(bytearray(binary_data), dtype=np.uint8)
        # Dekodowanie obrazu (cv2.IMREAD_COLOR dla 3 kanałów)
        img = cv2.imdecode(byte_array, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Błąd dekodowania obrazu. Sprawdź, czy URL jest poprawny.")

        print("Zdjęcie wczytane pomyślnie.")
        return img

    except Exception as e:
        print(f"Błąd wczytywania zdjęcia z URL: {e}")
        return None


# WIZUALIZACJA HISTOGRAMU

def display_image_histogram(img):
    """Wyświetlanie histogramu oraz zdjęcia"""

    plt.figure(figsize=(16, 6), facecolor='#2c2c2c')


    # przekonwertowanie obraz z BGR na RGB.
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # lewa pozycja
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title('Wczytane Zdjęcie', color='white')
    plt.axis('off') # Ukrcie osi

    # --- PODWYKRES 2: WYŚWIETLANIE HISTOGRAMÓW ---

    # prawa pozycja
    plt.subplot(1, 2, 2)
    ax = plt.gca()
    ax.set_facecolor('#545454')



    color_bgr = ('b', 'g', 'r')
    legend_bgr = ('Niebieski', 'Zielony', 'Czerwony')
    brightness_levels = np.arange(256)

    # Kanały kolorów bgr
    for i, color in enumerate(color_bgr):
        hist = cv2.calcHist([img], [i], None, [256], [0, 256])
        plt.bar(brightness_levels, hist.flatten(), color=color, alpha=0.7,
                width=1.0, label=legend_bgr[i])

    # Skala szarości
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist_grayscale = cv2.calcHist([grayscale], [0], None, [256], [0, 256])

    plt.bar(brightness_levels, hist_grayscale.flatten(), color='gray', alpha=1,
            width=1.0, label='Skala Szarości (Ogólna Jasność)')

    # Ustawienia końcowe
    plt.title('Histogram wczytanego zdjęcia z URL', color='white')
    plt.xlabel('Intensywność (0-255)', color='white')
    plt.ylabel('Liczba pikseli', color='white')
    plt.xlim([0, 256])
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tick_params(colors='white')
    plt.tight_layout()
    plt.show()


# OCENA JAKOŚCI NA PODSTAWIE HISTOGRAMU

def evaluate_quality_by_histogram(hist_data, total_pixel_count):
    """Ocena kontrastu, ekspozycji i przycięć na podstawie danych histogramu."""

    # spłaszczanie tablicy z (256, 1) na (256)
    hist_flat = hist_data.flatten()

    # Obliczenia wskaźników
    percent_black = (hist_flat[0] / total_pixel_count) * 100
    percent_white = (hist_flat[255] / total_pixel_count) * 100
    tonal_coverage = (np.sum(hist_data > 0) / 256) * 100
    brightness_levels = np.arange(256)
    # Obliczanie wskaźnik ogólnej jasności zdjęcia
    average_intensity = np.sum(hist_flat * brightness_levels) / total_pixel_count

    
    # OCENA ZDJĘCIA NA PODSATAWIE HISTOGRAMU
    problem = []

    # KRYTERIUM 1: EKSPOZYCJA
    # Oczekiwany zakres neutralnej ekspozycji to 88.0 do 168.0
    OCZEKIWANY_ZAKRES = "88.0 do 168.0"

    if abs(average_intensity - 128) > 40:
        if average_intensity > 168: 
            problem.append(
                f"Zdjęcie jest PRZEŚWIETLONE (za jasne). "
                f"Średnia wynosi: {average_intensity:.1f}, a oczekiwany zakres to {OCZEKIWANY_ZAKRES}."
            )
            
        # Sprawdzanie niedoświetlenia (za ciemno)
        elif average_intensity < 88: 
            problem.append(
                f"Zdjęcie jest NIEDOŚWIETLONE (za ciemne). "
                f"Średnia wynosi: {average_intensity:.1f}, a oczekiwany zakres to {OCZEKIWANY_ZAKRES}."
            )
            
    # KRYTERIUM 2: PRZYCIĘCIE DETALI (Cienie/Światła)
    if percent_black > 0.5:
        problem.append(f"Utrata detali w cieniach ({percent_black:.2f}% czarnych).")
    if percent_white > 0.5:
        problem.append(f"Utrata detali w światłach ({percent_white:.2f}% białych).")
        
    # KRYTERIUM 3: KONTRAST
    if tonal_coverage < 60:
        problem.append(f"Niski kontrast ({tonal_coverage:.1f}% pokrycia).")

    rating = "Bardzo dobra jakość tonalna i ekspozycja." if not problem else "Wykryto potencjalne problemy tonalne/ekspozycyjne."

    # OUTPUT W POSTACI RAPORTU
    print("\n--- Raport Jakości Zdjęcia ---")
    print(f"Ocena Końcowa: {rating}")
    if problem:
        print("\nWykryte problemy:")
        for p in problem:
            print(f"- {p}")

    print("\nSzczegóły:")
    print(f"  Średnia Jasność Zdjęcia (Intensywność): {average_intensity:.1f}")
    print(f"  Pokrycie Tonalne: {tonal_coverage:.1f}%")
    print(f"  Przycięcie (Cienie/Światła): {percent_black:.2f}% / {percent_white:.2f}%")
    print("-----------------------------------")


# PROCES ANLIZY HISTOGRAMU

# Adres URL do przetestowania
URL_ZDJECIA = "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8e/Squirrel_on_Linch_Down_-_geograph.org.uk_-_1937374.jpg/640px-Squirrel_on_Linch_Down_-_geograph.org.uk_-_1937374.jpg"

print("\n--- ROZPOCZĘCIE ANALIZY ---")
image = load_image_from_remote_url(URL_ZDJECIA)

if image is not None:

    display_image_histogram(image) # Wywołujemy funkcję histogramu

    # Przygotowanie danych do oceny jakości (histogram w skali szarości)
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist_grayscale_data = cv2.calcHist([grayscale], [0], None, [256], [0, 256])

    # Oszacowanie Jakości
    pixel_count = image.shape[0] * image.shape[1]
    evaluate_quality_by_histogram(hist_grayscale_data, pixel_count)
