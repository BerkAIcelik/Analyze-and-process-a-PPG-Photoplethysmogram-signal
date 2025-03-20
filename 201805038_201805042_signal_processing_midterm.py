import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import sys

# Veri setini yükleme
file_path = 'C:/Users/Berkay/Downloads/mimic_perform_af_csv/mimic_perform_af_003_data.csv'  # PPG verisi
data = pd.read_csv(file_path)

# PPG sinyalini seçme
ppg_signal = data['PPG'].to_numpy()

# frekans
fs = 125  # Veri frekansı (Hz)

# Zaman ekseni oluşturma
time = np.arange(len(ppg_signal)) / fs

# Sliding Window parametreleri
winsize = fs * 5  # 5 saniyelik pencere
winhop = fs       # 1 saniyelik kaydırma

from scipy.signal import find_peaks

def detect_pulse_wave_points(signal, fs):

    from scipy.signal import find_peaks

    # 1. PWSP (Systolic Peak): Ana tepe noktaları
    systolic_peaks, _ = find_peaks(signal, distance=fs * 0.47)  # Kalp atım aralıklarına göre arama

    # Türev hesaplamaları
    third_derivative = np.gradient(np.gradient(np.gradient(signal)))

    # Tüm lokal minimumları bulmak için sinyali ters çeviriyoruz
    inverted_signal = -signal
    all_minima, _ = find_peaks(inverted_signal)

    # Başlangıç noktaları (tüm noktalar için)
    points = {'PWB': [], 'PWSP': [], 'Notch': [], 'PWDP': []}

    for i in range(len(systolic_peaks) - 1):
        # Her dalganın başlangıç ve bitişini belirle
        current_peak = systolic_peaks[i]
        next_peak = systolic_peaks[i + 1]

        # 3. PWB (Pulse Wave Begin): Systolic Peak öncesi lokal minimum
        pwb_region = all_minima[all_minima < current_peak]  # Sadece peak öncesine bak
        if len(pwb_region) > 0:
            pwb = pwb_region[-1]  # Son lokal minimum
        else:
            pwb = None
        points['PWB'].append(pwb)

        # 4. PWSP (Systolic Peak): Mevcut tepe noktası
        points['PWSP'].append(current_peak)

        # İlk PWB sonrası lokal minimumu tespit et
        next_pwb_region = all_minima[all_minima > current_peak]  # PWSP sonrası lokal minimumlar
        if len(next_pwb_region) > 0:
            next_pwb = next_pwb_region[0]  # İlk PWB
        else:
            next_pwb = None

        # 5. Dicrotic Notch: 3. türevin sıfır geçiş analizi
        if current_peak is not None and next_pwb is not None:
            time_gap = int(0.05 * fs)  # 0.05 saniye örnekleme aralığına denk geliyor
            refined_region = np.arange(current_peak + time_gap, next_pwb - time_gap)  # [PWSP + 0.05s, PWB - 0.05s)

            if len(refined_region) > 0:  # Bölge geçerliyse
                d3 = third_derivative[refined_region]
                notch_candidates = np.where((d3[:-1] > 0) & (d3[1:] < 0))[0]  # Pozitiften negatife sıfır geçişi

                if len(notch_candidates) > 0:
                    notch_candidate = notch_candidates[0]
                    notch = refined_region[notch_candidate]
                else:
                    notch = None
            else:
                notch = None
        else:
            notch = None
        points['Notch'].append(notch)

        # 6. Diastolic Peak (PWDP): 3. türevin negatiften pozitife geçişi
        if notch is not None and next_pwb is not None:
            refined_region = np.arange(notch + time_gap, next_pwb - time_gap)  # Notch sonrası aralık
            if len(refined_region) > 0:  # Bölge geçerliyse
                d3 = third_derivative[refined_region]
                pwdp_candidates = np.where((d3[:-1] < 0) & (d3[1:] > 0))[0]  # Negatiften pozitife sıfır geçişi

                if len(pwdp_candidates) > 0:
                    pwdp_candidate = pwdp_candidates[0]
                    pwdp = refined_region[pwdp_candidate]
                else:
                    pwdp = None
            else:
                pwdp = None
        else:
            pwdp = None
        points['PWDP'].append(pwdp)

    return points


def calculate_heart_rate(points, fs):

    pwsp_list = points['PWSP']

    # Kalp atımı aralıklarını hesapla (PWSP'ler arası farklar)
    if len(pwsp_list) > 1:
        intervals = np.diff(pwsp_list) / fs  # Saniye cinsinden
        hr = 60 / np.mean(intervals)  # BPM cinsinden Heart Rate

        # Heart Rate Variability (HRV) hesapla
        hrv = np.sqrt(np.mean((intervals - np.mean(intervals))**2))  # RMS
    else:
        hr = None  # Yeterli veri yoksa None
        hrv = None

    return hr, hrv



def calculate_wave_metrics(points, window_time, window_signal):


    ppt_values = []
    pwa_values = []
    pwd_values = []

    pwb_list = points['PWB']
    pwsp_list = points['PWSP']
    pwdp_list = points['PWDP']

    for i in range(len(pwsp_list)):
        pwsp = pwsp_list[i]
        pwb = pwb_list[i] if i < len(pwb_list) else None
        pwdp = pwdp_list[i]

        # PPT (Pulse Transit Time)
        if pwsp is not None and pwdp is not None:
            ppt = window_time[pwdp] - window_time[pwsp]
            ppt_values.append(ppt)
        else:
            ppt_values.append(None)

        # PWA (Pulse Wave Amplitude)
        if pwsp is not None and pwb is not None:
            pwa = abs(window_signal[pwsp] - window_signal[pwb])
            pwa_values.append(pwa)
        else:
            pwa_values.append(None)

    # PWD (Pulse Wave Duration) for all consecutive PWB points
    for i in range(len(pwb_list) - 1):
        if pwb_list[i] is not None and pwb_list[i + 1] is not None:
            pwd = window_time[pwb_list[i + 1]] - window_time[pwb_list[i]]
            pwd_values.append(pwd)
        else:
            pwd_values.append(None)

    return ppt_values, pwa_values, pwd_values

# Başlangıç indeksi
i = -1
# Sliding window fonksiyonuna entegre
def on_press(event):
    global i
    lower = i
    upper = i + winsize

    if event.key == 'right':
        if i == -1:
            i = 0
        else:
            i += winhop
            if i + winsize > len(ppg_signal):
                i = 0
    elif event.key == 'left':
        if i == -1:
            i = len(ppg_signal) - winsize
        else:
            i -= winhop
            if i < 0:
                i = len(ppg_signal) - winsize

    lower = i
    upper = i + winsize
    window_signal = ppg_signal[lower:upper] if i >= 0 else None

    ax1.cla()
    ax1.plot(time, ppg_signal, 'g', label="Ham Sinyal")
    if i >= 0:
        ax1.plot(time[lower:upper], ppg_signal[lower:upper], 'r', label="Sliding Window")
    ax1.legend()
    ax1.grid()
    ax1.set_title("Ham Sinyal")

    ax2.cla()
    if i >= 0:
        window_time = time[lower:upper]

        points = detect_pulse_wave_points(window_signal, fs)
        ppt_values, pwa_values, pwd_values = calculate_wave_metrics(points, window_time, window_signal)
        hr, hrv = calculate_heart_rate(points, fs)
        ax2.plot(window_time, window_signal, 'r', label="Pencere Sinyali")

        for j, (pwb, pwsp, notch, pwdp) in enumerate(zip(points['PWB'], points['PWSP'], points['Notch'], points['PWDP'])):
            if pwb is not None:
                ax2.plot(window_time[pwb], window_signal[pwb], 'go', label="PWB" if j == 0 else "")
            if pwsp is not None:
                ax2.plot(window_time[pwsp], window_signal[pwsp], 'ro', label="PWSP" if j == 0 else "")
            if notch is not None:
                ax2.plot(window_time[notch], window_signal[notch], 'mo', label="Notch" if j == 0 else "")
                ax2.text(window_time[notch], window_signal[notch], 'Notch', color='magenta', fontsize=8)
            if pwdp is not None:
                ax2.plot(window_time[pwdp], window_signal[pwdp], 'bo', label="Diastolic Peak" if j == 0 else "")
                ax2.text(window_time[pwdp], window_signal[pwdp], 'PWDP', color='blue', fontsize=8)

            # Systolic Phase ve Diastolic Phase görselleştirme
            if pwb is not None and pwsp is not None:
                # Systolic Phase: PWB ile PWSP arasındaki alan
                systolic_region_x = window_time[pwb:pwsp + 1]
                systolic_region_y = window_signal[pwb:pwsp + 1]
                ax2.fill_between(systolic_region_x, systolic_region_y, min(window_signal) - 0.1, color='yellow', alpha=0.3, label="Systolic Phase" if j == 0 else "")

            if pwsp is not None and j < len(points['PWB']) - 1 and points['PWB'][j + 1] is not None:
                # Diastolic Phase: PWSP ile sonraki PWB arasındaki alan
                diastolic_region_x = window_time[pwsp:points['PWB'][j + 1] + 1]
                diastolic_region_y = window_signal[pwsp:points['PWB'][j + 1] + 1]
                ax2.fill_between(diastolic_region_x, diastolic_region_y, min(window_signal) - 0.1, color='cyan', alpha=0.3, label="Diastolic Phase" if j == 0 else "")

            # PPT Görselleştirme
            if pwsp is not None and pwdp is not None and ppt_values[j] is not None:
                y_baseline = max(window_signal)  - 0.01
                ax2.annotate('', xy=(window_time[pwsp], y_baseline),
                             xytext=(window_time[pwdp], y_baseline),
                             arrowprops=dict(arrowstyle='<->', color='red'))
                ax2.text((window_time[pwsp] + window_time[pwdp]) / 2, y_baseline - 0.02,
                         f"PPT: {ppt_values[j]:.2f} s", color='red', fontsize=8, ha='center')

            # PWA Görselleştirme
            if pwsp is not None and pwb is not None and pwa_values[j] is not None:
                ax2.annotate('', xy=(window_time[pwsp] + 0.05, window_signal[pwsp]),
                             xytext=(window_time[pwsp] + 0.05, window_signal[pwb]),
                             arrowprops=dict(arrowstyle='<->', color='blue'))
                ax2.text(window_time[pwsp] + 0.07, (window_signal[pwsp] + window_signal[pwb]) / 2,
                         f"PWA: {pwa_values[j]:.2f}", color='blue', fontsize=8, ha='center')

            # PWD Görselleştirme (Yatayda sabit)
            if j < len(points['PWB']) - 1 and pwb is not None and points['PWB'][j + 1] is not None and pwd_values[j] is not None:
                y_baseline = min(window_signal) - 0.1
                ax2.annotate('', xy=(window_time[pwb], y_baseline),
                             xytext=(window_time[points['PWB'][j + 1]], y_baseline),
                             arrowprops=dict(arrowstyle='<->', color='green'))
                ax2.text((window_time[pwb] + window_time[points['PWB'][j + 1]]) / 2, y_baseline - 0.02,
                         f"PWD: {pwd_values[j]:.2f} s", color='green', fontsize=8, ha='center')

    ax2.legend()
    ax2.grid()
    ax2.set_xlabel("Zaman (s)")
    ax2.set_title(f"Sliding Window Analizi (Heart Rate: {hr:.2f} BPM, HRV: {hrv:.2f} s)" if hr is not None else "Sliding Window Analizi (Heart Rate: N/A, HRV: N/A)")

    fig.canvas.draw()


# Grafik düzeni 2sinin yaklaşık aynı olması için
#fig = plt.figure(figsize=(12, 8))  
#ax1 = fig.add_subplot(211)  # Üstteki ham sinyal grafiği
#ax2 = fig.add_subplot(212)  # Alttaki sliding window analizi grafiği

# Başlangıç görselleştirme
#ax1.plot(time, ppg_signal, 'g', label="Ham Sinyal")
#ax1.grid()
#ax1.set_title("Ham Sinyal", fontsize=14)  
#ax2.grid()
#ax2.set_title("Başlangıç Ekranı (Heart Rate: N/A)", fontsize=14)  
#fig.canvas.mpl_connect('key_press_event', on_press)

# İki grafik arasındaki boşluğu artırma
#plt.subplots_adjust(hspace=0.4)  

#plt.tight_layout()
#plt.show()



# Grafik düzeni üstteki daha küçük alttaki daha büyük olsun diye
fig = plt.figure(figsize=(12, 8))  # Daha geniş grafik boyutu

# Alt grafiklerin oranlarını belirleyerek eksenleri tanımlıyoruz
ax1 = fig.add_axes([0.1, 0.7, 0.8, 0.2])  
ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.55])  

# Başlangıç görselleştirme
ax1.plot(time, ppg_signal, 'g', label="Ham Sinyal")
ax1.grid()
ax1.set_title("Ham Sinyal", fontsize=14)  # Ham sinyal başlığı
ax2.grid()
ax2.set_title("Başlangıç Ekranı (Heart Rate: N/A)", fontsize=14)  # Sliding Window başlığı
fig.canvas.mpl_connect('key_press_event', on_press)

plt.show()













