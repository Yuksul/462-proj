import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import line_aa
from time import time
import cv2
from pathlib import Path
# preprocess_stringart.py dosyasının yanımızda olduğundan eminiz
from preprocess_stringart import StringArtPreprocess

# --- AYARLAR ---
INPUT_IMAGE = str((Path(__file__).parent / "testimage.jpeg").resolve())
NUM_NAILS = 240                # Kullanıcı isteği: 240 çivi
MAX_LINES = 5500               # Daha fazla ip
LINE_WEIGHT = 0.045            # İnce ip, sonda daha da inceliyor
OUTPUT_SIZE = 800              
MIN_DISTANCE = 18              # Daha az çiviyle mesafeyi biraz düşürdük
BAN_DEPTH = 20                 # Topaklanmayı önlemek için orta seviye taboo

# --- Yardımcı Fonksiyonlar ---
def cache_lines(nails):
    """
    Hız için tüm çivi kombinasyonlarının piksel koordinatlarını hafızaya alır.
    """
    cache = {}
    print("Önbellek oluşturuluyor (Bu işlem bir kez yapılır)...")
    for i in range(len(nails)):
        for j in range(i + 1, len(nails)):
            rr, cc, val = line_aa(nails[i][0], nails[i][1], nails[j][0], nails[j][1])
            # Veriyi sıkıştırıp saklayalım
            cache[(i, j)] = (rr, cc, val)
    return cache

def get_cached_line(i, j, line_cache, nails):
    if i == j:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=float)
    # Her zaman küçük indeks önce gelir (kombinasyon standardı)
    a, b = (i, j) if i < j else (j, i)
    
    # Eğer önbellekte yoksa (çok nadir), anlık hesapla
    if (a, b) not in line_cache:
        rr, cc, val = line_aa(nails[a][0], nails[a][1], nails[b][0], nails[b][1])
        return rr, cc, val
    
    return line_cache[(a, b)]

def create_circle_nail_positions(height, width, num_nails):
    centre = (height // 2, width // 2)
    radius = min(height, width) // 2 - 2
    angles = np.linspace(0, 2 * np.pi, num_nails, endpoint=False)
    nails = []
    for angle in angles:
        r = int(centre[0] + radius * np.sin(angle))
        c = int(centre[1] + radius * np.cos(angle))
        nails.append((r, c))
    return nails

def init_black_canvas(height, width):
    return np.zeros((height, width), dtype=np.float32)

def get_aa_line(from_idx, to_idx, str_strength, picture, nails, line_cache):
    rr, cc, val = get_cached_line(from_idx, to_idx, line_cache, nails)
    if rr.size == 0:
        return picture, rr, cc
    line = picture[rr, cc] + str_strength * val
    line = np.clip(line, a_min=0, a_max=1)
    return line, rr, cc

def save_instructions(pull_order, filename="instructions.txt"):
    """
    Tabloyu yapmak için gereken çivi sırasını dosyaya kaydeder.
    """
    print(f"Talimatlar {filename} dosyasına kaydediliyor...")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"String Art Instructions / Talimatlar\n")
        f.write(f"Gorsel: {INPUT_IMAGE}\n")
        f.write(f"Civi Sayisi: {NUM_NAILS}, Toplam Ip: {len(pull_order)-1}\n")
        f.write("-" * 40 + "\n")
        
        # Okunabilir format: 15'li gruplar halinde yaz
        for i in range(0, len(pull_order), 15):
            chunk = pull_order[i:i+15]
            line_str = " -> ".join(map(str, chunk))
            f.write(line_str + " ->\n")
    print("Talimat dosyası hazır.")

def find_best_nail_weighted(current_idx, nails, str_pic, orig_pic, str_strength, error_weights, line_cache, ban_list, iteration, max_lines):
    best_improvement = -np.inf
    best_nail_idx = -1
    num_nails = len(nails)
    
    # Adaptif Aday Sayısı: Başta hızlı (60 aday), sona doğru hassas (daha çok aday)
    base_candidates = 60
    max_candidates = num_nails // 2 # Maksimum aday sayısı
    frac = min(1.0, iteration / max_lines)
    candidate_count = int(base_candidates + frac * (max_candidates - base_candidates))
    
    # Rastgele adaylar seç (Hızlandırma)
    search_indices = np.random.choice(num_nails, candidate_count, replace=False)

    for target_idx in search_indices:
        # 1. Taboo (Yasak) Listesi Kontrolü
        if target_idx in ban_list:
            continue
            
        # 2. Mesafe Kontrolü (Çok yakına gitme)
        dist = abs(current_idx - target_idx)
        if dist > num_nails // 2: dist = num_nails - dist
        if dist < MIN_DISTANCE: continue

        rr, cc, val = get_cached_line(current_idx, target_idx, line_cache, nails)
        if rr.size == 0: continue
        
        # --- GELİŞMİŞ HATA HESAPLAMA ---
        # Hata = (Mevcut - Hedef)^2 * Önem Haritası
        # Önem Haritası (error_weights) sayesinde gözlere ve karanlık bölgelere farklı davranıyoruz.
        
        # Mevcut Durum Hatası
        diff_current = (str_pic[rr, cc] - orig_pic[rr, cc]) ** 2
        current_error = np.sum(diff_current * error_weights[rr, cc])
        
        # Yeni Durum (Simülasyon) Hatası
        new_line_vals = str_pic[rr, cc] + str_strength * val
        new_line_vals = np.clip(new_line_vals, 0, 1)
        
        diff_new = (new_line_vals - orig_pic[rr, cc]) ** 2
        new_error = np.sum(diff_new * error_weights[rr, cc])
        
        improvement = current_error - new_error

        if improvement > best_improvement:
            best_improvement = improvement
            best_nail_idx = target_idx

    return best_nail_idx, best_improvement

def create_art(nails, orig_pic, str_pic, str_strength, max_lines):
    start_time = time()
    current_nail_idx = 0
    pull_order = [0]
    num_nails = len(nails)
    
    # 1. Önbellek Hazırla
    line_cache = cache_lines(nails)
    
    # 2. Değişkenler
    recent_nails = [0]
    plateau_patience = 400 
    no_gain_steps = 0

    # --- GELİŞMİŞ ÖNEM HARİTASI (HYBRID IMPORTANCE MAP) OLUŞTURMA ---
    print("Gelişmiş Önem Haritası (Karanlık + Kenarlar) oluşturuluyor...")
    h, w = orig_pic.shape
    
    # A. Karanlık Koruma (Darkness Protection)
    # Şapka gibi karanlık (düşük değerli) pikselleri korumak için ceza puanı veriyoruz.
    # 0.0 (siyah) olan yere ip atarsan ceza katsayısı yüksek olur.
    base_weights = 1.0 + (1.0 - orig_pic) * 6.0
    
    # B. Kenar Vurgusu (Edge Emphasis) - YENİ ÖZELLİK
    # Preprocess edilmiş görselin üzerinden bir kez daha geçip keskin kenarları (gözler, dudaklar) buluyoruz.
    img_u8 = (orig_pic * 255).astype(np.uint8)
    edges = cv2.Canny(img_u8, 50, 150).astype(np.float32) / 255.0
    # Kenarlara ekstra +3.0 önem puanı ekliyoruz. Algoritma burayı es geçemeyecek.
    edge_weights = edges * 3.0
    
    # C. Merkez Odaklaması (Radial Mask)
    # Kenarlardaki (çerçeveye yakın) hataları çok önemseme, merkeze odaklan.
    yy, xx = np.ogrid[:h, :w]
    cy, cx = h / 2.0, w / 2.0
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r_norm = r / (0.5 * min(h, w))
    radial_mask = np.clip(1.0 - 0.35 * r_norm, 0.5, 1.0)
    
    # Çene ve saç üstü için ek önem: alt ve üst bantlara daha güçlü ağırlık
    chin_boost = (yy > h * 0.55).astype(np.float32) * 0.5
    hair_boost = (yy < h * 0.25).astype(np.float32) * 0.4
    
    # HEPSİNİ BİRLEŞTİR: (Karanlık + Kenarlar + Bölgesel destek) * Merkez Maskesi
    error_weights = (base_weights + edge_weights + chin_boost + hair_boost) * radial_mask
    
    # Haritayı hafif yumuşat (Keskin piksel hatalarını önler)
    error_weights = cv2.GaussianBlur(error_weights, (3, 3), 0)
    
    # --- ÜRETİM DÖNGÜSÜ ---
    print(f"Üretim başlıyor: {max_lines} hat çizilecek...")
    
    for i in range(max_lines):
        if i % 500 == 0: print(f"İlerleme: {i}/{max_lines}...")

        # Dinamik İp Parlaklığı:
        # Başlangıçta daha belirgin çizgiler, sona doğru detay için daha ince çizgiler.
        dynamic_strength = str_strength * (0.8 + 0.2 * (1.0 - i / max_lines))

        # Taboo Listesi: Son N çiviyi yasakla
        ban_list = set(recent_nails[-BAN_DEPTH:])
        
        best_nail_idx, improvement = find_best_nail_weighted(
            current_nail_idx, nails, str_pic, orig_pic, dynamic_strength, 
            error_weights, line_cache, ban_list, i, max_lines
        )
        
        # Tıkanma Kontrolü
        if best_nail_idx == -1 or improvement <= 0:
             # Eğer iyileştirme bulamazsan, rastgele (ama mantıklı) bir yere atla
             for _ in range(50):
                 candidate = np.random.randint(0, num_nails)
                 if candidate not in ban_list and abs(candidate - current_nail_idx) > MIN_DISTANCE:
                     best_nail_idx = candidate
                     break
             no_gain_steps += 1
        else:
             no_gain_steps = 0

        # Erken Durdurma (Eğer 400 adım boyunca iyileşme yoksa bitir)
        if no_gain_steps > plateau_patience:
            print(f"Erken durma: {i}. adımda uzun süredir iyileşme yok.")
            break

        # Çizgiyi Tuvale İşle
        line_vals, rr, cc = get_aa_line(current_nail_idx, best_nail_idx, dynamic_strength, str_pic, nails, line_cache)
        if rr.size == 0: continue
        
        str_pic[rr, cc] = line_vals
        pull_order.append(best_nail_idx)
        current_nail_idx = best_nail_idx
        recent_nails.append(best_nail_idx)

    print(f"Tamamlandı. Süre: {time() - start_time:.2f}s")
    return pull_order, str_pic

# --- Ana Çalıştırma ---
if __name__ == "__main__":
    print("Görüntü İşleniyor (Preprocessing)...")
    processor = StringArtPreprocess(size=OUTPUT_SIZE) 
    
    # Preprocess dosyasını çalıştır ve hedef resmi al
    # Not: Preprocess dosyanız tek bir 'return' yapıyorsa (target), direkt alırız.
    target_img = processor.run(INPUT_IMAGE)
    
    # Kontrol için ön işleme sonucunu kaydet
    cv2.imwrite("preprocessed_target.png", (target_img * 255).astype(np.uint8))

    # Algoritma Hazırlığı
    num_nails = NUM_NAILS
    nails = create_circle_nail_positions(OUTPUT_SIZE, OUTPUT_SIZE, NUM_NAILS)
    canvas = init_black_canvas(OUTPUT_SIZE, OUTPUT_SIZE)

    # Sanatı Başlat
    pull_order, final_canvas = create_art(nails, target_img, canvas, LINE_WEIGHT, MAX_LINES)

    # Sonucu Kaydet
    final_output = (final_canvas * 255).astype(np.uint8)
    cv2.imwrite("final_string_art.png", final_output)
    
    # Talimatları Kaydet
    save_instructions(pull_order, "instructions.txt")
    
    print("İşlem Bitti! 'final_string_art.png' ve 'instructions.txt' dosyalarını kontrol edin.")

    # Ekranda Göster
    plt.figure(figsize=(10, 10))
    plt.imshow(final_output, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')
    plt.show()