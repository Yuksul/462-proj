import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import line_aa
from time import time
import cv2
from preprocess_stringart import StringArtPreprocess

# --- AYARLAR ---
INPUT_IMAGE = "testimage.jpeg"
NUM_NAILS = 280                # Standart sayıya döndük, karmaşayı azaltmak için
MAX_LINES = 3500               # İp sayısı
LINE_WEIGHT = 0.05             # İp parlaklığı
OUTPUT_SIZE = 800              
MIN_DISTANCE = 15              # Yan yana çivilere gitmesin

# --- Fonksiyonlar ---
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

def get_aa_line(from_pos, to_pos, str_strength, picture):
    rr, cc, val = line_aa(from_pos[0], from_pos[1], to_pos[0], to_pos[1])
    line = picture[rr, cc] + str_strength * val
    line = np.clip(line, a_min=0, a_max=1)
    return line, rr, cc

def find_best_nail_weighted(current_idx, nails, str_pic, orig_pic, str_strength, error_weights):
    best_improvement = -np.inf
    best_nail_idx = -1
    current_pos = nails[current_idx]
    
    num_nails = len(nails)
    
    # Hız ve Kalite Dengesi: Rastgele 60 çiviye bak
    search_indices = np.random.choice(num_nails, 60, replace=False)

    for target_idx in search_indices:
        # Mesafe kontrolü
        dist = abs(current_idx - target_idx)
        if dist > num_nails // 2: dist = num_nails - dist
        if dist < MIN_DISTANCE: continue

        target_pos = nails[target_idx]
        
        rr, cc, val = line_aa(current_pos[0], current_pos[1], target_pos[0], target_pos[1])
        
        # --- YENİ MANTIK: AĞIRLIKLI HATA ---
        # error_weights haritası, karanlık bölgelerde (şapka) çok yüksek değerlere sahip.
        # Bu sayede oraya ip atarsak "ceza" (hata artışı) çok büyük oluyor.
        
        # Mevcut Hata
        diff_current = str_pic[rr, cc] - orig_pic[rr, cc]
        # Hatayı ağırlık haritasıyla çarpıyoruz!
        current_error = np.sum((diff_current ** 2) * error_weights[rr, cc])
        
        # Yeni Hata
        new_line_vals = str_pic[rr, cc] + str_strength * val
        new_line_vals = np.clip(new_line_vals, 0, 1)
        
        diff_new = new_line_vals - orig_pic[rr, cc]
        new_error = np.sum((diff_new ** 2) * error_weights[rr, cc])
        
        improvement = current_error - new_error

        if improvement > best_improvement:
            best_improvement = improvement
            best_nail_idx = target_idx

    return best_nail_idx, best_improvement

def create_art(nails, orig_pic, str_pic, str_strength, max_lines):
    start_time = time()
    current_nail_idx = 0
    pull_order = [0]

    # --- AĞIRLIK HARİTASI OLUŞTURMA ---
    print("Generating Weight Map (Darkness Protection)...")
    # Karanlık yerler (0.0) -> Yüksek Ceza (5.0 kat)
    # Aydınlık yerler (1.0) -> Normal Ceza (1.0 kat)
    # Formül: 1 + (1 - pixel_value) * 6
    error_weights = 1.0 + (1.0 - orig_pic) * 6.0
    
    print(f"Starting generation with {max_lines} lines...")
    
    for i in range(max_lines):
        if i % 500 == 0: print(f"Progress: {i}/{max_lines}...")

        best_nail_idx, improvement = find_best_nail_weighted(
            current_nail_idx, nails, str_pic, orig_pic, str_strength, error_weights
        )
        
        if best_nail_idx == -1 or improvement <= 0:
             best_nail_idx = (current_nail_idx + np.random.randint(20, num_nails//2)) % num_nails

        line_vals, rr, cc = get_aa_line(nails[current_nail_idx], nails[best_nail_idx], str_strength, str_pic)
        str_pic[rr, cc] = line_vals
        
        pull_order.append(best_nail_idx)
        current_nail_idx = best_nail_idx

    print(f"Time: {time() - start_time:.2f}s")
    return pull_order, str_pic

if __name__ == "__main__":
    print("Preprocessing...")
    processor = StringArtPreprocess(size=OUTPUT_SIZE) 
    target_img = processor.run(INPUT_IMAGE)
    cv2.imwrite("preprocessed_target.png", (target_img * 255).astype(np.uint8))

    # --- Görsel Kontrol ---
    # Hedef resimde şapkanın SİMSİYAH olduğundan emin ol. 
    # Değilse preprocess'te kontrastı artırmamız gerekir.

    num_nails = NUM_NAILS
    nails = create_circle_nail_positions(OUTPUT_SIZE, OUTPUT_SIZE, NUM_NAILS)
    canvas = init_black_canvas(OUTPUT_SIZE, OUTPUT_SIZE)

    pull_order, final_canvas = create_art(nails, target_img, canvas, LINE_WEIGHT, MAX_LINES)

    final_output = (final_canvas * 255).astype(np.uint8)
    cv2.imwrite("final_string_art.png", final_output)
    print("Done.")

    plt.figure(figsize=(10, 10))
    plt.imshow(final_output, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')
    plt.show()