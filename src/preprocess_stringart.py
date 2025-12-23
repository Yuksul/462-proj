import cv2
import numpy as np
import os

SIZE = 800

class StringArtPreprocess:
    def __init__(self, size=SIZE):
        self.size = size

    def read(self, path):
        if not os.path.isfile(path):
            raise ValueError(f"Image not found: {path}")
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Image could not be loaded: {path}")
        return img

    def resize_and_center(self, img):
        h, w = img.shape[:2]
        scale = self.size / min(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        start_x = (new_w - self.size) // 2
        start_y = (new_h - self.size) // 2
        cropped = resized[start_y:start_y+self.size, start_x:start_x+self.size]
        return cropped

    def run(self, path):
        img = self.read(path)
        sq = self.resize_and_center(img)
        
        # 1. Griye Çevir
        g = cv2.cvtColor(sq, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        
        # 2. Global kontrast: siyah noktasını yukarı çek
        g = np.clip((g - 0.15) * 1.2, 0, 1)
        
        # 3. CLAHE (Yerel kontrast)
        g8 = (g * 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        target = clahe.apply(g8).astype(np.float32) / 255.0

        # 3.5 Arka plan bastırma: düşük frekanslı bileşeni geri it
        bg = cv2.GaussianBlur(target, (31, 31), 0)
        target = np.clip(target - 0.6 * (bg - target.mean()), 0, 1)
        
        # 4. KOYULAŞTIRMA (Gamma > 1)
        # Hedef görüntüyü biraz karanlık yapıyoruz ki ipler seyrek olsun, detaylar belli olsun.
        target = np.power(target, 1.25)
        
        # 5. Kenar vurgusu: kenar haritasını hedefle harmanla
        edges = cv2.Canny((target * 255).astype(np.uint8), 40, 120).astype(np.float32) / 255.0
        target = np.clip(0.9 * target + 0.1 * edges, 0.0, 1.0)
        
        # 6. Hafif yumuşatma: ceza haritasında keskin geçişleri azalt
        target = cv2.GaussianBlur(target, (3, 3), 0)
        
        # 7. Maskeleme
        h, w = target.shape
        y, x = np.ogrid[:h, :w]
        center = (h//2, w//2)
        radius = min(h, w) // 2 - 6
        mask = (x - center[1])**2 + (y - center[0])**2 <= radius**2
        target[~mask] = 0
        
        return np.clip(target, 0, 1)