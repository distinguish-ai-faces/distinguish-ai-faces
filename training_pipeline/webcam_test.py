#!/usr/bin/env python
"""
Webcam görüntüsü alıp AI yüz tespiti modeli ile tahmin yapan script.
"""
import os
import sys
import argparse
import cv2
import numpy as np
import time
from pathlib import Path

# Src klasörünü Python yoluna ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Predict modülünü içe aktar
from src.predict import Predictor

def parse_args():
    """Komut satırı argümanlarını ayrıştır."""
    parser = argparse.ArgumentParser(description='Webcam görüntüsünü AI yüz tespiti modeliyle test et')
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Kullanılacak model dosyasının yolu (örn. outputs/final_model.h5)'
    )
    
    parser.add_argument(
        '--img-size',
        type=int,
        nargs=2,
        default=[224, 224],
        help='Görüntü boyutları (genişlik, yükseklik)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='webcam_results',
        help='Çıktı dosyalarının kaydedileceği dizin'
    )
    
    parser.add_argument(
        '--capture-delay',
        type=int,
        default=3,
        help='Webcam görüntüsü çekmeden önce beklenecek saniye sayısı'
    )
    
    return parser.parse_args()

def main():
    """Ana fonksiyon."""
    # Argümanları ayrıştır
    args = parse_args()
    
    # Çıktı dizinini oluştur
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Modeli yükle
    predictor = Predictor(args.model, img_size=tuple(args.img_size))
    
    # Webcam'i aç
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Webcam açılamadı!")
        return 1
    
    print(f"Webcam açıldı. {args.capture_delay} saniye sonra görüntü çekilecek...")
    
    # Geri sayım göster
    for i in range(args.capture_delay, 0, -1):
        print(f"{i}...")
        time.sleep(1)
    
    print("Görüntü çekiliyor!")
    
    # Görüntüyü çek
    ret, frame = cap.read()
    
    if not ret:
        print("Görüntü çekilemedi!")
        cap.release()
        return 1
    
    # Görüntüyü kaydet
    timestamp = int(time.time())
    image_path = output_dir / f"webcam_{timestamp}.jpg"
    cv2.imwrite(str(image_path), frame)
    
    print(f"Görüntü {image_path} olarak kaydedildi.")
    
    # Webcam'i kapat
    cap.release()
    
    # Modelle tahmin yap
    print("Model ile tahmin yapılıyor...")
    result = predictor.predict_single(image_path)
    
    # Sonuçları görselleştir
    predictor.visualize_prediction(
        image_path,
        show_confidence=True,
        save_path=output_dir / f"result_{timestamp}.jpg"
    )
    
    # Sonuçları ekrana yazdır
    print("\n------------ SONUÇLAR ------------")
    print(f"Sınıf: {'AI-üretilmiş yüz' if result['prediction'] == 'AI' else 'Gerçek insan yüzü'}")
    print(f"Güven Oranı: {result['confidence']:.4f}")
    print(f"Görselleştirilmiş sonuç: {output_dir / f'result_{timestamp}.jpg'}")
    print("-----------------------------------\n")
    
    return 0

if __name__ == '__main__':
    sys.exit(main()) 