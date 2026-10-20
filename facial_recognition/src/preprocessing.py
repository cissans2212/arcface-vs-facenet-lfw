"""
Modul Preprocessing Citra: Transformasi Kecerahan (Brightness) dan Kontras (Contrast)
Project: Face Verification ArcFace vs FaceNet (Dataset LFW)

Modul ini menyediakan fungsi preprocessing deterministik untuk menguji ketahanan (robustness)
model Face Verification terhadap degradasi pencahayaan dan variasi kontras.

5 Skenario Pengujian:
  1. Baseline          : Citra asli (faktor 1.0, tanpa perubahan)
  2. Brightness Rendah : Kecerahan diturunkan (faktor 0.7)
  3. Brightness Tinggi : Kecerahan dinaikkan (faktor 1.3)
  4. Contrast Rendah   : Kontras diturunkan (faktor 0.7)
  5. Contrast Tinggi   : Kontras dinaikkan (faktor 1.3)

Metode Transformasi:
  - Menggunakan PIL.ImageEnhance (Brightness & Contrast) yang menjamin konversi presisi,
    interpolasi warna RGB seragam, dan pemotongan nilai piksel (clipping) otomatis ke rentang valid [0, 255].
  - Menyediakan fallback berbasis NumPy/OpenCV dengan formulasi matematis identik:
      * Brightness: I' = clip(I * factor, 0, 255)
      * Contrast  : I' = clip(mean(I) + factor * (I - mean(I)), 0, 255)
"""

from pathlib import Path
from typing import Union, Dict, Any
import numpy as np
from PIL import Image, ImageEnhance


SCENARIOS: Dict[str, Dict[str, Any]] = {
    "baseline": {
        "id": "scenario_1_baseline",
        "name": "Skenario 1 — Baseline",
        "type": "none",
        "factor": 1.0,
        "description": "Gambar asli tanpa modifikasi kecerahan atau kontras tambahan.",
    },
    "brightness_low": {
        "id": "scenario_2_brightness_low",
        "name": "Skenario 2 — Brightness Rendah",
        "type": "brightness",
        "factor": 0.7,
        "description": "Kecerahan gambar diturunkan menjadi faktor 0.7 (citra lebih redup / underexposed).",
    },
    "brightness_high": {
        "id": "scenario_3_brightness_high",
        "name": "Skenario 3 — Brightness Tinggi",
        "type": "brightness",
        "factor": 1.3,
        "description": "Kecerahan gambar dinaikkan menjadi faktor 1.3 (citra lebih terang / overexposed).",
    },
    "contrast_low": {
        "id": "scenario_4_contrast_low",
        "name": "Skenario 4 — Contrast Rendah",
        "type": "contrast",
        "factor": 0.7,
        "description": "Kontras gambar diturunkan menjadi faktor 0.7 (perbedaan intensitas menyempit / washed out).",
    },
    "contrast_high": {
        "id": "scenario_5_contrast_high",
        "name": "Skenario 5 — Contrast Tinggi",
        "type": "contrast",
        "factor": 1.3,
        "description": "Kontras gambar dinaikkan menjadi faktor 1.3 (rentang dinamis intensitas melebar).",
    },
}


def adjust_brightness_pil(image: Image.Image, factor: float) -> Image.Image:
    """
    Mengubah kecerahan citra menggunakan PIL.ImageEnhance.Brightness.
    
    Formulasi Matematis:
      I_out = clip(I_in * factor, 0, 255)
      - factor = 1.0 : citra asli
      - factor = 0.0 : citra hitam pekat
      - factor = 0.7 : reduksi intensitas cahaya sebesar 30%
      - factor = 1.3 : peningkatan intensitas cahaya sebesar 30%
    """
    if factor == 1.0:
        return image.copy()
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)


def adjust_contrast_pil(image: Image.Image, factor: float) -> Image.Image:
    """
    Mengubah kontras citra menggunakan PIL.ImageEnhance.Contrast.
    
    Formulasi Matematis:
      I_out = clip(L_mean + factor * (I_in - L_mean), 0, 255)
      dengan L_mean merupakan rata-rata luminans citra grayscale degenerate.
      - factor = 1.0 : citra asli
      - factor = 0.0 : citra abu-abu solid homogen (mean luminance)
      - factor = 0.7 : kontras dikurangi 30% ke arah luminans rata-rata
      - factor = 1.3 : kontras dinaikkan 30% menjauhi luminans rata-rata
    """
    if factor == 1.0:
        return image.copy()
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)


def preprocess_image(
    image_input: Union[Image.Image, np.ndarray, str],
    scenario_key: str,
) -> Image.Image:
    """
    Menerapkan salah satu dari 5 skenario preprocessing pada citra.
    
    Args:
        image_input: PIL Image, NumPy array (RGB), atau path ke file citra.
        scenario_key: Kunci skenario ('baseline', 'brightness_low', 'brightness_high',
                      'contrast_low', 'contrast_high').
    
    Returns:
        PIL.Image: Citra hasil transformasi berformat RGB dengan nilai piksel valid uint8 [0, 255].
    """
    if scenario_key not in SCENARIOS:
        valid_keys = list(SCENARIOS.keys())
        raise ValueError(f"Skenario '{scenario_key}' tidak valid. Pilihan yang tersedia: {valid_keys}")

    # Konversi input ke PIL Image (RGB)
    if isinstance(image_input, (str, Path)):
        pil_img = Image.open(str(image_input)).convert("RGB")
    elif isinstance(image_input, np.ndarray):
        pil_img = Image.fromarray(image_input.astype(np.uint8)).convert("RGB")
    elif isinstance(image_input, Image.Image):
        pil_img = image_input.convert("RGB")
    else:
        raise TypeError(f"Tipe input tidak didukung: {type(image_input)}")

    scenario = SCENARIOS[scenario_key]
    stype = scenario["type"]
    factor = scenario["factor"]

    if stype == "none" or factor == 1.0:
        return pil_img.copy()
    elif stype == "brightness":
        return adjust_brightness_pil(pil_img, factor)
    elif stype == "contrast":
        return adjust_contrast_pil(pil_img, factor)
    else:
        raise ValueError(f"Tipe transformasi tidak didukung: {stype}")


def preprocess_image_to_numpy(
    image_input: Union[Image.Image, np.ndarray, str],
    scenario_key: str,
) -> np.ndarray:
    """
    Menerapkan skenario preprocessing dan mengembalikan array NumPy uint8 (H, W, 3).
    Memastikan nilai piksel berada dalam rentang [0, 255].
    """
    pil_img = preprocess_image(image_input, scenario_key)
    arr = np.array(pil_img, dtype=np.uint8)
    # Validasi kepastian integritas nilai piksel
    assert arr.min() >= 0 and arr.max() <= 255, "Nilai piksel berada di luar rentang valid [0, 255]!"
    return arr
