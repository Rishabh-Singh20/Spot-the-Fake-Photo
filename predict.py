#!/usr/bin/env python3
"""
predict.py  --  Real-photo vs Screen-recapture classifier (inference)

Loading priority:
  1. model.joblib  (full GBM + StandardScaler -- saved by train.py)
  2. model_weights.npz  (legacy linear proxy -- fallback only)
"""
import sys, time, io
import numpy as np
from PIL import Image, ImageFilter
import cv2


# ──────────────────────────────────────────────
# Image helpers
# ──────────────────────────────────────────────
def _load_rgb(path):
    img = Image.open(path).convert('RGB')
    if max(img.size) > 512:
        img.thumbnail((512, 512), Image.LANCZOS)
    return img


def _np_rgb(img):
    return np.asarray(img, dtype=np.uint8)


def _gray(img):
    return np.asarray(img.convert('L'), dtype=np.uint8)


# ──────────────────────────────────────────────
# Feature extractors
# ──────────────────────────────────────────────
def _feat_original5(img, gray, rgb):
    blur  = np.asarray(img.filter(ImageFilter.GaussianBlur(radius=1)).convert('L'), dtype=np.uint8)
    lap   = float(cv2.Laplacian(blur, cv2.CV_32F).var())
    contrast  = float(gray.std())
    edges     = cv2.Canny(gray, 60, 150)
    edge_frac = float(edges.mean() / 255.0)
    hsv   = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    glare = float(((hsv[:, :, 2] > 235) & (hsv[:, :, 1] < 50)).mean())
    f     = np.fft.fft2(gray.astype(np.float32))
    mag   = np.log1p(np.abs(np.fft.fftshift(f)))
    h, w  = mag.shape
    cy, cx = h // 2, w // 2
    Y, X   = np.ogrid[:h, :w]
    r      = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    max_r  = np.sqrt(cx ** 2 + cy ** 2) + 1e-6
    low    = mag[r < 0.12 * max_r].mean()
    mid    = mag[(r >= 0.12 * max_r) & (r < 0.52 * max_r)].mean()
    moire  = float(mid / (low + 1e-6))
    return moire, lap, contrast, edge_frac, glare


def _feat_jpeg_artifact(img):
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=75)
    buf.seek(0)
    compressed = np.asarray(Image.open(buf).convert('L'), dtype=np.float32)
    original   = np.asarray(img.convert('L'), dtype=np.float32)
    if compressed.shape != original.shape:
        compressed = cv2.resize(compressed, (original.shape[1], original.shape[0]))
    return float(np.mean(np.abs(original - compressed)))


def _feat_noise(gray):
    blurred = cv2.GaussianBlur(gray.astype(np.float32), (5, 5), 0)
    noise   = np.abs(gray.astype(np.float32) - blurred)
    return float(noise.mean()), float(noise.std())


def _feat_uniformity(gray):
    lap_abs = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
    return float((lap_abs < 5).mean())


def _feat_color_stats(rgb):
    feats = []
    for c in range(3):
        ch = rgb[:, :, c].astype(np.float32).flatten()
        feats.append(float(ch.mean()))
        feats.append(float(ch.std()))
        feats.append(float(np.percentile(ch, 25)))
        feats.append(float(np.percentile(ch, 75)))
    return feats


def _feat_resolution(img):
    w, h   = img.size
    aspect = float(w) / float(h) if h > 0 else 1.0
    log_px = float(np.log1p(w * h))
    return aspect, log_px


def _feat_saturation(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1].astype(np.float32)
    return float(sat.mean()), float(sat.std())


def _feat_unique_color_ratio(rgb):
    small  = cv2.resize(rgb, (64, 64))
    pixels = small.reshape(-1, 3)
    unique = len(np.unique(pixels, axis=0))
    return float(unique) / float(len(pixels))


# ──────────────────────────────────────────────
# Feature names (used by train.py diagnostics)
# ──────────────────────────────────────────────
FEATURE_NAMES = [
    'moire', 'laplacian', 'contrast', 'edge_frac', 'glare',
    'jpeg_artifact',
    'noise_mean', 'noise_std',
    'uniformity',
    'r_mean', 'r_std', 'r_p25', 'r_p75',
    'g_mean', 'g_std', 'g_p25', 'g_p75',
    'b_mean', 'b_std', 'b_p25', 'b_p75',
    'aspect_ratio', 'log_pixel_count',
    'sat_mean', 'sat_std',
    'unique_color_ratio',
]
N_FEATURES = len(FEATURE_NAMES)  # 26


# ──────────────────────────────────────────────
# Master feature vector -- accepts a PIL image
# ──────────────────────────────────────────────
def _features_pil(img):
    """Extract the 26-dim feature vector from an already-loaded PIL image."""
    rgb  = _np_rgb(img)
    gray = _gray(img)

    moire, lap, contrast, edge_frac, glare = _feat_original5(img, gray, rgb)
    jpeg_art              = _feat_jpeg_artifact(img)
    noise_mean, noise_std = _feat_noise(gray)
    uniformity            = _feat_uniformity(gray)
    color_stats           = _feat_color_stats(rgb)
    aspect, log_px        = _feat_resolution(img)
    sat_mean, sat_std     = _feat_saturation(rgb)
    ucr                   = _feat_unique_color_ratio(rgb)

    return np.array([
        moire, lap, contrast, edge_frac, glare,
        jpeg_art,
        noise_mean, noise_std,
        uniformity,
        *color_stats,
        aspect, log_px,
        sat_mean, sat_std,
        ucr,
    ], dtype=np.float32)


def _features(img_path):
    """Extract features from an image file path."""
    return _features_pil(_load_rgb(img_path))


# ──────────────────────────────────────────────
# Fallback weights (dot-product, used if neither
# model.joblib nor model_weights.npz exist)
# ──────────────────────────────────────────────
DEFAULT_MEAN = np.zeros(N_FEATURES, dtype=np.float32)
DEFAULT_STD  = np.ones(N_FEATURES,  dtype=np.float32)
DEFAULT_W    = np.zeros(N_FEATURES, dtype=np.float32)
DEFAULT_B    = 0.0


# ──────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────
def predict_score(img_path):
    """
    Returns float in [0, 1].
      < 0.5  ->  real photo  (label 0)
      >= 0.5 ->  screen recapture  (label 1)
    """
    x = _features(img_path)

    # --- Priority 1: full GBM model saved by train.py ---
    try:
        import joblib
        bundle = joblib.load('model.joblib')
        clf    = bundle['clf']
        scaler = bundle['scaler']
        x_scaled = scaler.transform(x.reshape(1, -1))
        return float(clf.predict_proba(x_scaled)[0, 1])
    except Exception:
        pass

    # --- Priority 2: legacy linear proxy (model_weights.npz) ---
    try:
        m    = np.load('model_weights.npz', allow_pickle=False)
        mean = m['mean']; std = m['std']; w = m['w']; b = float(m['b'])
        if len(w) < N_FEATURES:
            pad  = N_FEATURES - len(w)
            w    = np.concatenate([w,    np.zeros(pad, dtype=np.float32)])
            mean = np.concatenate([mean, np.zeros(pad, dtype=np.float32)])
            std  = np.concatenate([std,  np.ones(pad,  dtype=np.float32)])
        x_scaled = (x - mean) / (std + 1e-6)
        z = float(np.dot(w, x_scaled) + b)
        return float(1.0 / (1.0 + np.exp(-z)))
    except Exception:
        pass

    # --- Fallback: untrained defaults ---
    x_scaled = (x - DEFAULT_MEAN) / (DEFAULT_STD + 1e-6)
    z = float(np.dot(DEFAULT_W, x_scaled) + DEFAULT_B)
    return float(1.0 / (1.0 + np.exp(-z)))


def main():
    if len(sys.argv) < 2:
        print('Usage: python predict.py <image_path>')
        sys.exit(1)
    t0    = time.perf_counter()
    score = predict_score(sys.argv[1])
    t1    = time.perf_counter()
    label = 'SCREENSHOT/RECAPTURE' if score >= 0.5 else 'REAL PHOTO'
    print(f'{score:.4f}  ->  {label}')
    sys.stderr.write(f'[latency] {(t1 - t0) * 1000:.1f} ms\n')


if __name__ == '__main__':
    main()
