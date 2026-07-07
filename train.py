#!/usr/bin/env python3
"""
train.py  --  Real-photo vs Screen-recapture classifier (training)
"""
import argparse, glob, os, joblib
import numpy as np
from PIL import Image, ImageEnhance
from sklearn.model_selection import StratifiedGroupKFold, train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.inspection import permutation_importance
from predict import _features_pil, _load_rgb, FEATURE_NAMES, N_FEATURES


# ──────────────────────────────────────────────
# Augmentation
# ──────────────────────────────────────────────
def _augment(img):

    variants = []

    variants.append(ImageEnhance.Brightness(img).enhance(0.65))
    variants.append(ImageEnhance.Brightness(img).enhance(1.45))

    variants.append(ImageEnhance.Contrast(img).enhance(1.4))

    arr = np.asarray(img, dtype=np.float32)
    arr[:, :, 0] = np.clip(arr[:, :, 0] * 0.88, 0, 255) 
    arr[:, :, 2] = np.clip(arr[:, :, 2] * 1.18, 0, 255)  
    variants.append(Image.fromarray(arr.astype(np.uint8)))

    w, h   = img.size
    left   = int(w * 0.075); top = int(h * 0.075)
    right  = w - left;        bottom = h - top
    variants.append(img.crop((left, top, right, bottom)).resize((w, h), Image.LANCZOS))

    return variants


# ──────────────────────────────────────────────
# Data loading with augmentation
# ──────────────────────────────────────────────
def load_folder(folder, label, augment=True):
    if not os.path.isdir(folder):
        return [], [], []
    paths = []
    for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp',
                '*.JPG', '*.JPEG', '*.PNG', '*.BMP', '*.WEBP'):
        paths.extend(glob.glob(os.path.join(folder, ext)))
        paths.extend(glob.glob(os.path.join(folder, '**', ext), recursive=True))
    paths = sorted(set(paths))

    X, y, groups = [], [], []
    for group_id, p in enumerate(paths):
        try:
            img  = _load_rgb(p)
            imgs = [img] + (_augment(img) if augment else [])
            for aug_img in imgs:
                X.append(_features_pil(aug_img))
                y.append(label)
                groups.append(group_id)  
            print(f'  [{group_id+1}/{len(paths)}] {os.path.relpath(p)} '
                  f'(+{len(imgs)-1} augments)', end='\r')
        except Exception as e:
            print(f'\n  skip {os.path.relpath(p)}: {e}')
    if paths:
        print()
    return X, y, groups


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description='Train real-vs-recapture classifier')
    ap.add_argument('--real_dir',      default='real')
    ap.add_argument('--screen_dir',    default='screen')
    ap.add_argument('--n_estimators',  type=int,   default=300)
    ap.add_argument('--learning_rate', type=float, default=0.05)
    ap.add_argument('--max_depth',     type=int,   default=4)
    ap.add_argument('--no_augment',    action='store_true',
                    help='Disable augmentation (faster, for debugging)')
    args = ap.parse_args()

    augment = not args.no_augment

    # -- Load data --------------------------------
    print(f'\nLoading real photos from   : {args.real_dir}')
    Xr, yr, gr = load_folder(args.real_dir,   0, augment=augment)
    print(f'Loading recaptures from    : {args.screen_dir}')
    Xs, ys, gs = load_folder(args.screen_dir, 1, augment=augment)

    n_real_orig   = len([g for g in gr if g == gr[0]]) and len(set(gr))
    n_screen_orig = len(set(gs))
    print(f'\nOriginal images : {n_real_orig} real + {n_screen_orig} screen')
    print(f'After augment   : {len(Xr)} real + {len(Xs)} screen = {len(Xr)+len(Xs)} total')

    if n_real_orig == 0 or n_screen_orig == 0:
        print('ERROR: Need images in both folders.')
        return

    X      = np.array(Xr + Xs, dtype=np.float32)
    y      = np.array(yr + ys, dtype=np.int32)
    # Offset screen group ids so they don't collide with real group ids
    groups = np.array(gr + [g + max(gr) + 1 for g in gs], dtype=np.int32)

    # -- Scale ------------------------------------
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # -- Cross-validation (group-aware) -----------
    print(f'\n-- 5-Fold Cross-Validation (StratifiedGroupKFold) --')
    clf_cv = GradientBoostingClassifier(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        subsample=0.8,
        min_samples_leaf=2,
        random_state=42,
    )
    n_splits = min(5, len(set(groups)) // 2)
    if n_splits < 2:
        print('  (Not enough groups for CV - skipping)')
        cv_scores = np.array([])
    else:
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_scores = cross_val_score(clf_cv, X_scaled, y, cv=cv,
                                    groups=groups, scoring='accuracy')
        print(f'  CV accuracy : {cv_scores.mean()*100:.1f}% +/- {cv_scores.std()*100:.1f}%')
        print(f'  Per-fold    : {[f"{s*100:.1f}%" for s in cv_scores]}')

    # -- Held-out split (group-aware) -------------
    unique_groups = np.unique(groups)
    np.random.seed(42)
    test_groups  = np.random.choice(unique_groups,
                                    size=max(1, int(len(unique_groups) * 0.2)),
                                    replace=False)
    test_mask  = np.isin(groups, test_groups)
    train_mask = ~test_mask

    X_train, X_test = X_scaled[train_mask], X_scaled[test_mask]
    y_train, y_test = y[train_mask],         y[test_mask]

    # -- Train final model ------------------------
    print(f'\n-- Training final model ({args.n_estimators} estimators) --')
    clf = GradientBoostingClassifier(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        subsample=0.8,
        min_samples_leaf=2,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    # -- Evaluation -------------------------------
    pred = clf.predict(X_test)
    acc  = accuracy_score(y_test, pred)
    print(f'\n-- Held-out Accuracy: {acc * 100:.1f}% --')
    print('\nClassification Report:')
    print(classification_report(y_test, pred, target_names=['real', 'recapture']))
    print('Confusion Matrix (rows=actual, cols=predicted):')
    cm = confusion_matrix(y_test, pred)
    print(f'  real->real={cm[0,0]}  real->screen={cm[0,1]}')
    print(f'  screen->real={cm[1,0]}  screen->screen={cm[1,1]}')

    # -- Feature importance -----------------------
    print('\n-- Feature Importances (permutation) --')
    try:
        pi    = permutation_importance(clf, X_test, y_test,
                                       n_repeats=15, random_state=42, n_jobs=-1)
        order = np.argsort(pi.importances_mean)[::-1]
        for rank, idx in enumerate(order[:15], 1):
            bar = '#' * max(1, int(pi.importances_mean[idx] * 200))
            print(f'  {rank:2}. {FEATURE_NAMES[idx]:<22} {pi.importances_mean[idx]:+.4f}  {bar}')
    except Exception as e:
        print(f'  (unavailable: {e})')

    # -- Save full GBM model (Fix #1) -------------
    bundle = {'clf': clf, 'scaler': scaler}
    joblib.dump(bundle, 'model.joblib', compress=3)
    print(f'\nSaved model.joblib  (GBM + scaler -- used by predict.py)')

    # -- Also save linear proxy as fallback -------
    fi       = clf.feature_importances_.astype(np.float32)
    mean_r   = X[y == 0].mean(axis=0)
    mean_s   = X[y == 1].mean(axis=0)
    sign     = np.sign(mean_s - mean_r).astype(np.float32)
    sign[sign == 0] = 1.0
    w_proxy  = fi * sign
    logits   = ((X - scaler.mean_) / (scaler.scale_ + 1e-6)) @ w_proxy
    b_proxy  = float(-np.median(logits))
    np.savez('model_weights.npz',
             mean=scaler.mean_.astype(np.float32),
             std=scaler.scale_.astype(np.float32),
             w=w_proxy, b=np.float32(b_proxy))
    print('Saved model_weights.npz (linear fallback)')
    print('\nDone. Run: python predict.py <image_path>')


if __name__ == '__main__':
    main()
