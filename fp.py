# fp.py
# Unimodal fingerprint recognition on SOCOFing (contact-based).
# Modes:
#   ident  -> subject ID classification (Top-1 accuracy)
#   verify -> Siamese verification (ROC-AUC, EER)
#
# Auto-detects data root from common locations relative to this file:
#   ../archive/SOCOFing
#   ../archive/socofing/SOCOFing
#
# Usage (from ~/biometrics/codes):
#   python fp.py --mode ident
#   python fp.py --mode verify
import argparse, os, random, math, glob, sys
from typing import List, Tuple
import numpy as np
from PIL import Image
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import roc_auc_score

# -----------------------
# Utilities
# -----------------------
def set_seed(s=1337):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def parse_filename(fn: str):
    """
    SOCOFing Real naming: 001_M_Left_index_finger.bmp
    Returns (subject_id:int, gender:str, hand:str, finger:str)
    """
    base = os.path.basename(fn)
    name, _ = os.path.splitext(base)
    parts = name.split('_')
    sid = int(parts[0])
    gender = parts[1] if len(parts) > 1 else "U"
    hand = parts[2] if len(parts) > 2 else "U"
    finger = parts[3] if len(parts) > 3 else "U"
    return sid, gender, hand, finger

def find_data_root(explicit_path: str | None):
    """Try to locate SOCOFing/Real with BMPs."""
    candidates = []
    if explicit_path:
        candidates.append(explicit_path)
    # Relative to this script:
    here = os.path.dirname(os.path.abspath(__file__))
    candidates += [
        os.path.normpath(os.path.join(here, "..", "archive", "SOCOFing")),
        os.path.normpath(os.path.join(here, "..", "archive", "socofing", "SOCOFing")),
        os.path.normpath(os.path.join(here, "SOCOFing")),
    ]
    tried = []
    for root in candidates:
        real_glob = os.path.join(root, "Real", "*.bmp")
        tried.append(real_glob)
        if len(glob.glob(real_glob)) > 0:
            return root
    raise SystemExit(
        "Could not find SOCOFing. Tried:\n  " + "\n  ".join(tried) +
        "\nPass --data_root /full/path/to/SOCOFing"
    )

def train_val_split_by_subject(paths: List[str], val_rate=0.2):
    subjects = sorted({parse_filename(p)[0] for p in paths})
    random.shuffle(subjects)
    n_val = max(1, int(len(subjects) * val_rate))
    val_subjects = set(subjects[:n_val])
    tr, va = [], []
    for p in paths:
        (va if parse_filename(p)[0] in val_subjects else tr).append(p)
    return tr, va

def ensure_out_dir(p):
    os.makedirs(p, exist_ok=True)

def eer_from_scores(scores, labels):
    s = np.array(scores); y = np.array(labels)
    thr = np.linspace(-1, 1, 400)
    fars, frrs = [], []
    for t in thr:
        yhat = (s >= t).astype(int)
        far = ((yhat==1) & (y==0)).sum() / max((y==0).sum(), 1)
        frr = ((yhat==0) & (y==1)).s
