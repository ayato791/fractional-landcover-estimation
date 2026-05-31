#Sentinel-2とEmbeddingV1の両方を用いたデータセットでの2D-CNNモデル
import os, json, random, glob, time
from datetime import datetime

import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ========= 設定 =========
INPUT_DIR   = "/mnt/d/kanno/s2_oem/combined_tif"   # S2 + Embedding + 8バンドTIFが入っているディレクトリ
OUTPUT_DIR  = "/home/kanno/code/outputs/2dcnn/s2" # 出力ディレクトリ
SEED        = 42
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.10
TEST_RATIO  = 0.20
assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-8

S2_BANDS  = 10
EMB_BANDS = 64
N_CLASSES  = 8
OEM_BANDS  = N_CLASSES  # OEMラベルバンド数


# 学習ハイパラ
BATCH_SIZE_2D = 256
EPOCHS        = 200
LR            = 3e-4
WEIGHT_DECAY  = 1e-4
DROPOUT       = 0.20
BASE_CH       = 16
EARLY_STOP_PATIENCE = 20

ALPHA_AITCHISON = 0.0   # Aitchisonの割合（例）
USE_HUBER_AITCH = True
HUBER_DELTA     = 0.5

# データ仕様
N_CLASSES   = 8
PATCH_SIZE  = 5         # 5/7/9でアブレーション
CENTER_STRIDE_TRAIN = 3  # 学習サンプル間引き
CENTER_STRIDE_VAL   = 3
CENTER_STRIDE_TEST  = 1

# 数値安定
DELTA_ZERO  = 1e-5
EPS_LOG     = 1e-6

DEVICE      = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
"""
クラス
1 裸地2 草地3 開発地4 道路
5 森林6 水域7 農地8 建物
"""
# ========= ユーティリティ =========
def set_seed(s=42):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)

# NaNが含まれるため除く
SKIP = {
    "kyoto_54_combined.tif",
    "dhaka_39_combined.tif",
    "santiago_18_combined.tif",
    "santiago_5_combined.tif",
    "soriano_4_combined.tif",
    "soriano_7_combined.tif",
}

def list_tifs(d):
    files = sorted(glob.glob(os.path.join(d, "*.tif"))) + \
            sorted(glob.glob(os.path.join(d, "*.tiff")))
    return [f for f in files if os.path.basename(f) not in SKIP]

def load_XY_tif(path):
    with rasterio.open(path) as src:
        arr = src.read().astype(np.float32)  # (B, H, W)
        meta = src.meta.copy()

    total_bands = arr.shape[0]
    if total_bands <= N_CLASSES:
        raise ValueError(
            f"{os.path.basename(path)}: bands {total_bands} <= N_CLASSES({N_CLASSES})"
        )

    in_bands = total_bands - N_CLASSES
    X = arr[:in_bands]               # (in_bands, H, W) = S2 + Embedding
    Y = arr[in_bands:]               # (N_CLASSES, H, W) = OEM ratio
    return X, Y, meta

def infer_inbands_from_file(path: str) -> int:
    with rasterio.open(path) as src:
        total_bands = src.count
    if total_bands <= N_CLASSES:
        raise ValueError(f"{os.path.basename(path)}: bands {total_bands} <= N_CLASSES({N_CLASSES})")
    return total_bands - N_CLASSES  # 入力バンド数

def check_inbands_consistency(files: list[str]) -> int:
    if len(files) == 0:
        raise RuntimeError("No tif files.")
    inb0 = infer_inbands_from_file(files[0])
    bad = []
    for p in files[1:]:
        inb = infer_inbands_from_file(p)
        if inb != inb0:
            bad.append((os.path.basename(p), inb))
    if bad:
        msg = "\n".join([f"  {name}: in_bands={inb}" for name, inb in bad[:20]])
        raise RuntimeError(
            f"Input band count mismatch in directory.\n"
            f"Expected in_bands={inb0} but found:\n{msg}"
        )
    return inb0

def infer_dataset_mode(in_bands: int) -> str:
    if in_bands == S2_BANDS + EMB_BANDS:
        return "S2+EMB"
    if in_bands == S2_BANDS:
        return "S2"
    if in_bands == EMB_BANDS:
        return "EMB"
    raise RuntimeError(
        f"Unexpected in_bands={in_bands}. "
        f"Expected {S2_BANDS+EMB_BANDS} (S2+EMB) / {S2_BANDS} (S2) / {EMB_BANDS} (EMB)."
    )

def scale_input_X(X: np.ndarray, xmin: np.ndarray | None, xmax: np.ndarray | None, mode: str,
                  eps: float = 1e-6) -> np.ndarray:
    """
    X: (C,H,W)
    mode: "S2+EMB" / "S2" / "EMB"
    - S2 部分だけ min-max
    - EMB 部分は無変換
    """
    C, H, W = X.shape
    Xf = X.reshape(C, -1).T  # (N,C)

    Xn = Xf.copy()

    if mode in ("S2+EMB", "S2"):
        assert xmin is not None and xmax is not None
        # S2は先頭10バンド
        Xn[:, :S2_BANDS] = minmax_scale_X(Xf[:, :S2_BANDS],
                                          xmin[:, :S2_BANDS],
                                          xmax[:, :S2_BANDS],
                                          eps=eps)

    # mode=="EMB" は何もしない（Embeddingは無変換）
    return Xn.T.reshape(C, H, W).astype(np.float32)


def valid_mask(X4, Y8):
    # (H,W) で両方 finite な画素のみ True
    return np.isfinite(X4).all(axis=0) & np.isfinite(Y8).all(axis=0)

def multiplicative_replacement(Y, delta=1e-4): # ← 1e-5 から 1e-4
    """
    Y: (N,K), K=クラス数
    0 を delta に置き換えて合計を1に調整
    """
    K = Y.shape[1]
    Y = Y.copy()
    zeros = (Y <= 0)
    zc = zeros.sum(axis=1, keepdims=True)

    Y[zeros] = delta
    s = Y.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    Y = Y / s

    nonzero_mask = ~zeros
    nonzero_sum  = (Y * nonzero_mask).sum(axis=1, keepdims=True)
    nonzero_sum[nonzero_sum == 0] = 1.0

    scale = (1.0 - delta * zc)
    scale = np.clip(scale, 1e-8, None)

    Y[nonzero_mask] = (Y[nonzero_mask] /
                       nonzero_sum.repeat(K, axis=1)[nonzero_mask]) * \
                      scale.repeat(K, axis=1)[nonzero_mask]
    return Y.astype(np.float32)

def compute_minmax_over_train_s2only(train_files, mode: str, max_pix=200_000):
    """
    S2 を含むモードのときだけ、先頭10バンド(S2)について xmin/xmax を計算する。
    返り値 xmin/xmax は shape (1, in_bands) に揃えておく（Embedding側はダミーでOK）。
    """
    # in_bands を1枚目から取得
    X0, _, _ = load_XY_tif(train_files[0])
    in_bands = X0.shape[0]

    # xmin/xmax を in_bands 全体の形で用意（Embedding側は使わないのでダミー）
    xmin = np.zeros((1, in_bands), dtype=np.float32)
    xmax = np.ones((1, in_bands), dtype=np.float32)

    if mode == "EMB":
        # S2が無いので何もしない（ダミーのまま返す）
        return xmin, xmax

    # S2 がある場合：先頭10バンドだけ統計を取る
    mins, maxs = [], []
    for p in train_files:
        X, Y, _ = load_XY_tif(p)
        vm = valid_mask(X, Y)
        if not vm.any():
            continue
        Xs2 = X[:S2_BANDS, vm].T  # (N_valid, 10)
        if Xs2.shape[0] > max_pix:
            idx = np.random.choice(Xs2.shape[0], size=max_pix, replace=False)
            Xs2 = Xs2[idx]
        mins.append(Xs2.min(axis=0, keepdims=True))
        maxs.append(Xs2.max(axis=0, keepdims=True))

    if len(mins) == 0:
        raise RuntimeError("No valid pixels found to compute min-max for S2.")

    xmin_s2 = np.vstack(mins).min(axis=0, keepdims=True)
    xmax_s2 = np.vstack(maxs).max(axis=0, keepdims=True)

    xmin[:, :S2_BANDS] = xmin_s2.astype(np.float32)
    xmax[:, :S2_BANDS] = xmax_s2.astype(np.float32)
    return xmin, xmax
import numpy as np

def compute_minmax_over_train_s2emb(train_files, max_pix=200_000, eps=1e-6):
    """
    S2+EMB の入力X（例: 74バンド）について、学習trainからバンド毎min/maxを推定。
    max_pix: 使う画素数の上限（ランダムサンプル）
    """
    mins = None
    maxs = None
    seen = 0

    rng = np.random.default_rng(0)

    for tif_path in train_files:
        # ここは既存の読み込み関数に合わせる（あなたの断片に load_XY_tif が出てた）
        X, Y, meta = load_XY_tif(tif_path)  # X: (C,H,W)

        C, H, W = X.shape

        # valid_mask があるならそれを優先（雲・欠損・ラベル無効など除外）
        if "valid_mask" in globals():
            vm = valid_mask(X, Y)  # (H,W) bool
            idx_all = np.flatnonzero(vm.reshape(-1))
        else:
            idx_all = np.arange(H * W, dtype=np.int64)

        if idx_all.size == 0:
            continue

        # サンプル数決定（残り枠と相談）
        remain = max_pix - seen
        if remain <= 0:
            break

        take = min(remain, idx_all.size)
        pick = rng.choice(idx_all, size=take, replace=False)

        # (H*W, C) にしてから pick
        X_flat = X.reshape(C, -1).T  # (H*W, C)
        Xs = X_flat[pick, :]         # (take, C)

        # NaN/inf 除去（安全策）
        finite = np.isfinite(Xs).all(axis=1)
        Xs = Xs[finite]
        if Xs.shape[0] == 0:
            continue

        cur_min = Xs.min(axis=0)
        cur_max = Xs.max(axis=0)

        if mins is None:
            mins = cur_min
            maxs = cur_max
        else:
            mins = np.minimum(mins, cur_min)
            maxs = np.maximum(maxs, cur_max)

        seen += Xs.shape[0]

    if mins is None:
        raise RuntimeError("Failed to compute minmax: no valid pixels found in train set.")

    # 念のため max==min を避ける（scale時のゼロ割れ防止）
    maxs = np.where(maxs - mins < eps, mins + eps, maxs)

    return mins.astype(np.float32), maxs.astype(np.float32)

def compute_minmax_over_train(train_files, mode: str, max_pix=200_000):
    """
    train_files から入力Xのmin/maxを推定する統一窓口。
    ただし、あなたの設計では「S2のみmin-max」「EMBは無変換」なので、
    実際に必要なのは S2(先頭10バンド)の xmin/xmax。
    """
    # 既に作ってある関数を使う（S2+EMB / S2 / EMB 全対応）
    return compute_minmax_over_train_s2only(train_files, mode=mode, max_pix=max_pix)


def minmax_scale_X(X, xmin, xmax, eps=1e-6):
    return (X - xmin) / (xmax - xmin + eps)

def helmert_submatrix(K: int) -> np.ndarray:
    V = np.zeros((K, K-1), dtype=np.float64)
    for i in range(1, K):
        e = np.ones(i) / i
        v = np.concatenate([e, [-1], np.zeros(K - i - 1)])
        V[:, i-1] = v / np.linalg.norm(v)
    return V.astype(np.float32)

def clr_numpy(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = np.clip(x, eps, None)
    lx = np.log(x)
    return lx - lx.mean(axis=1, keepdims=True)

def ilr_numpy(x: np.ndarray, V: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return clr_numpy(x, eps=eps) @ V

def aitchison_mean_loss_numpy(pred: np.ndarray, true: np.ndarray,
                              V: np.ndarray, eps: float = 1e-6) -> float:
    zp = ilr_numpy(pred, V, eps=eps)
    zt = ilr_numpy(true, V, eps=eps)
    d2 = ((zp - zt) ** 2).sum(axis=1)
    return float(d2.mean())

def safe_simplex_from_logits(logits: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = torch.softmax(logits, dim=1)
    p = torch.clamp(p, eps, 1.0)
    p = p / p.sum(dim=1, keepdim=True).clamp_min(eps)
    return p

def mae_loss_torch(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # pred, target: (B,K) on simplex
    return torch.mean(torch.abs(pred - target))

def mixed_aitchison_mae_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    V_t: torch.Tensor,
    alpha: float = 0.7,          # Aitchisonの割合
    use_huber: bool = True,
    eps_ait: float = 1e-4,
    huber_delta: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    loss = alpha * Aitchison + (1-alpha) * MAE
    return: (loss_total, loss_ait, loss_mae)
    """
    alpha = float(alpha)
    alpha = max(0.0, min(1.0, alpha))

    if use_huber:
        lait = aitchison_huber_loss_torch(pred, target, V_t, delta=huber_delta, eps=eps_ait)
    else:
        lait = aitchison_loss_torch(pred, target, V_t, eps=eps_ait)

    lmae = mae_loss_torch(pred, target)
    ltot = alpha * lait + (1.0 - alpha) * lmae
    return ltot, lait.detach(), lmae.detach()


# ========= 2DパッチDataset =========
class AbundancePatch2DScaled(Dataset):
    """
    file_list: 入力特徴量 + OEMラベルを含むGeoTIFFのパス配列
    xmin, xmax: 入力特徴量のmin-max（学習セットから推定したもの）
    patch_size: Kは奇数（例: 5）
    center_stride: 中心画素のサンプリング間隔
    """
    def __init__(self, file_list, xmin, xmax, mode: str,
                 patch_size=7, center_stride=3,
                 ignore_nan=True, clamp_bounds=True, for_eval=False):
        assert patch_size % 2 == 1, "patch_sizeは奇数にしてください"
        self.ps   = patch_size
        self.half = patch_size // 2
        self.items = []
        self.for_eval = for_eval
        self.file_idx = []
        self.mode = mode

        xmin = np.asarray(xmin, dtype=np.float32)  # (1,64)
        xmax = np.asarray(xmax, dtype=np.float32)

        for fi, tif in enumerate(file_list):
            X4, Y8, _ = load_XY_tif(tif)  # (64,H,W), (8,H,W)
            C, H, W = X4.shape

            # --- スケーリング（S2のみmin-max、Embは無変換） ---
            Xs = scale_input_X(X4, xmin, xmax, mode=self.mode, eps=1e-6)


            # --- valid mask ---
            Ym = np.moveaxis(Y8, 0, -1)  # (H,W,8)
            valid_y = np.isfinite(Ym).all(axis=-1)
            if ignore_nan:
                valid_y &= (Ym >= -1e-6).all(axis=-1) & (Ym <= 1 + 1e-6).all(axis=-1)

            Xm = np.moveaxis(Xs, 0, -1)  # (H,W,64)
            valid_x = np.isfinite(Xm).all(axis=-1)

            valid = valid_x & valid_y

            # --- パディング ---
            Xp = np.pad(
                Xs,
                ((0, 0), (self.half, self.half), (self.half, self.half)),
                mode='edge'
            )
            Yp = np.pad(
                Y8,
                ((0, 0), (self.half, self.half), (self.half, self.half)),
                mode='edge'
            )
            valid_p = np.pad(
                valid,
                ((self.half, self.half), (self.half, self.half)),
                mode='constant',
                constant_values=False
            )

            # --- パッチ生成 ---
            for iy in range(self.half, H + self.half, center_stride):
                for ix in range(self.half, W + self.half, center_stride):
                    if not valid_p[iy, ix]:
                        continue

                    xpatch = Xp[:, iy-self.half:iy+self.half+1,
                                   ix-self.half:ix+self.half+1]  # (64,K,K)
                    if not np.isfinite(xpatch).all():
                        continue

                    ycenter = Yp[:, iy, ix]  # (8,)

                    if clamp_bounds:
                        ycenter = np.clip(ycenter, 0.0, 1.0)
                        s = float(ycenter.sum())
                        if not (0.999 <= s <= 1.001):
                            ycenter = ycenter / (s + 1e-12)
                        """
                        # 追加：混ざり具合による間引き
                        max_frac = float(ycenter.max())

                        if max_frac > 0.9:
                            if random.random() < 0.8:
                                continue
                        elif max_frac > 0.7:
                            if random.random() < 0.5:
                                continue
                        """
                    self.items.append(
                        (xpatch.astype(np.float32),   # (C,K,K)
                        ycenter.astype(np.float32))
                    )
                    if self.for_eval:
                        self.file_idx.append(fi)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        x, y = self.items[idx]
        if self.for_eval:
            return torch.from_numpy(x), torch.from_numpy(y), self.file_idx[idx]
        return torch.from_numpy(x), torch.from_numpy(y)

# ========= 2D-CNNモデル（Res なし） =========
class SpectralSpatial2DCNN(nn.Module):
    def __init__(self, in_ch, n_classes=8, base=32, dropout=0.20):
        super().__init__()

        self.pad = nn.ReplicationPad2d(1)

        self.conv1 = nn.Conv2d(in_ch,  base,   kernel_size=3, padding=0, bias=False)
        self.gn1   = nn.GroupNorm(num_groups=8, num_channels=base)

        self.conv2 = nn.Conv2d(base,   base,   kernel_size=3, padding=0, bias=False)
        self.gn2   = nn.GroupNorm(num_groups=8, num_channels=base)

        self.conv3 = nn.Conv2d(base,   base*2, kernel_size=3, padding=0, bias=False)
        self.gn3   = nn.GroupNorm(num_groups=8, num_channels=base*2)

        self.conv4 = nn.Conv2d(base*2, base*2, kernel_size=3, padding=0, bias=False)
        self.gn4   = nn.GroupNorm(num_groups=8, num_channels=base*2)

        self.fc1 = nn.Linear(base*2, 384)
        self.fc2 = nn.Linear(384, n_classes)
        self.dropout = nn.Dropout(dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: (B,C,H,W)
        x = self.pad(x); x = F.relu(self.gn1(self.conv1(x)), inplace=True)
        x = self.pad(x); x = F.relu(self.gn2(self.conv2(x)), inplace=True)
        x = self.pad(x); x = F.relu(self.gn3(self.conv3(x)), inplace=True)
        x = self.pad(x); x = F.relu(self.gn4(self.conv4(x)), inplace=True)

        # 中心ピクセルだけ抜く（H,W の中心）
        h = x.shape[2] // 2
        w = x.shape[3] // 2
        x = x[:, :, h, w]  # (B, 2*base)

        x = self.dropout(F.relu(self.fc1(x), inplace=True))
        return self.fc2(x)


# ========= Aitchison(ILR)損失 =========
def aitchison_loss_torch(pred, target, V_t, eps=1e-6):
    pred_l = torch.log(torch.clamp(pred,  min=eps))
    pred_c = pred_l - pred_l.mean(dim=1, keepdim=True)
    targ_l = torch.log(torch.clamp(target, min=eps))
    targ_c = targ_l - targ_l.mean(dim=1, keepdim=True)
    pi = pred_c @ V_t
    ti = targ_c @ V_t
    d2 = torch.sum((pi - ti) ** 2, dim=1)
    return torch.mean(d2)

def aitchison_huber_loss_torch(pred, target, V_t, delta=0.5, eps=1e-4):# ← 1e-6 から 1e-4
    pred_l = torch.log(torch.clamp(pred,  min=eps))
    pred_c = pred_l - pred_l.mean(dim=1, keepdim=True)
    targ_l = torch.log(torch.clamp(target, min=eps))
    targ_c = targ_l - targ_l.mean(dim=1, keepdim=True)
    diff = (pred_c @ V_t) - (targ_c @ V_t)
    absd = torch.abs(diff)
    quad = 0.5 * (diff ** 2)
    lin  = delta * (absd - 0.5 * delta)
    huber = torch.where(absd <= delta, quad, lin)
    return torch.mean(torch.sum(huber, dim=1))

#Loss関数を作る
def loss_train(pred, target, V_t,
               use_huber: bool = True, #hurber版を使用する
               eps: float = 1e-4): #少し大きめ

    if use_huber:
        return aitchison_huber_loss_torch(
            pred, target, V_t, delta=0.5, eps=eps)
    else:
        return aitchison_loss_torch(
            pred, target, V_t, eps=eps
        )


# ========= 学習（2D） =========
def train_with_validation_2d(train_ds, val_ds, V,
                             epochs, bs, lr, wd, device,
                             in_ch):

    model = SpectralSpatial2DCNN(
        in_ch=in_ch, n_classes=N_CLASSES, base=BASE_CH, dropout=DROPOUT
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    V_t = torch.from_numpy(V).to(device)

    tr_dl = DataLoader(train_ds, batch_size=bs, shuffle=True,
                       num_workers=4, pin_memory=True)
    va_dl = DataLoader(val_ds, batch_size=bs, shuffle=False,
                       num_workers=4, pin_memory=True)

    steps_per_epoch = max(1, len(tr_dl))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, steps_per_epoch=steps_per_epoch, epochs=epochs,
        pct_start=0.1, div_factor=10.0, final_div_factor=10.0
    )

    best_val = float("inf")
    best_ep  = 0
    no_imp   = 0
    best_state = None

    # --- sanity check ---
    xb0, yb0 = next(iter(tr_dl))
    xb0 = xb0.to(device)
    yb0 = yb0.to(device)
    with torch.no_grad():
        lg0 = model(xb0)
        lg0 = torch.clamp(lg0, -30.0, 30.0)
        pr0 = safe_simplex_from_logits(lg0, 1e-6)
        l0  = aitchison_loss_torch(pr0, yb0, V_t)
    print("[Sanity] x_finite:", torch.isfinite(xb0).all().item(),
          "logits_finite:", torch.isfinite(lg0).all().item(),
          "pred_finite:", torch.isfinite(pr0).all().item(),
          "loss0:", float(l0))
    # --------------------

    for ep in range(1, epochs + 1):
        # ---- Train ----
        model.train()
        tr_sum = 0.0
        tr_a_sum = 0.0
        tr_m_sum = 0.0
        n_b = 0

        for xb, yb in tr_dl:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)

            logits = model(xb)
            logits = torch.clamp(logits, -30.0, 30.0)
            pred   = safe_simplex_from_logits(logits, 1e-6)

            # ★ trainは複合損失
            loss, la, lm = mixed_aitchison_mae_loss(
                pred, yb, V_t,
                alpha=ALPHA_AITCHISON,
                use_huber=USE_HUBER_AITCH,
                eps_ait=1e-4,
                huber_delta=HUBER_DELTA,
            )

            if not torch.isfinite(loss):
                print("[SkipBatch] non-finite loss detected",
                      "pred_minmax=", (pred.min().item(), pred.max().item()),
                      "logits_minmax=", (logits.min().item(), logits.max().item()))
                print(" has_nan(pred)=", torch.isnan(pred).any().item(),
                      " has_nan(y)=", torch.isnan(yb).any().item())
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # 勾配NaNチェック
            bad_grad = False
            for p in model.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    bad_grad = True
                    break
            if bad_grad:
                print("[SkipBatch] non-finite grad detected")
                opt.zero_grad(set_to_none=True)
                continue

            opt.step()

            # パラメータNaNチェック
            param_nan = False
            for p in model.parameters():
                if not torch.isfinite(p.data).all():
                    param_nan = True
                    break
            if param_nan:
                print("[ParamNaN] model weights became non-finite → reducing LR and skipping step")
                for g in opt.param_groups:
                    g["lr"] = max(g["lr"] * 0.2, 1e-6)
                continue

            scheduler.step()

            tr_sum   += loss.item()
            tr_a_sum += float(la)
            tr_m_sum += float(lm)
            n_b      += 1

        tr_loss = tr_sum / max(1, n_b)
        tr_a    = tr_a_sum / max(1, n_b)
        tr_m    = tr_m_sum / max(1, n_b)

        # ---- Val ----（★ ここはAitchisonだけで監視する例）
        model.eval()
        va_sum = 0.0
        n_b = 0
        with torch.no_grad():
            for xb, yb in va_dl:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)

                logits = model(xb)
                logits = torch.clamp(logits, -30.0, 30.0)
                pr     = safe_simplex_from_logits(logits, 1e-6)

                l = aitchison_loss_torch(pr, yb, V_t)  # ★valはAitchisonのみ
                va_sum += l.item()
                n_b    += 1

        va_loss = va_sum / max(1, n_b)

        print(f"[Epoch {ep:03d}] train={tr_loss:.3f} (ait={tr_a:.3f}, mae={tr_m:.3f})  "
              f"val_ait={va_loss:.3f}")

        # EarlyStop判定（Aitchisonで）
        if va_loss + 1e-9 < best_val:
            best_val = va_loss
            best_ep  = ep
            no_imp   = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_imp += 1
            if no_imp >= EARLY_STOP_PATIENCE:
                print(f"[EarlyStop] no improvement for {EARLY_STOP_PATIENCE} epochs. "
                      f"Best@{best_ep} val={best_val:.3f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val, best_ep


# ========= 評価 & 予測TIF保存 =========
def predict_single_image_metric_and_save_2d(tif_path, model, xmin, xmax, mode: str, V, eps, delta,
                                           patch=PATCH_SIZE, stride=CENTER_STRIDE_VAL, out_tif_path=None):
    X4, Y8, meta = load_XY_tif(tif_path)
    C, H, W = X4.shape
    vm = valid_mask(X4, Y8)
    if not vm.any():
        return os.path.basename(tif_path), np.nan, 0.0, 0

    Xs = scale_input_X(X4, xmin, xmax, mode=mode, eps=1e-6)

    half = patch // 2
    Xp = np.pad(Xs, ((0, 0), (half, half), (half, half)), mode="reflect")
    Yp = np.pad(Y8, ((0, 0), (half, half), (half, half)), mode="reflect")

    outs = []
    gts  = []
    pred_sum = np.zeros((N_CLASSES, H, W), dtype=np.float32) if out_tif_path else None
    pred_count = np.zeros((H, W), dtype=np.int32) if out_tif_path else None

    model.eval()
    with torch.no_grad():
        for iy in range(half, H + half, stride):
            for ix in range(half, W + half, stride):
                if not vm[iy-half, ix-half]:
                    continue

                xpatch = Xp[:, iy-half:iy+half+1,
                               ix-half:ix+half+1][None, ...]
                ycenter = Yp[:, iy, ix][None, ...]
                xb = torch.from_numpy(xpatch).to(DEVICE)

                logits = model(xb)
                pr = safe_simplex_from_logits(logits, 1e-6).cpu().numpy()
                outs.append(pr[0])
                gts.append(ycenter[0])

                if out_tif_path:
                    y0 = iy - half
                    x0 = ix - half
                    pred_sum[:, y0, x0] += pr[0]
                    pred_count[y0, x0]  += 1

    if len(outs) == 0:
        return os.path.basename(tif_path), np.nan, 0.0, 0

    outs_np = np.asarray(outs, np.float32)
    gts_np  = np.asarray(gts,  np.float32)

    Ymr = multiplicative_replacement(gts_np, delta=delta)
    s = Ymr.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    Ymr = Ymr / s

    zp = ilr_numpy(outs_np, np.asarray(V, dtype=np.float32), eps=eps)
    zt = ilr_numpy(Ymr, np.asarray(V, dtype=np.float32), eps=eps)
    d2 = ((zp - zt) ** 2).sum(axis=1)

    mean_ait = float(d2.mean())
    d2_sum = float(d2.sum())
    count = int(d2.size)

    rmse_per_class = np.sqrt(np.mean((outs_np - Ymr) ** 2, axis=0))
    mrmse = float(np.mean(rmse_per_class))

    mae = float(np.mean(np.abs(outs_np - Ymr)))

    if out_tif_path:
        pred_map = np.full((N_CLASSES, H, W), np.nan, dtype=np.float32)
        mask = pred_count > 0
        for k in range(N_CLASSES):
            tmp = pred_map[k]
            tmp[mask] = pred_sum[k][mask] / pred_count[mask]

        out_meta = meta.copy()
        out_meta.update({
            "count": N_CLASSES,
            "dtype": "float32",
            "nodata": np.nan,
        })

        os.makedirs(os.path.dirname(out_tif_path), exist_ok=True)
        with rasterio.open(out_tif_path, "w", **out_meta) as dst:
            dst.write(pred_map)
        print(f"[Predicted] {os.path.basename(tif_path)} -> {out_tif_path}")

    return os.path.basename(tif_path), mean_ait, mrmse, mae, d2_sum, count


def predict_images_metric_and_save_2d(files, model, xmin, xmax, mode: str, V, eps, delta,
                                     patch=PATCH_SIZE, stride=CENTER_STRIDE_VAL, preds_dir=None):
    if preds_dir is not None:
        os.makedirs(preds_dir, exist_ok=True)

    rows = []
    total_d2_sum = 0.0
    total_count = 0
    mrmse_values = []
    mae_values = []

    for p in files:
        stem = os.path.splitext(os.path.basename(p))[0]
        out_tif_path = os.path.join(preds_dir, f"{stem}_pred.tif") if preds_dir else None

        image_name, mean_ait, mrmse, mae, d2_sum, count = predict_single_image_metric_and_save_2d(
            p, model, xmin, xmax, mode, V, eps, delta,
            patch=patch, stride=stride, out_tif_path=out_tif_path
        )

        rows.append({
            "image": image_name, 
            "mean_aitchison_ilr": mean_ait,
            "mrmse": mrmse,
            "mae": mae,
            })
        total_d2_sum += d2_sum
        total_count += count
        mrmse_values.append(mrmse)
        mae_values.append(mae)
    global_mean_ait = float(total_d2_sum / total_count) if total_count > 0 else np.nan
    global_mrmse = float(np.mean(mrmse_values)) if len(mrmse_values) > 0 else np.nan
    global_mae = float(np.mean(mae_values)) if len(mae_values) > 0 else np.nan
    return rows, global_mean_ait, global_mrmse, global_mae

# ========= Main =========
def main():
    set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    run_dir = os.path.join(OUTPUT_DIR, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)

    files = list_tifs(INPUT_DIR)
    if len(files) == 0:
        raise RuntimeError("入力ディレクトリにTIFがありません。")

    IN_BANDS = check_inbands_consistency(files)
    MODE = infer_dataset_mode(IN_BANDS)
    print(f"[Data] MODE={MODE}, IN_BANDS={IN_BANDS}, TOTAL={IN_BANDS + N_CLASSES}")
    
    # ---- split ----
    FORCE_TEST = {
        "rotterdam_42_combined.tif",
        "tokyo_3_combined.tif",
        "daressalaam_71_combined.tif",
        "chicago_24_combined.tif",
    }

    force_test_files = [f for f in files if os.path.basename(f) in FORCE_TEST]
    other_files      = [f for f in files if os.path.basename(f) not in FORCE_TEST]

    rng = np.random.default_rng(SEED)
    idx = np.arange(len(other_files))
    rng.shuffle(idx)

    n      = len(other_files)
    n_train= int(round(n * TRAIN_RATIO))
    n_val  = int(round(n * VAL_RATIO))

    train_files = [other_files[i] for i in idx[:n_train]]
    val_files   = [other_files[i] for i in idx[n_train:n_train+n_val]]
    test_files  = [other_files[i] for i in idx[n_train+n_val:]]
    test_files  = force_test_files + test_files

    print("[Split]")
    print("  train:", len(train_files))
    print("   val :", len(val_files))
    print("  test:", len(test_files), "(forced:", len(force_test_files), ")")

    # Min-max（入力特徴量）
    xmin, xmax = compute_minmax_over_train(train_files, mode=MODE, max_pix=200_000)

    scaler = {"xmin": xmin.tolist(), "xmax": xmax.tolist(), "eps": EPS_LOG}

    # Datasets（2Dパッチ）
    train_ds = AbundancePatch2DScaled(
        train_files, xmin, xmax, mode=MODE,
        patch_size=PATCH_SIZE, center_stride=CENTER_STRIDE_TRAIN
    )
    val_ds   = AbundancePatch2DScaled(
        val_files, xmin, xmax, mode=MODE,
        patch_size=PATCH_SIZE, center_stride=CENTER_STRIDE_VAL
    )
    print(f"[Patch] train_samples={len(train_ds)}  val_samples={len(val_ds)}")

    # ILR基底
    V = helmert_submatrix(N_CLASSES)

    # Train
    model, best_val, best_ep = train_with_validation_2d(
        train_ds, val_ds, V,
        epochs=EPOCHS, bs=BATCH_SIZE_2D, lr=LR, wd=WEIGHT_DECAY,
        device=DEVICE, in_ch=IN_BANDS
    )

    print(f"[Best] epoch={best_ep}  val_aitchison={best_val:.6f}")

    # Val 評価（平均Aitchison）
    rows_val,  val_global_mean, val_global_mrmse, val_global_mae = predict_images_metric_and_save_2d(
        val_files,  model, xmin, xmax, MODE, V, EPS_LOG, DELTA_ZERO,
        patch=PATCH_SIZE, stride=CENTER_STRIDE_VAL, preds_dir=None
    )

    # Test 評価 + 予測TIF保存（1回の推論で両方）
    preds_dir = os.path.join(run_dir, "preds")
    os.makedirs(preds_dir, exist_ok=True)
    rows_test, test_global_mean, test_global_mrmse, test_global_mae = predict_images_metric_and_save_2d(
        test_files, model, xmin, xmax, MODE, V, EPS_LOG, DELTA_ZERO,
        patch=PATCH_SIZE, stride=CENTER_STRIDE_TEST, preds_dir=preds_dir
    )

    print(f"[Val ] global_mean_aitchison_ilr  = {val_global_mean:.6f}  global_mrmse = {val_global_mrmse:.6f}  global_mae = {val_global_mae:.6f}")
    print(f"[Test] global_mean_aitchison_ilr = {test_global_mean:.6f}  global_mrmse = {test_global_mrmse:.6f}  global_mae = {test_global_mae:.6f}")
    print("[Pred] all test prediction TIFs saved.")

    # 評価系だけ欲しければ、ここだけ残して他の保存は削ってOK
    pd.DataFrame(rows_val).to_csv(
        os.path.join(run_dir, "metrics_val_all.csv"),
        index=False
    )
    
    pd.DataFrame(rows_test).to_csv(
        os.path.join(run_dir, "metrics_test_all.csv"),
        index=False
    )
    with open(os.path.join(run_dir, "summary_val.json"), "w") as f:
        json.dump({
            "global_mean_aitchison_ilr": val_global_mean,
            "global_mrmse": val_global_mrmse,
            "global_mae": val_global_mae
        }, f, indent=2)
    with open(os.path.join(run_dir, "summary_test.json"), "w") as f: 
        json.dump({
            "global_mean_aitchison_ilr": test_global_mean,
            "global_mrmse": test_global_mrmse,
            "global_mae": test_global_mae
        }, f, indent=2)

    print("[Saved] metrics_val_all.csv / metrics_test_all.csv")

    # --- 設定・モデル保存 ---
    with open(os.path.join(run_dir,"config.json"),"w",encoding="utf-8") as f:
        json.dump({
            "input_dir":INPUT_DIR,"output_dir":run_dir,"seed":SEED,
            "train_ratio":TRAIN_RATIO,"batch_size":BATCH_SIZE_2D,
            "epochs":EPOCHS,"lr":LR,"weight_decay":WEIGHT_DECAY,"dropout":DROPOUT,
            "base_ch":BASE_CH,"patch_size":PATCH_SIZE,
            "center_stride_train":CENTER_STRIDE_TRAIN,"center_stride_val":CENTER_STRIDE_VAL,"center_stride_test":CENTER_STRIDE_TEST,
            "delta_zero":DELTA_ZERO,"eps_log":EPS_LOG,
            "device":str(DEVICE),"loss": "mixed_aitchison_mae",
            "early_stop_patience":EARLY_STOP_PATIENCE,
            "alpha_aitchison":ALPHA_AITCHISON,"use_huber_aitch":USE_HUBER_AITCH,"huber_delta":HUBER_DELTA,
            "mode":MODE
        }, f, ensure_ascii=False, indent=2)

    with open(os.path.join(run_dir,"scaler.json"),"w",encoding="utf-8") as f:
        json.dump(scaler,f,indent=2)

    torch.save(model.state_dict(), os.path.join(run_dir,"model.pt"))
    print(f"[Done] outputs: {run_dir}")

if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    main()

