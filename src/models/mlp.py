# -*- coding: utf-8 -*-
import os, json, random, glob
from datetime import datetime
import numpy as np, pandas as pd, rasterio
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# ========= 設定 =========
INPUT_DIR   = "/workspace/data/combined_tif"
OUTPUT_DIR  = "/workspace/outputs/mlp/embedding"
SEED        = 42
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.10
TEST_RATIO  = 0.20
assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-8, "Split ratios must sum to 1.0"

# 学習ハイパラ
BATCH_SIZE  = 8192
EPOCHS      = 200
LR          = 1e-3
WEIGHT_DECAY= 1e-4           # L2正則化
HIDDEN1     = 384
HIDDEN2     = 384
HIDDEN3     = 384
DROPOUT     = 0.20


# 早期終了 & LRスケジューラ
SCHEDULER_PATIENCE  = 10     # ReduceLROnPlateau の patience
EARLY_STOP_PATIENCE = 20     # 改善なしエポックで停止

# 複合損失
ALPHA_AITCHISON = 0.0   # Aitchisonの割合（混合lossにしたいなら）
USE_HUBER_AITCH = True  # Huber版Aitchisonを使用
HUBER_DELTA     = 0.5   # Huberのデルタパラメータ

# データ仕様
S2_BANDS    = 10
EMB_BANDS   = 64
N_CLASSES   = 8
OEM_BANDS   = N_CLASSES  # OEMラベルバンド数
IN_BANDS    = S2_BANDS   # デフォルトはS2のみ
DELTA_ZERO  = 1e-5           # ゼロ置換 δ=1e-4を上げた
EPS_LOG     = 1e-6           # logクリップ ε=1e-5から上げた

# サンプリング上限（各画像あたり）
MAX_PIX_PER_IMG_TRAIN = 300_000   
MAX_PIX_PER_IMG_VAL   = 120_000  

DEVICE      = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# ========================

def set_seed(s=42):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

# NaNが含まれるため除く
SKIP = {
    "kyoto_54_combined.tif",
    "dhaka_39_combined.tif",
    "santiago_18_combined.tif",
    "santiago_5_combined.tif",
    "soriano_4_combined.tif",
    "soriano_7_combined.tif",
}

# .tif を列挙
def list_tifs(d):
    files=[]
    for e in ["*.tif","*.tiff"]:
        files+=glob.glob(os.path.join(d,e))
    return [f for f in sorted(files) if os.path.basename(f) not in SKIP]

# データ読み込み → (input_features, labels, meta)
def load_combined_tif(path):
    with rasterio.open(path) as src:
        arr=src.read().astype(np.float32); meta=src.meta.copy()
    total_bands = arr.shape[0]
    if total_bands <= N_CLASSES:
        raise ValueError(f"{os.path.basename(path)}: bands {total_bands} <= N_CLASSES({N_CLASSES})")
    in_bands = total_bands - N_CLASSES
    return arr[:in_bands], arr[in_bands:in_bands+N_CLASSES], meta

# NaN/Inf 除外マスク
def valid_mask(input_features, labels):
    return np.isfinite(input_features).all(axis=0) & np.isfinite(labels).all(axis=0)

# ゼロ対策（multiplicative replacement）
def multiplicative_replacement(Y, delta=1e-5):
    K=Y.shape[1]; Y=Y.copy()
    zeros=(Y<=0); zc=zeros.sum(axis=1,keepdims=True)
    Y[zeros]=delta
    s=Y.sum(axis=1,keepdims=True); s[s==0]=1.0
    Y=Y/s
    nonzero_mask = ~zeros
    nonzero_sum  = (Y*nonzero_mask).sum(axis=1,keepdims=True); nonzero_sum[nonzero_sum==0]=1.0
    scale=(1.0-delta*zc); scale=np.clip(scale,1e-8,None)
    Y[nonzero_mask]= (Y[nonzero_mask]/nonzero_sum.repeat(K,axis=1)[nonzero_mask])*scale.repeat(K,axis=1)[nonzero_mask]
    return Y.astype(np.float32)

# 対数変換
def to_log_domain(X: np.ndarray) -> np.ndarray:
    X = np.clip(X, 0.0, None)
    return np.log1p(X).astype(np.float32)

# 学習セットから min/max を推定（対数空間）
def infer_inbands_from_file(path: str) -> int:
    with rasterio.open(path) as src:
        total_bands = src.count
    if total_bands <= N_CLASSES:
        raise ValueError(f"{os.path.basename(path)}: bands {total_bands} <= N_CLASSES({N_CLASSES})")
    return total_bands - N_CLASSES


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


def check_inbands_consistency(files) -> int:
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


def compute_minmax_over_train(train_files, mode: str, max_pix=200_000):
    in_bands = infer_inbands_from_file(train_files[0])
    if mode == "EMB":
        return np.zeros((1, in_bands), dtype=np.float32), np.ones((1, in_bands), dtype=np.float32)

    mins, maxs = [], []
    for p in train_files:
        input_features, labels, _ = load_combined_tif(p)
        vm = valid_mask(input_features, labels)
        if not vm.any():
            continue
        if mode == "S2+EMB":
            X = input_features[:S2_BANDS, vm].T
        else:
            X = input_features[:, vm].T
        if X.shape[0] > max_pix:
            idx = np.random.choice(X.shape[0], size=max_pix, replace=False)
            X = X[idx]
        X_log = to_log_domain(X)
        mins.append(X_log.min(axis=0, keepdims=True))
        maxs.append(X_log.max(axis=0, keepdims=True))

    if not mins:
        raise RuntimeError("Valid pixels not found for min-max computation.")

    xmin = np.vstack(mins).min(axis=0, keepdims=True)
    xmax = np.vstack(maxs).max(axis=0, keepdims=True)

    if mode == "S2+EMB":
        full_xmin = np.zeros((1, in_bands), dtype=np.float32)
        full_xmax = np.ones((1, in_bands), dtype=np.float32)
        full_xmin[:, :S2_BANDS] = xmin
        full_xmax[:, :S2_BANDS] = xmax
        return full_xmin, full_xmax

    return xmin.astype(np.float32), xmax.astype(np.float32)


# スケーリング（対数空間 min-max）
def minmax_scale_X(X, xmin, xmax, mode: str = "S2", eps=1e-6):
    if mode == "EMB":
        return X.astype(np.float32)

    if mode == "S2":
        X_log = to_log_domain(X)
        return (X_log - xmin) / (xmax - xmin + eps)

    # mode == "S2+EMB"
    Xs = X.astype(np.float32).copy()
    Xs2 = to_log_domain(X[:, :S2_BANDS])
    Xs[:, :S2_BANDS] = (Xs2 - xmin[:, :S2_BANDS]) / (xmax[:, :S2_BANDS] + eps - xmin[:, :S2_BANDS])
    return Xs
    
# Helmert 部分行列
def helmert_submatrix(K: int) -> np.ndarray:
    V = np.zeros((K, K-1), dtype=np.float64)
    for i in range(1, K):
        e = np.ones(i) / i
        v = np.concatenate([e, [-1], np.zeros(K - i - 1)])
        V[:, i-1] = v / np.linalg.norm(v)
    return V.astype(np.float32)

# clr, ilr, Aitchison距離
def clr_numpy(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = np.clip(x, eps, None) # 最小値をeps1e-6に置き換え
    lx = np.log(x) # 成分ごとに対数を取る
    return lx - lx.mean(axis=1, keepdims=True) # 平均を引く→logの中心化

def ilr_numpy(x: np.ndarray, V: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return clr_numpy(x, eps=eps) @ V # 直行座標へ写像する　 @は行列積

def aitchison_mean_loss_numpy(pred: np.ndarray, true: np.ndarray, V: np.ndarray, eps: float = 1e-6) -> float:
    zp = ilr_numpy(pred, V, eps=eps)
    zt = ilr_numpy(true, V, eps=eps)
    d2 = ((zp - zt) ** 2).sum(axis=1)
    return float(d2.mean())

#Huber版のAitchison損失
def aitchison_huber_loss_torch(pred, target, V_t, delta=1.0, eps=1e-6):
    pred_l = torch.log(torch.clamp(pred, min=eps))
    targ_l = torch.log(torch.clamp(target, min=eps))
    pred_c = pred_l - pred_l.mean(dim=1, keepdim=True)
    targ_c = targ_l - targ_l.mean(dim=1, keepdim=True)
    diff = (pred_c @ V_t) - (targ_c @ V_t)  # (N, K-1)

    absd = torch.abs(diff)
    quad = 0.5 * (diff ** 2) 
    lin  = delta * (absd - 0.5 * delta)
    huber = torch.where(absd <= delta, quad, lin)  # 要素ごと
    return torch.mean(torch.sum(huber, dim=1))

def mae_loss_torch(pred, target):
    # pred, target: (B,K) on simplex
    return torch.mean(torch.abs(pred - target))

def mixed_aitchison_mae_loss(
    pred,
    target,
    V_t,
    alpha=0.7,          # Aitchisonの割合
    use_huber=True,
    eps_ait=1e-4,
    huber_delta=0.5,
):
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

# ---- 学習/検証ピクセルの配列化 ----
def build_arrays(files, xmin, xmax, max_pix_per_img, delta, eps, mode: str):
    Xs_all=[]; Ys_all=[]
    for p in files:
        input_features, labels, _=load_combined_tif(p); vm=valid_mask(input_features, labels)
        if not vm.any(): continue
        X=input_features[:,vm].T; Y=labels[:,vm].T
        if X.shape[0]>max_pix_per_img:
            idx=np.random.choice(X.shape[0],size=max_pix_per_img,replace=False); X=X[idx]; Y=Y[idx]
        Xs=minmax_scale_X(X,xmin,xmax,mode=mode,eps=eps)
        Ymr=multiplicative_replacement(Y,delta=delta)
        s=Ymr.sum(axis=1,keepdims=True); s[s==0]=1.0; Ymr=Ymr/s
        Xs_all.append(Xs.astype(np.float32)); Ys_all.append(Ymr.astype(np.float32))
    if not Xs_all: raise RuntimeError("有効ピクセルが見つかりません。")
    return np.vstack(Xs_all), np.vstack(Ys_all)

# ---- モデル ----
class PixelMLP(nn.Module):
    def __init__(self,in_bands=IN_BANDS,num_classes=8,h1=256,h2=256,h3=256,drop=0.2):
        super().__init__()
        self.fc1=nn.Linear(in_bands,h1)
        self.fc2=nn.Linear(h1,h2)
        self.fc3=nn.Linear(h2,h3)
        self.out=nn.Linear(h3,num_classes)
        self.drop=nn.Dropout(drop)
    def forward(self,x): #実際の流れ
        x=F.relu(self.fc1(x)); x=self.drop(x)
        x=F.relu(self.fc2(x)); x=self.drop(x)
        x=F.relu(self.fc3(x)); x=self.drop(x)
        x=self.out(x)
        return F.softmax(x,dim=1)

def aitchison_loss_torch(pred,target,V_t,eps=1e-6):
    pred_l=torch.log(torch.clamp(pred,min=eps)); pred_c=pred_l - pred_l.mean(dim=1,keepdim=True)
    targ_l=torch.log(torch.clamp(target,min=eps)); targ_c=targ_l - targ_l.mean(dim=1,keepdim=True)
    pi=pred_c @ V_t; ti=targ_c @ V_t
    d2=torch.sum((pi-ti)**2,dim=1)
    return torch.mean(d2)

# ---- 学習ループ ----
def train_with_validation(train_X,train_Y,val_X,val_Y,V,epochs,bs,lr,wd,device,eps):
    in_bands = train_X.shape[1]
    # mlpモデルを作る
    model=PixelMLP(in_bands,N_CLASSES,HIDDEN1,HIDDEN2,HIDDEN3,DROPOUT).to(device)
    # Adamオプティマイザを作る
    opt=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=wd)# weight_decay=wdはL2正則化(過学習対策)
    V_t=torch.from_numpy(V).to(device)# V_tをPyTorchテンソルに変換してGPUに転送

    # DataLoaderを作る（batchごとに取り出すためのもの）
    tr_dl=DataLoader(TensorDataset(torch.from_numpy(train_X),torch.from_numpy(train_Y)),
                     batch_size=bs,shuffle=True,num_workers=2,pin_memory=True) # shuffle=Trueで毎エポック順番をシャッフル
    va_dl=DataLoader(TensorDataset(torch.from_numpy(val_X),torch.from_numpy(val_Y)),
                     batch_size=bs,shuffle=False,num_workers=2,pin_memory=True) # 検証はシャッフル不要

    steps_per_epoch = max(1, len(tr_dl))  # 1エポックあたりのステップ数（DataLoaderのバッチ数）

    # OneCycleLR スケジューラを作る（学習率をエポックごとに変化させるためのもの）
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=lr,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        pct_start=0.1,          # 10%をウォームアップ
        div_factor=10.0,        # 初期LR = LR/10
        final_div_factor=10.0   # 終盤は LR/10 まで落とす
    )

    # 最良の検証損失とその時のエポック数、改善が見られなかったエポック数、最良の状態を記録
    best_val=float('inf'); # inf(無限大で初期化して、どんな値もこれより小さくなるようにする
    best_ep=0; no_improve=0; best_state=None

    for ep in range(1,epochs+1): # エポックループ（1からEPOCHSまで）
        # --- 学習　---
        model.train() # 学習モードにする（ドロップアウトなどが有効になる） 
        tr_loss=0.0; tr_a_loss=0.0; tr_m_loss=0.0; n_b=0

        for xb,yb in tr_dl: #batchを1つづつdlから取り出す　xbは特徴量、ybはラベル
            xb=xb.to(device,dtype=torch.float32); yb=yb.to(device,dtype=torch.float32) # GPUに転送して型をfloat32にする
            # 予測を行う
            pred=model(xb)
            # trainは複合損失 lossは合計、laはAitchison、lmはMAE
            loss, la, lm = mixed_aitchison_mae_loss( 
                pred, yb, V_t,
                alpha=ALPHA_AITCHISON,
                use_huber=USE_HUBER_AITCH,
                eps_ait=1e-4,
                huber_delta=HUBER_DELTA,
            )

            opt.zero_grad() # greadient(勾配)を初期化
            loss.backward() # 勾配を計算
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)     # ← 勾配クリップを追加
            # 勾配を更新
            opt.step()

            scheduler.step() 
            tr_loss+=loss.item(); tr_a_loss+=float(la); tr_m_loss+=float(lm); n_b+=1 #バッチ数カウント
        
        # バッチ数で割って平均損失を計算(maxで0での割り算回避)
        tr_loss/=max(n_b,1); tr_a_loss/=max(n_b,1); tr_m_loss/=max(n_b,1)


        # --- Validation ---
        model.eval(); va_sum=0.0; n_b=0 # 検証モードにする　va_sumは検証損失の合計、n_bは検証バッチ数

        with torch.no_grad(): # 勾配計算しない
            for xb,yb in va_dl: 
                # バッチを1つづつ取り出す
                xb=xb.to(device,dtype=torch.float32); yb=yb.to(device,dtype=torch.float32)

                # 予測して損失を計算する
                pr=model(xb) 
                l=aitchison_loss_torch(pr,yb,V_t,eps=eps)

                # 検証損失を合計 バッチ数をカウント
                va_sum+=l.item(); n_b+=1
        # バッチ数で割って平均を計算
        if n_b==0:
            va_loss = float('inf') 
        else:
            va_loss = va_sum/n_b

        print(f"[Epoch {ep:03d}] train={tr_loss:.6f} (ait={tr_a_loss:.6f}, mae={tr_m_loss:.6f})  val={va_loss:.6f}  lr={opt.param_groups[0]['lr']:.2e}")

        if va_loss+1e-9 < best_val:
            best_val=va_loss; best_ep=ep; no_improve=0
            best_state={k:v.cpu().clone() for k,v in model.state_dict().items()}
        else:
            no_improve+=1
            if no_improve>=EARLY_STOP_PATIENCE:
                print(f"[EarlyStop] no improvement for {EARLY_STOP_PATIENCE} epochs. Best@{best_ep} val={best_val:.6f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val, best_ep

# ---- 予想 & 評価 ----
def predict_single_image_metric_and_save_mlp(tif_path, model, xmin, xmax, V, eps, delta, device, mode: str, out_tif_path=None):
    input_features, labels, meta = load_combined_tif(tif_path)

    # NaN/Inf除外マスク(vmはvalid_maskの略)
    vm = valid_mask(input_features, labels)
    # 予測対象ピクセルがない場合(vmがFalseの場合はスキップ）（NaNで返す） 
    if not vm.any(): 
        return os.path.basename(tif_path), np.nan, np.nan
    
    #[:,vm]で有効ピクセルのみ抽出して転置（行がピクセル数、列がバンド/クラス）
    X = input_features[:,vm].T 
    Y = labels[:,vm].T # 予測対象ピクセルの正解ラベル

    # min-max正規化（対数空間）
    Xs = minmax_scale_X(X, xmin, xmax, mode=mode, eps=eps)

    N = Xs.shape[0] #対象ピクセル数
    B = 16384 #バッチサイズ
    out = np.empty((N, N_CLASSES), np.float32)

    model.eval() # evalで検証モードにする。(ドロップアウトの挙動が変わる)
    with torch.no_grad(): # 勾配計算をオフにして予測する
        for i in range(0, N, B):
            j = min(i + B, N) # 最後のバッチは残り全部
            # PyTorchテンソルに変換してGPUに転送
            xb = torch.from_numpy(Xs[i:j]).to(device, dtype=torch.float32)
            # 推測し、CPUに戻してNumPy配列に変換して保存
            out[i:j] = model(xb).cpu().numpy()

    Ymr = multiplicative_replacement(Y, delta=delta) # ゼロ置換
    s = Ymr.sum(axis=1, keepdims=True) # 行方向の合計を計算
    s[s == 0] = 1.0; Ymr = Ymr / s # 1に正規化

    # aitchison距離の平均を計算
    mean_ait = aitchison_mean_loss_numpy(out, Ymr, V, eps=eps)

    # mRMSE を計算
    rmse_per_class = np.sqrt(
        np.mean((out - Ymr) ** 2, axis=0)
    )
    mrmse = float(np.mean(rmse_per_class))

    # MAE を計算
    mae = float(np.mean(np.abs(out - Ymr)))

    # GeoTIFF保存（オプション）
    if out_tif_path is not None:
        H, W = vm.shape
        pred_stack = np.zeros((N_CLASSES, H, W), np.float32)
        pred_stack[:, vm] = out.T

        meta_out = meta.copy()
        meta_out.update({
            "count": N_CLASSES,
            "dtype": "float32",
            "compress": "lzw"
        })

        os.makedirs(os.path.dirname(out_tif_path), exist_ok=True)

        with rasterio.open(out_tif_path, "w", **meta_out) as dst:
            dst.write(pred_stack)
            for b in range(N_CLASSES):
                dst.set_band_description(b + 1, f"pred_ratio_class_{b+1}")

        print(f"[Pred] saved: {out_tif_path}")

    # ファイル名と平均Aitchison距離を返す           
    return os.path.basename(tif_path), mean_ait, mrmse, mae

def predict_images_metric_and_save_mlp(files, model, xmin, xmax, V, eps, delta, device, mode: str, preds_dir=None):
    rows = []
    total_d2_sum = 0.0
    total_count = 0
    mrmse_values = []
    mae_values = []

    for p in files:
        base = os.path.basename(p)
        stem, ext = os.path.splitext(base)
        out_tif_path = os.path.join(preds_dir, f"{stem}_pred.tif") if preds_dir else None

        image_name, mean_ait, mrmse, mae = predict_single_image_metric_and_save_mlp(
            p, model, xmin, xmax, V, eps, delta, device, mode, out_tif_path
        )

        rows.append({"image": image_name, "mean_aitchison_ilr": mean_ait, "mrmse": mrmse, "mae": mae})

    if not np.isnan(mrmse):
        mrmse_values.append(mrmse)
    if not np.isnan(mae):
        mae_values.append(mae)

    valid_ait = [
        r["mean_aitchison_ilr"]
        for r in rows
        if not np.isnan(r["mean_aitchison_ilr"])
    ]

    global_mean_aitchison_ilr = (
        float(np.mean(valid_ait))
        if len(valid_ait) > 0 else np.nan
    )

    global_mrmse = (
        float(np.mean(mrmse_values))
        if len(mrmse_values) > 0 else np.nan
    )

    global_mae = (
        float(np.mean(mae_values))
        if len(mae_values) > 0 else np.nan
    )

    return rows, global_mean_aitchison_ilr, global_mrmse, global_mae

# ---- Main ----
def main():
    set_seed(SEED)
    os.makedirs(OUTPUT_DIR,exist_ok=True)
    run_dir=os.path.join(OUTPUT_DIR, datetime.now().strftime("%Y%m%d_%H%M%S")); os.makedirs(run_dir,exist_ok=True)

    files=list_tifs(INPUT_DIR)
    if len(files)==0: raise RuntimeError("入力ディレクトリにTIFがありません。")

    # --- Train/Val/Test 分割  ---
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
    random_test_files  = [other_files[i] for i in idx[n_train+n_val:]]
    test_files  = force_test_files + random_test_files

    print("[Split]")
    print("  train:", len(train_files))
    print("   val :", len(val_files))
    print("  test:", len(test_files), "(forced:", len(force_test_files), ")")


    # --- データ仕様を推定 ---
    IN_BANDS = check_inbands_consistency(files)
    MODE = infer_dataset_mode(IN_BANDS)
    print(f"[Data] IN_BANDS={IN_BANDS} MODE={MODE} total={IN_BANDS + N_CLASSES}")

    # --- スケーリングパラメータ ---
    xmin,xmax=compute_minmax_over_train(train_files, mode=MODE, max_pix=200_000)
    scaler={"xmin":xmin.tolist(),"xmax":xmax.tolist(),"eps":EPS_LOG}

    # --- 配列化 ---
    train_X,train_Y=build_arrays(train_files,xmin,xmax,MAX_PIX_PER_IMG_TRAIN,DELTA_ZERO,EPS_LOG,mode=MODE)
    val_X,val_Y    =build_arrays(val_files,  xmin,xmax,MAX_PIX_PER_IMG_VAL,  DELTA_ZERO,EPS_LOG,mode=MODE)
    print(f"[TrainData] X={train_X.shape} Y={train_Y.shape}  [ValData] X={val_X.shape} Y={val_Y.shape}")
       
    V=helmert_submatrix(N_CLASSES)

    # --- 学習 ---
    model,best_val,best_ep = train_with_validation(
        train_X,train_Y,val_X,val_Y,V,
        epochs=EPOCHS,bs=BATCH_SIZE,lr=LR,wd=WEIGHT_DECAY,device=DEVICE,eps=EPS_LOG
    )
    print(f"[Best] epoch={best_ep}  val_aitchison={best_val:.6f}")

    # --- 検証（Val）を全件評価 ---
    rows_val, val_global_mean, val_global_mrmse, val_global_mae = predict_images_metric_and_save_mlp(
        val_files, model, xmin, xmax, V,
        EPS_LOG, DELTA_ZERO, DEVICE, MODE,
        preds_dir=None
    )

    print(f"[Val ] global_mean_aitchison_ilr = {val_global_mean:.6f}  global_mrmse = {val_global_mrmse:.6f}  global_mae = {val_global_mae:.6f}")

    # --- テスト（Test）を全件評価 + 予測TIF保存 ---
    preds_dir = os.path.join(run_dir, "preds")
    os.makedirs(preds_dir, exist_ok=True)

    print(f"[Pred] saving predictions for {len(test_files)} test images ...")

    rows_test, test_global_mean, test_global_mrmse, test_global_mae = predict_images_metric_and_save_mlp(
        test_files, model, xmin, xmax, V,
        EPS_LOG, DELTA_ZERO, DEVICE, MODE,
        preds_dir=preds_dir
    )

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
            "train_ratio":TRAIN_RATIO,"batch_size":BATCH_SIZE,
            "epochs":EPOCHS,"lr":LR,"weight_decay":WEIGHT_DECAY,"dropout":DROPOUT,
            "h1":HIDDEN1,"h2":HIDDEN2,"h3":HIDDEN3,
            "delta_zero":DELTA_ZERO,"eps_log":EPS_LOG,
            "max_pix_per_img_train":MAX_PIX_PER_IMG_TRAIN,"max_pix_per_img_val":MAX_PIX_PER_IMG_VAL,
            "device":str(DEVICE),"loss": "mixed_aitchison_mae",
            "scheduler_patience":SCHEDULER_PATIENCE,"early_stop_patience":EARLY_STOP_PATIENCE,
            "alpha_aitchison":ALPHA_AITCHISON,"use_huber_aitch":USE_HUBER_AITCH,"huber_delta":HUBER_DELTA
        }, f, ensure_ascii=False, indent=2)

    with open(os.path.join(run_dir,"scaler.json"),"w",encoding="utf-8") as f:
        json.dump(scaler,f,indent=2)

    torch.save(model.state_dict(), os.path.join(run_dir,"model.pt"))
    print(f"[Done] outputs: {run_dir}")

if __name__=="__main__":
    print(f"Device: {DEVICE}")
    main()
                 