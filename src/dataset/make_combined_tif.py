"""
OEM（*_oem.tif）を参照として target-res グリッドを定義し、
指定した要素（S2 / Embedding / OEM比率）を同じグリッドに揃えて結合し、
多バンドGeoTIFFとして保存する。

対応ケース:
- S2 + OEM        : --use-s2 --use-oem
- Emb + OEM       : --use-emb --use-oem

入力対応（ファイル名ルール）:
  oem_dir:  <base>_oem.tif
  s2_dir :  <base>{reproj-suffix}  (use-s2 のときだけ必要)
  emb_dir:  <base>{reproj-suffix}  (use-emb のときだけ必要)

実行例：S2 + OEM（S2quantile_2575の場合）
  python3 make_combined_tif.py \
  --use-s2 --use-oem \
  --oem-dir /mnt/d/kanno/oem_tif \
  --s2-dir /mnt/d/kanno/S2quantile_2575/s2_on_oem_tif \
  --reproj-suffix _s2_on_oem.tif \
  --out-dir /mnt/d/kanno/out/s2_oem_combined
"""

import argparse
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling


def parse_args():
    p = argparse.ArgumentParser(description="Create combined GeoTIFF (S2/Emb/OEM ratios on OEM grid).")

    p.add_argument("--oem-dir", required=True, help="Directory containing *_oem.tif (reference grid)")
    p.add_argument("--s2-dir", default="", help="Directory containing <base>_reproj.tif for S2")
    p.add_argument("--emb-dir", default="", help="Directory containing <base>_reproj.tif for Embedding")
    p.add_argument("--out-dir", required=True, help="Output directory")

    p.add_argument("--use-s2", action="store_true", help="Include Sentinel-2 bands")
    p.add_argument("--use-emb", action="store_true", help="Include Embedding bands")
    p.add_argument("--use-oem", action="store_true", help="Include OEM class ratios")

    p.add_argument("--oem-suffix", default="_oem.tif")
    p.add_argument("--reproj-suffix", default="_reproj.tif")
    p.add_argument("--out-suffix", default="_combined.tif")

    p.add_argument("--target-res", type=float, default=10.0, help="Target resolution in meters (default: 10)")
    p.add_argument("--classes", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6, 7, 8])

    p.add_argument(
        "--emb-resampling",
        default="bilinear",
        choices=["nearest", "bilinear", "cubic"],
        help="Resampling for embedding (default: bilinear)",
    )
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def _resampling(name: str) -> Resampling:
    return {"nearest": Resampling.nearest, "bilinear": Resampling.bilinear, "cubic": Resampling.cubic}[name]


def oem_ratio_1m_to_target(oem_1m, nodata, res_x, res_y, h10, w10, target_res, classes):
    num_classes = len(classes)
    class_to_index = {cls_id: i for i, cls_id in enumerate(classes)}

    # メモリ効率化：フラット化してからマスク処理
    pixel = oem_1m.ravel()
    
    if nodata is None:
        mask = np.ones(pixel.shape, dtype=bool)
    else:
        mask = pixel != nodata

    # 有効なピクセルのインデックスだけを計算
    valid_indices = np.where(mask)[0]
    pixel = pixel[valid_indices]

    # グリッドインデックスの計算（有効ピクセルのみ）
    row_idx, col_idx = np.unravel_index(valid_indices, oem_1m.shape)
    row10 = (row_idx * res_y // target_res).astype(np.int64)
    col10 = (col_idx * res_x // target_res).astype(np.int64)
    cell_id = row10 * w10 + col10

    # ベクトル化されたクラスマッピング
    out = np.zeros((num_classes, h10 * w10), dtype=np.float32)
    for cls_id in classes:
        idx = class_to_index[cls_id]
        m = pixel == cls_id
        if np.any(m):
            out[idx] = np.bincount(cell_id[m], minlength=h10 * w10)

    out = out.reshape(num_classes, h10, w10)
    totals = out.sum(axis=0, keepdims=True)
    totals[totals == 0] = 1
    out /= totals
    return out


def reproject_stack_to_grid(src_path, dst_crs, dst_transform, dst_w, dst_h, resampling):
    with rasterio.open(src_path) as src:
        if src.crs is None:
            raise ValueError(f"Missing CRS: {src_path}")

        arr = np.empty((src.count, dst_h, dst_w), dtype=np.float32)
        
        # 複数バンドを効率的に処理
        for b in range(1, src.count + 1):
            reproject(
                source=rasterio.band(src, b),
                destination=arr[b - 1],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                dst_width=dst_w,
                dst_height=dst_h,
                resampling=resampling,
            )
        meta = src.meta.copy()
    
    # メモリが逼迫している場合のため、float32で最適化
    return arr.astype(np.float32), meta


def process_one(oem_path, s2_path, emb_path, out_path, use_s2, use_emb, use_oem, classes, target_res, emb_resampling):
    # --- OEM読み込み（参照グリッド） ---
    with rasterio.open(oem_path) as ref:
        oem_1m = ref.read(1)
        nodata = ref.nodata
        crs = ref.crs
        if crs is None:
            raise ValueError(f"Missing CRS in OEM: {oem_path}")

        h1, w1 = oem_1m.shape
        res_x, res_y = ref.res
        xmin, ymin, xmax, ymax = ref.bounds

    h10 = int(np.ceil(h1 * res_y / target_res))
    w10 = int(np.ceil(w1 * res_x / target_res))
    dst_transform = rasterio.transform.from_bounds(xmin, ymin, xmax, ymax, w10, h10)

    stacks = []
    band_names = []

    # --- S2 ---
    base_meta = None
    if use_s2:
        s2_arr, s2_meta = reproject_stack_to_grid(
            s2_path, crs, dst_transform, w10, h10, Resampling.bilinear
        )
        stacks.append(s2_arr)
        band_names += [f"S2_band_{i+1}" for i in range(s2_arr.shape[0])]
        base_meta = s2_meta

    # --- Embedding ---
    if use_emb:
        emb_arr, emb_meta = reproject_stack_to_grid(
            emb_path, crs, dst_transform, w10, h10, _resampling(emb_resampling)
        )
        stacks.append(emb_arr)
        band_names += [f"Emb_band_{i:02d}" for i in range(emb_arr.shape[0])]
        if base_meta is None:
            base_meta = emb_meta

    # --- OEM比率 ---
    if use_oem:
        ratio = oem_ratio_1m_to_target(oem_1m, nodata, res_x, res_y, h10, w10, target_res, classes)
        stacks.append(ratio)
        band_names += [f"OEM_class_ratio_{cls}" for cls in classes]

    if not stacks:
        raise ValueError("No layers selected. Use at least one of --use-s2/--use-emb/--use-oem")

    # メモリ効率的に配列連結
    combined = np.concatenate(stacks, axis=0, dtype=np.float32)

    # --- 出力メタ（どれか1つのmetaをベースに） ---
    if base_meta is None:
        # use_oem only の場合（metaが無いので最小限作る）
        base_meta = {
            "driver": "GTiff",
            "dtype": "float32",
            "count": combined.shape[0],
            "height": h10,
            "width": w10,
        }

    meta = dict(base_meta)
    meta.update(
        {
            "count": combined.shape[0],
            "height": h10,
            "width": w10,
            "crs": crs,
            "transform": dst_transform,
            "dtype": "float32",
            "compress": "lzw",
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
        }
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(combined)
        for i, name in enumerate(band_names, start=1):
            dst.set_band_description(i, name)


def main():
    args = parse_args()

    if not (args.use_s2 or args.use_emb or args.use_oem):
        print("[ERROR] Please specify at least one: --use-s2 / --use-emb / --use-oem")
        return

    oem_dir = Path(args.oem_dir)
    s2_dir = Path(args.s2_dir) if args.s2_dir else None
    emb_dir = Path(args.emb_dir) if args.emb_dir else None
    out_dir = Path(args.out_dir)

    oem_files = sorted([p for p in oem_dir.iterdir() if p.is_file() and p.name.endswith(args.oem_suffix)])
    if not oem_files:
        print(f"[ERROR] No OEM files found: {oem_dir} (suffix={args.oem_suffix})")
        return

    total = len(oem_files)
    done = 0
    failed = 0

    for idx, oem_path in enumerate(oem_files, start=1):
        base = oem_path.name[: -len(args.oem_suffix)]

        s2_path = (s2_dir / f"{base}{args.reproj_suffix}") if args.use_s2 else None
        emb_path = (emb_dir / f"{base}{args.reproj_suffix}") if args.use_emb else None
        out_path = out_dir / f"{base}{args.out_suffix}"

        # 必要な入力の存在チェック（先にスキップ判定）
        skip = False
        
        if args.use_s2:
            if s2_dir is None:
                print("[ERROR] --use-s2 requires --s2-dir")
                return
            if not s2_path.exists():
                print(f"[WARN] ({idx}/{total}) missing S2: {s2_path}")
                skip = True

        if args.use_emb and not skip:
            if emb_dir is None:
                print("[ERROR] --use-emb requires --emb-dir")
                return
            if not emb_path.exists():
                print(f"[WARN] ({idx}/{total}) missing Emb: {emb_path}")
                skip = True

        if skip:
            continue

        if out_path.exists() and not args.overwrite:
            print(f"[SKIP] ({idx}/{total}) exists: {out_path}")
            continue

        print(f"[INFO] ({idx}/{total}) make: {out_path.name}")

        try:
            process_one(
                oem_path=oem_path,
                s2_path=s2_path,
                emb_path=emb_path,
                out_path=out_path,
                use_s2=args.use_s2,
                use_emb=args.use_emb,
                use_oem=args.use_oem,
                classes=args.classes,
                target_res=args.target_res,
                emb_resampling=args.emb_resampling,
            )
            done += 1
        except Exception as e:
            print(f"[ERROR] {base}: {e}")
            failed += 1

    print(f"[DONE] processed={done}/{total}, failed={failed}")


if __name__ == "__main__":
    main()
