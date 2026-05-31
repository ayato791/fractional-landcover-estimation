"""
OEM（*_oem.tif）を参照画像として、
対応する Sentinel-2（base.tif）を OEM と同じ CRS / 解像度 / 範囲 / グリッドに再投影して保存する。
実行例:
  python3 reproject_s2_to_oem.py \
    --s2-dir  /mnt/d/kanno/embeddings_v1_annual/s2_tif \
    --oem-dir /mnt/d/kanno/oem_tif \
    --out-dir /mnt/d/kanno/embeddings_v1_annual/s2_on_oem_tif
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import rasterio
from rasterio.warp import reproject, Resampling


def parse_args():
    p = argparse.ArgumentParser(description="Reproject S2 to match OEM grid (parallel).")
    p.add_argument("--s2-dir",  required=True, help="Directory of S2 GeoTIFFs (<base>.tif)")
    p.add_argument("--oem-dir", required=True, help="Directory of OEM GeoTIFFs (<base>_oem.tif)")
    p.add_argument("--out-dir", required=True, help="Output directory")
    p.add_argument("--suffix-oem", default="_oem.tif", help='OEM filename suffix (default: "_oem.tif")')
    p.add_argument("--out-suffix", default="_s2_on_oem.tif", help='Output suffix (default: "_s2_on_oem.tif")')
    p.add_argument(
        "--resampling",
        default="bilinear",
        choices=["nearest", "bilinear", "cubic"],
        help="Resampling method for S2 (default: bilinear)",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=None,
        help="並列プロセス数（省略時は CPU コア数を自動検出）",
    )
    return p.parse_args()


def get_resampling(name: str) -> Resampling:
    return {
        "nearest": Resampling.nearest,
        "bilinear": Resampling.bilinear,
        "cubic": Resampling.cubic,
    }[name]


# ── ワーカー関数（pickle 可能にするためモジュールレベルに定義） ──────────────

def _reproject_one(task: dict) -> dict:
    """
    1 ペア分の再投影を実行してステータスを返す。
    task keys: s2_path, oem_path, out_path, resampling_name
    """
    s2_path   = Path(task["s2_path"])
    oem_path  = Path(task["oem_path"])
    out_path  = Path(task["out_path"])
    resampling = get_resampling(task["resampling_name"])

    try:
        with rasterio.open(s2_path) as src, rasterio.open(oem_path) as ref:
            if ref.crs is None:
                return {"status": "warn", "msg": f"OEM CRS missing: {oem_path}"}
            if src.crs is None:
                return {"status": "warn", "msg": f"S2 CRS missing: {s2_path}"}

            dst_crs       = ref.crs
            dst_transform = ref.transform
            dst_width     = ref.width
            dst_height    = ref.height

            profile = src.profile.copy()
            profile.update(
                crs=dst_crs,
                transform=dst_transform,
                width=dst_width,
                height=dst_height,
            )
            # 必要であれば圧縮を有効化（例: profile.update(compress="deflate", predictor=2)）

            with rasterio.open(out_path, "w", **profile) as dst:
                for b in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, b),
                        destination=rasterio.band(dst, b),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=dst_transform,
                        dst_crs=dst_crs,
                        resampling=resampling,
                    )

        return {"status": "done"}

    except Exception as e:
        # 書きかけのファイルが残らないよう削除
        if out_path.exists():
            out_path.unlink()
        return {"status": "error", "msg": f"{s2_path.name}: {e}"}


# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    s2_dir  = Path(args.s2_dir)
    oem_dir = Path(args.oem_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    oem_files = sorted(
        p for p in oem_dir.iterdir()
        if p.is_file() and p.name.endswith(args.suffix_oem)
    )
    if not oem_files:
        print(f"[ERROR] No OEM files found in {oem_dir} with suffix {args.suffix_oem}")
        return

    total      = len(oem_files)
    tasks      = []
    missing_s2 = 0
    skipped    = 0

    for oem_path in oem_files:
        base     = oem_path.name[: -len(args.suffix_oem)]
        s2_path  = s2_dir  / f"{base}.tif"
        out_path = out_dir / f"{base}{args.out_suffix}"

        if not s2_path.exists():
            print(f"[WARN] missing S2: {s2_path}")
            missing_s2 += 1
            continue

        if out_path.exists():
            skipped += 1
            continue

        tasks.append({
            "s2_path":        str(s2_path),
            "oem_path":       str(oem_path),
            "out_path":       str(out_path),
            "resampling_name": args.resampling,
        })

    n_tasks  = len(tasks)
    n_workers = args.workers or os.cpu_count() or 4
    print(f"[INFO] total_oem={total}, to_process={n_tasks}, "
          f"missing_s2={missing_s2}, already_skipped={skipped}, workers={n_workers}")

    if n_tasks == 0:
        print("[DONE] Nothing to do.")
        return

    done  = 0
    error = 0

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_name = {
            executor.submit(_reproject_one, t): Path(t["s2_path"]).name
            for t in tasks
        }

        for i, future in enumerate(as_completed(future_to_name), start=1):
            name   = future_to_name[future]
            result = future.result()

            if result["status"] == "done":
                done += 1
                print(f"[{i:>5}/{n_tasks}] done : {name}")
            elif result["status"] == "warn":
                print(f"[{i:>5}/{n_tasks}] WARN : {result['msg']}")
            else:
                error += 1
                print(f"[{i:>5}/{n_tasks}] ERROR: {result['msg']}")

    print(
        f"\n[DONE] total_oem={total}, done={done}, error={error}, "
        f"missing_s2={missing_s2}, skipped={skipped}"
    )


if __name__ == "__main__":
    main()
