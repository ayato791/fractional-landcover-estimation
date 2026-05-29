# Dataset Preparation

このディレクトリには、学習用GeoTIFFを作成するためのスクリプトが含まれている。

## データ取得

学習用GeoTIFFを作成する前に、以下のデータを準備すること。

### 1. OpenEarthMapラベルデータ

OpenEarthMapの土地被覆ラベルデータをダウンロードし、任意のディレクトリに配置する。
- [OpenEarthMap Dataset (Zenodo)](https://zenodo.org/records/7223446)から
例：

```text
data/
└── oem_tif/
    ├── tokyo_1_oem.tif
    ├── tokyo_2_oem.tif
    └── ...
```

### 2. Sentinel-2画像

[../sentinel2](../sentinel2) 内の `Sentinel_2_get.ipynb` を Google Colab 上で実行し、対象地域・対象期間の Sentinel-2 画像を取得する。

例：

```text
data/
└── s2_tif/
    ├── tokyo_1.tif
    ├── tokyo_2.tif
    └── ...
```

---

## データセット作成手順

### 1. Sentinel-2画像の再投影

`reproject_s2_to_oem.py` を実行し、Sentinel-2画像をOpenEarthMapと同じ座標系およびグリッドへ再投影する。

```bash
python reproject_s2_to_oem.py \
    --s2-dir <s2_dir> \
    --oem-dir <oem_dir> \
    --out-dir <output_dir>
```

実行例：

```bash
python reproject_s2_to_oem.py \
    --s2-dir data/s2_tif \
    --oem-dir data/oem_tif \
    --out-dir data/s2_reproj_tif
```

出力：

```text
data/
└── s2_reproj_tif/
    ├── tokyo_1_reproj.tif
    ├── tokyo_2_reproj.tif
    └── ...
```

---

### 2. 学習用GeoTIFFの作成

#### Sentinel-2 + OpenEarthMap

```bash
python make_combined_tif.py \
    --oem-dir data/oem_tif \
    --s2-dir data/s2_reproj_tif \
    --out-dir data/combined_tif \
    --use-s2 \
    --use-oem
```
---
### 出力データ
生成された `*_combined.tif` は学習スクリプトの入力として利用される。

例：

```text
data/
└── combined_tif/
    ├── tokyo_1_combined.tif
    ├── tokyo_2_combined.tif
    └── ...
```

### バンド構成

`*_combined.tif` には以下の情報が含まれる。

* Sentinel-2スペクトルバンド
* OpenEarthMap土地被覆構成比率

学習スクリプト（MLP、2D CNN、3D CNN）は、この `combined_tif` ディレクトリを入力として使用する。
