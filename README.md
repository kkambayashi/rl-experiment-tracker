# RL Experiment Tracker (最小版)

SRS_v2 の実験バックアップ (HandyRL の `trainlog/`) を読み、**保存した図** と **何回の平均か** をひと目で確認するための小さな Streamlit アプリです。

- `Start...` フォルダに保存した png をまとめて見られる
- 各条件フォルダの `train_log_XX.txt` の本数 = その条件を回した回数 (= 平均に使った回数) が分かる
  - `train_log_20.txt` まであれば 20 回
  - 末尾に `time :` が無いログ (途中で止まった物) は別に数える
- 同じバッチ内で設定が違う項目 (η など) を表に出すので、その図が何を比べているか分かる

バックアップは読むだけで、書き込み・移動・外部送信はしません。

## ファイル構成 (3 つだけ)

| ファイル | 役割 | 行数の目安 |
|---|---|---|
| `scan.py` | `trainlog/` を走査して `Batch` (図 + 条件ごとの実行回数) にまとめる。Streamlit に依存しない | 約 150 行 |
| `app.py` | 画面。`scan()` を呼んで並べるだけ | 約 90 行 |
| `test_scan.py` | `scan.py` のテスト (tmp フォルダに小さな trainlog を作って確認) | 約 90 行 |

`archive_v1/` には、以前の多機能版 (一覧・比較・集計・学習曲線) がそのまま残っています。不要なら削除して構いません。

## 起動

```bash
cd rl-experiment-tracker
uv sync --extra dev          # 初回のみ (pip なら: pip install -r requirements.txt)
uv run streamlit run app.py
```

`~/Desktop/SRS_v2_backup/trainlog` があれば自動でそこを読みます。無ければ `sample_data/trainlog` (合成データ) を表示します。
サイドバーでパスを変えられます。

## テスト

```bash
uv run pytest -q
```

## 対応している構造

```
trainlog/
  Start202609301533/            # バッチ (実験スクリプト 1 回分)
    202609301533/               # 条件 (config 1 つ分)
      config_01.yaml
      train_log_01.txt ... train_log_20.txt   # 本数 = 回数
    eta_vs_c_floor.png          # 図
  202606051724/                 # 旧形式 (trainlog 直下に条件フォルダ) も可
  SRS_v2_v3_eta_...png          # trainlog 直下の図は先頭に表示
```

設定 (アルゴリズム・η など) は `train_log_01.txt` の 1 行目 (実行時の設定 dict) から `ast.literal_eval` で読みます。
読めない場合は「不明」と表示し、推測はしません。zip のみ残っているバッチは対象外です。

## 制約と今後

- 「実行回数」はファイル本数です。途中で止まったログも本数に含まれるので、`正常終了` 列と合わせて見てください
- seed は全実験で 0 (研究コードの仕様) のため表示していません
- 学習曲線の再計算や条件横断の集計が必要になったら `archive_v1/` の `rl_tracker/` を参考にできます

## AI の利用について

このコードは Claude と対話しながら作成しました。実データのフォルダ構造の調査、`scan.py` / `app.py` / テストの初版は AI が生成し、
「何を最低限の機能とするか」「実行回数はファイル本数で定義する」「未完了ログは別に数える」といった判断は人が行っています。
