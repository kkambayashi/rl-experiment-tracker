# RL Experiment Tracker (最小版)

SRS_v2 の実験バックアップ (HandyRL の `trainlog/`) を読み、**保存した図** と **何回の平均か** をひと目で確認するための小さな Streamlit アプリです。

- `Start...` フォルダに保存した png をまとめて見られる
- 各条件フォルダの `train_log_XX.txt` の本数 = その条件を回した回数 (= 平均に使った回数) が分かる
  - `train_log_20.txt` まであれば 20 回
  - 末尾に `time :` が無いログ (途中で止まった物) は別に数える
- 同じバッチ内で設定が違う項目 (η など) を表に出すので、その図が何を比べているか分かる
- 一覧表から見たい図を選択して見ることができる

バックアップは読むだけで、書き込み・移動・外部送信はしません。

## ファイル構成

| ファイル | 役割 | 行数の目安 |
|---|---|---|
| `scan.py` | `trainlog/` を走査して `Batch` (図 + 条件ごとの実行回数) にまとめる。Streamlit に依存しない | 約 170 行 |
| `app.py` | 一覧表から選んだバッチを表示する | 約 130 行 |
| `test_scan.py` | `scan.py` のテスト (tmp フォルダに小さな trainlog を作って確認) | 約 90 行 |
| `アプリを起動.command` | ダブルクリックでアプリを起動する　| 約 20 行　|

## 起動

- ダブルクリックで起動する方法
  - `アプリを起動.command` をダブルクリック。止めるときはターミナルのウィンドウを閉じる。

- ターミナルから起動する方法
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
  SRS_v2_v3_eta_...png          # trainlog 直下の図は一番下に表示
```

設定 (アルゴリズム・η など) は `train_log_01.txt` の 1 行目 (実行時の設定 dict) から `ast.literal_eval` で読みます。
読めない場合は「不明」と表示し、推測はしません。zip のみ残っているバッチは対象外です。

## 制約と今後

- 「実行回数」はファイル本数です。途中で止まったログも本数に含まれるので、`正常終了` 列と合わせて見てください
- seed は全実験で 0 (研究コードの仕様) のため表示していません
- 学習曲線の再計算や条件横断の集計が必要になったら git の最初のコミット dd3ead9 (以前の多機能版（archive_v1/rl_tracker/）が残っており、
git show dd3ead9:archive_v1/rl_tracker/aggregate.py のように取り出せる)を参考にできます。

## AI の利用について

このコードは Claude と対話しながら作成しました。実データのフォルダ構造の調査、`scan.py` / `app.py` / テストの初版は AI が生成し、
「何を最低限の機能とするか」「実行回数はファイル本数で定義する」「未完了ログは別に数える」といった判断は人が行っています。
一覧表は AI に手順とコード例を教わりながら自分で実装し、起動ファイルは AI が作成しました。
