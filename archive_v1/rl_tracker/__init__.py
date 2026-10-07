"""RL Experiment Tracker - 実験バックアップを読み込んで検索・比較するためのコア層。

モジュール構成:
    models    : データ構造 (ExperimentRun など)
    parsers   : ログ / config の文字列解析 (純粋関数)
    loaders/  : ファイル走査。形式ごとに 1 ファイル
    aggregate : 条件ごとの集計 (平均・標準偏差・実行回数)
    frame     : ExperimentRun の list を pandas.DataFrame に変換 (UI 用)
"""

__version__ = "0.1.0"
