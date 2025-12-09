"""
ロギング設定の一元管理モジュール

このモジュールでアプリケーション全体のロギング設定を行います。
各モジュールではget_logger()を呼び出してロガーインスタンスを取得してください。
"""

import logging
import sys
from pathlib import Path


def setup_logger(
    log_level=logging.INFO,
    log_file=None,
    log_format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    date_format="%Y-%m-%d %H:%M:%S",
    show_console=True,
):
    """
    アプリケーション全体のロギング設定を初期化

    Args:
        log_level: ログレベル (デフォルト: logging.INFO)
        log_file: ログファイルのパス (Noneの場合はコンソールのみ)
        log_format: ログメッセージのフォーマット
        date_format: 日時のフォーマット

    Note:
        この関数は通常、アプリケーションのエントリーポイントで一度だけ呼び出します。
    """
    # ルートロガーの取得
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # 既存のハンドラをクリア（重複を防ぐ）
    root_logger.handlers.clear()

    # フォーマッタの作成
    formatter = logging.Formatter(log_format, datefmt=date_format)

    if show_console:
        # コンソールハンドラの追加
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # ファイルハンドラの追加（指定された場合）
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def get_logger(name):
    """
    指定された名前のロガーインスタンスを取得

    Args:
        name: ロガー名（通常は __name__ を指定）

    Returns:
        logging.Logger: ロガーインスタンス

    Example:
        >>> from logger import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Hello, World!")
    """
    return logging.getLogger(name)
