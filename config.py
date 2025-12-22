"""設定ファイル

このモジュールでは、プロジェクト全体で使用される定数とハイパーパラメータを定義します。
"""

import os
from dataclasses import dataclass

import torch


# ===== ゲーム設定 =====
@dataclass
class GameConfig:
    """ゲームの基本設定"""

    BOARD_SIZE: int = 5
    NUM_BOARD_POSITIONS: int = BOARD_SIZE * BOARD_SIZE  # 25
    NUM_MOVES: int = NUM_BOARD_POSITIONS * NUM_BOARD_POSITIONS  # 625
    NUM_TILE_ACTIONS: int = 1 + NUM_BOARD_POSITIONS * 2  # 51: pass + black + gray

    INITIAL_BLACK_TILES: int = 3
    INITIAL_GRAY_TILES: int = 1

    HISTORY_SIZE: int = 8
    INPUT_CHANNELS: int = 90  # 履歴8手分の特徴量

    MAX_STEPS_PER_GAME: int = 150  # ゲームの最大手数


# ===== MCTS設定 =====
@dataclass
class MCTSConfig:
    """MCTSのハイパーパラメータ"""

    NUM_SIMULATIONS: int = 50  # デフォルトのシミュレーション回数
    DIRICHLET_ALPHA: float = 0.3  # ディリクレノイズのパラメータ
    DIRICHLET_EPSILON: float = 0.25  # ノイズの混合比率
    C_PUCT: float = 1.0  # PUCTアルゴリズムの探索係数

    # 温度パラメータの設定
    TEMPERATURE_THRESHOLD: int = 30  # この手数まではランダム性を残す


# ===== ニューラルネットワーク設定 =====
@dataclass
class NetworkConfig:
    """ニューラルネットワークのアーキテクチャ設定"""

    NUM_RES_BLOCKS: int = 4  # Residualブロックの数
    NUM_FILTERS: int = 64  # 畳み込み層のフィルタ数

    # ヘッドの設定
    MOVE_HEAD_FILTERS: int = 32
    TILE_HEAD_FILTERS: int = 16
    VALUE_HEAD_FILTERS: int = 4
    VALUE_HIDDEN_SIZE: int = 32


# ===== 学習設定 =====
@dataclass
class TrainingConfig:
    """学習のハイパーパラメータ"""

    # デバイス
    DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 並列処理
    NUM_CPUS: int = os.cpu_count() or 2

    # バッチとバッファ
    BATCH_SIZE: int = 1024
    BUFFER_SIZE: int = 20_000

    # 最適化
    LEARNING_RATE: float = 0.001
    WEIGHT_DECAY: float = 1e-4
    LR_STEP_SIZE: int = 50_000  # 学習率を減衰させるステップ間隔
    LR_GAMMA: float = 0.5  # 学習率の減衰率

    # 学習ステップ
    MAX_STEPS: int = 150  # 1ゲームあたりの最大手数（引き分け防止）
    MAX_EPOCH: int = 50 * 10_000  # 総学習ステップ数

    # ログとチェックポイント
    LOG_INTERVAL: int = 50  # ログ出力の間隔
    SAVE_INTERVAL: int = 1000  # モデル保存の間隔

    def __str__(self) -> str:
        return (
            f"{self.BATCH_SIZE=}, {self.BUFFER_SIZE=}, {self.LEARNING_RATE=}, {self.WEIGHT_DECAY=}"
            f", {self.LR_STEP_SIZE=}, {self.LR_GAMMA=}, {self.MAX_EPOCH=}"
        )


# ===== 評価設定 =====
@dataclass
class EvaluationConfig:
    """モデル評価の設定"""

    EVAL_INTERVAL: int = 1000  # 評価を実行する間隔（学習ステップ数）
    EVAL_NUM_GAMES: int = 10  # 評価時の対戦回数
    EVAL_MCTS_SIMS: int = 50  # 評価時のMCTSシミュレーション回数

    BASELINE_ELO: int = 1000  # ベースラインAIの初期ELO
    K_FACTOR: int = 32  # ELO計算のK係数

    def __str__(self) -> str:
        return f"{self.EVAL_NUM_GAMES=}, {self.EVAL_MCTS_SIMS=}, {self.BASELINE_ELO=}, {self.K_FACTOR=}"


# ===== パス設定 =====
@dataclass
class PathConfig:
    """ディレクトリとファイルパスの設定"""

    LOGS_DIR: str = "logs"
    MODELS_DIR: str = "models"
    FINAL_MODEL_PATH: str = "contrast_model_final.pth"


# グローバル設定インスタンス（簡単にアクセスできるように）
game_config = GameConfig()
mcts_config = MCTSConfig()
network_config = NetworkConfig()
training_config = TrainingConfig()
evaluation_config = EvaluationConfig()
path_config = PathConfig()
