import logging
from pathlib import Path

import torch

from config import play_mcts_config, training_config
from mcts import MCTS
from model import ContrastDualPolicyNet

from .base import BasePlayer

logger = logging.getLogger(__name__)


class AlphaZeroPlayer(BasePlayer):
    def __init__(
        self,
        player_id: int,
        model_path: str,
        num_simulations: int = None,
        c_puct: float = None,
    ):
        """AlphaZeroスタイルのプレイヤー
        
        Args:
            player_id: プレイヤーID (1 or 2)
            model_path: 学習済みモデルのパス
            num_simulations: MCTSシミュレーション回数（Noneの場合はplay_mcts_configから取得）
            c_puct: PUCT係数（Noneの場合はplay_mcts_configから取得）
        """
        super().__init__(player_id)
        device = torch.device(training_config.DEVICE)
        model = ContrastDualPolicyNet().to(device)
        
        # プレイ時のデフォルト設定を使用
        self.num_simulations = num_simulations if num_simulations is not None else play_mcts_config.NUM_SIMULATIONS
        self.c_puct = c_puct if c_puct is not None else play_mcts_config.C_PUCT
        
        if Path(model_path).exists():
            model.load_state_dict(torch.load(model_path, map_location=device))
            logger.info(f"Model loaded from {model_path}")
        else:
            logger.warning(
                f"Model file not found: {model_path}. Using untrained model."
            )
        model.eval()
        
        # プレイ時の設定でMCTSを初期化（ノイズなし、より多くのシミュレーション）
        self.mcts = MCTS(
            network=model,
            device=device,
            c_puct=self.c_puct,
            alpha=play_mcts_config.DIRICHLET_ALPHA,
            epsilon=play_mcts_config.DIRICHLET_EPSILON,
        )
        
        logger.info(f"AlphaZero player initialized with {self.num_simulations} simulations, C_PUCT={self.c_puct}")

    def get_action(self, game):
        """ゲームの状態に基づいてAlphaZeroスタイルで行動を選択するメソッド

        Args:
            game: 現在のゲーム状態を表すContrastGameオブジェクト

        Returns:
            選択された行動のインデックス (int) と評価値 (float)
        """
        policy, values = self.mcts.search(game, self.num_simulations)

        if not policy:
            logger.error("AIが行動を選択できませんでした")
            return None

        # 最も訪問回数が多いアクションを選択
        action = max(policy, key=lambda x: policy[x])
        value = values.get(action, 0.0)
        logger.info(f"AI selected action {action} with value {value:.3f}")
        return action, value
