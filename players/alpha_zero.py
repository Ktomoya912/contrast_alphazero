import logging
from pathlib import Path

import torch

from config import training_config
from mcts import MCTS
from model import ContrastDualPolicyNet

from .base import BasePlayer

logger = logging.getLogger(__name__)


class AlphaZeroPlayer(BasePlayer):
    def __init__(
        self,
        player_id: int,
        model_path: str,
        num_simulations: int = 100,
    ):
        super().__init__(player_id)
        device = torch.device(training_config.DEVICE)
        model = ContrastDualPolicyNet().to(device)
        self.num_simulations = num_simulations
        if Path(model_path).exists():
            model.load_state_dict(torch.load(model_path, map_location=device))
            logger.info(f"Model loaded from {model_path}")
        else:
            logger.warning(
                f"Model file not found: {model_path}. Using untrained model."
            )
        model.eval()
        self.mcts = MCTS(network=model, device=device)

    def get_action(self, game):
        """ゲームの状態に基づいてAlphaZeroスタイルで行動を選択するメソッド

        Args:
            game: 現在のゲーム状態を表すContrastGameオブジェクト

        Returns:
            選択された行動のインデックス (int) と評価値 (float)
        """
        self.mcts.search(game, num_simulations=100)

        policy, values = self.mcts.search(game, self.num_simulations)

        if not policy:
            logger.error("AIが行動を選択できませんでした")
            return None

        # 最も訪問回数が多いアクションを選択
        action = max(policy, key=lambda x: policy[x])
        value = values.get(action, 0.0)
        logger.info(f"AI selected action {action} with value {value:.3f}")
        return action, value
