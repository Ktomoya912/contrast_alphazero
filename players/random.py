import random

from contrast_game import ContrastGame

from .base import BasePlayer


class RandomPlayer(BasePlayer):
    def get_action(self, game: ContrastGame):
        """ゲームの状態に基づいてランダムに行動を選択するメソッド

        Args:
            game: 現在のゲーム状態を表すContrastGameオブジェクト

        Returns:
            選択された行動のインデックス (int)
        """
        valid_actions = game.get_all_legal_actions()
        if not valid_actions:
            return None  # 合法手がない場合はNoneを返す
        return random.choice(valid_actions)
