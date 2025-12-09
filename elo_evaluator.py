import logging
from pathlib import Path

import ray
import torch

from contrast_game import P1, P2, ContrastGame
from logger import setup_logger
from mcts import MCTS
from model import ContrastDualPolicyNet
from players import RuleBasedPlayer


# ★変更: Rayのアクターとして定義 (CPUを1つ専有)
@ray.remote(num_cpus=1)
class EloEvaluator:
    def __init__(self, device_str="cpu", baseline_elo=1000, k_factor=32):
        # Rayのシリアライズ対策で device は文字列で受け取り内部で変換
        self.device = torch.device(device_str)
        self.baseline_elo = baseline_elo
        self.agent_elo = 1000
        self.k_factor = k_factor
        self.rule_based_bot_p1 = RuleBasedPlayer(P1)
        self.rule_based_bot_p2 = RuleBasedPlayer(P2)
        setup_logger(log_file=Path(__file__).parent / "logs/elo.log")
        self.logger = logging.getLogger(__name__)

        # 評価用モデルのインスタンスを保持（都度生成しない）
        self.model = ContrastDualPolicyNet().to(self.device)
        self.model.eval()

    def evaluate(self, model_weights, step_count, num_games=10, mcts_simulations=50):
        """
        対戦を行いELOを更新し、モデルを保存する
        """
        # ウェイトのロード
        self.model.load_state_dict(model_weights)

        wins = 0
        losses = 0
        draws = 0

        # 対戦ループ
        for i in range(num_games):
            game = ContrastGame()
            mcts = MCTS(network=self.model, device=self.device)

            # 手番: 前半はモデル先手
            if i < num_games // 2:
                model_player = P1
                rb_player = P2
                rb_bot = self.rule_based_bot_p2
            else:
                model_player = P2
                rb_player = P1
                rb_bot = self.rule_based_bot_p1

            step = 0
            while not game.game_over and step < 150:
                if game.current_player == model_player:
                    policy, _ = mcts.search(game, mcts_simulations)
                    if not policy:
                        break
                    action = max(policy, key=lambda x: policy[x])
                else:
                    action = rb_bot.get_action(game)
                    if action is None:
                        break

                game.step(action)
                step += 1

            if game.winner == model_player:
                wins += 1
            elif game.winner == rb_player:
                losses += 1
            else:
                draws += 1

        # ELO計算
        total_score = wins + (draws * 0.5)
        expected_score = (
            self._get_expected_score(self.agent_elo, self.baseline_elo) * num_games
        )

        prev_elo = self.agent_elo
        self.agent_elo += self.k_factor * (total_score - expected_score)
        diff = self.agent_elo - prev_elo

        win_rate = (wins + 0.5 * draws) / num_games * 100

        # 結果のログ出力
        log_msg = (
            f"[Evaluation Step {step_count}] "
            f"Win: {wins}, Loss: {losses}, Draw: {draws} ({win_rate:.1f}%) | "
            f"ELO: {self.agent_elo:.1f} ({diff:+.1f})"
        )
        self.logger.info(log_msg)

        # モデルの保存 (評価プロセス側で行う)
        save_path = f"models/model_step_{step_count}_elo_{int(self.agent_elo)}.pth"
        torch.save(model_weights, save_path)

        return self.agent_elo, win_rate

    def _get_expected_score(self, rating_a, rating_b):
        """勝率の期待値を計算"""
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
