import logging
from pathlib import Path

import ray
import torch

from config import evaluation_config, game_config
from contrast_game import P1, P2, ContrastGame
from logger import setup_logger
from mcts import MCTS
from model import ContrastDualPolicyNet
from players.rule_based import RuleBasedPlayer


@ray.remote(num_cpus=1)
class EloEvaluator:
    """ELOレーティングによるモデル評価クラス

    ルールベースAIとの対戦を通じてモデルの強さを評価し、
    ELOレーティングを管理します。
    """

    def __init__(self, device_str="cpu", baseline_elo=None, k_factor=None):
        """
        Args:
            device_str: 計算デバイスの文字列表現
            baseline_elo: ベースラインAIのELOレーティング (Noneの場合はconfig.pyから取得)
            k_factor: ELO計算のK係数 (Noneの場合はconfig.pyから取得)
        """
        # config.pyからデフォルト値を取得
        if baseline_elo is None:
            baseline_elo = evaluation_config.BASELINE_ELO
        if k_factor is None:
            k_factor = evaluation_config.K_FACTOR

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

        self.save_path = Path("models")
        self.save_path.mkdir(exist_ok=True)

    def evaluate(self, model_weights, step_count, num_games=10, mcts_simulations=50):
        """モデルを評価してELOを更新

        Args:
            model_weights: 評価するモデルの重み
            step_count: 現在の学習ステップ数
            num_games: 対戦回数
            mcts_simulations: MCTSシミュレーション回数

        Returns:
            (elo, win_rate): 更新後のELOと勝率
        """
        start_msg = (
            f"Starting evaluation at step {step_count}: "
            f"{num_games} games, {mcts_simulations} MCTS sims"
        )
        self.logger.info(start_msg)
        print(start_msg)  # 即座に出力

        # ウエイトのロード
        self.model.load_state_dict(model_weights)

        wins = 0
        losses = 0
        draws = 0

        # 対戦ループ
        for i in range(num_games):
            # 進捗ログ（10ゲームごと）
            if (i + 1) % 10 == 0:
                progress_msg = (
                    f"Evaluation progress: {i + 1}/{num_games} games completed"
                )
                self.logger.info(progress_msg)
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
            max_steps = game_config.MAX_STEPS_PER_GAME

            while not game.game_over and step < max_steps:
                if game.current_player == model_player:
                    policy, _ = mcts.search(game, mcts_simulations)
                    if not policy:
                        self.logger.warning(f"Game {i + 1}: No valid policy for model")
                        break
                    action = max(policy, key=lambda x: policy[x])
                else:
                    action = rb_bot.get_action(game)
                    if action is None:
                        self.logger.warning(
                            f"Game {i + 1}: No valid action for rule-based AI"
                        )
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
        save_path = (
            self.save_path / f"model_step_{step_count}_elo_{int(self.agent_elo)}.pth"
        )
        torch.save(model_weights, save_path)

        return self.agent_elo, win_rate

    def _get_expected_score(self, rating_a, rating_b):
        """勝率の期待値を計算"""
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
