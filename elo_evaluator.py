from contrast_game import P1, P2, ContrastGame
from logger import get_logger
from mcts import MCTS
from model import ContrastDualPolicyNet
from rule_based_ai import RuleBasedPlayerV2

logger = get_logger(__name__)


class EloEvaluator:
    def __init__(self, device, baseline_elo=1000, k_factor=32):
        self.device = device
        self.baseline_elo = baseline_elo  # ルールベースAIのレート（固定）
        self.agent_elo = 1000  # 学習エージェントのレート（初期値）
        self.k_factor = k_factor
        self.rule_based_bot_p1 = RuleBasedPlayerV2(P1)
        self.rule_based_bot_p2 = RuleBasedPlayerV2(P2)

    def evaluate(self, model_weights, num_games=10, mcts_simulations=50):
        """
        現在のモデルとルールベースAIで対戦を行い、ELOを更新する
        """
        # モデルの準備
        model = ContrastDualPolicyNet().to(self.device)
        model.load_state_dict(model_weights)
        model.eval()

        wins = 0
        losses = 0
        draws = 0

        # 対戦ループ
        for i in range(num_games):
            game = ContrastGame()
            mcts = MCTS(network=model, device=self.device)

            # 手番の交代 (前半はモデルが先手、後半は後手)
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
                    # AlphaZeroのターン (Greedy)
                    policy, _ = mcts.search(game, mcts_simulations)
                    if not policy:
                        break
                    action = max(policy, key=policy.get)
                else:
                    # RuleBasedのターン
                    action = rb_bot.get_action(game)
                    if action is None:
                        break

                game.step(action)
                step += 1

            # 結果集計
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

        # レート更新
        new_elo = self.agent_elo + self.k_factor * (total_score - expected_score)
        diff = new_elo - self.agent_elo
        self.agent_elo = new_elo

        win_rate = (wins + 0.5 * draws) / num_games * 100

        logger.info(
            f"[Evaluation] Games: {num_games}, Win: {wins}, Loss: {losses}, Draw: {draws}"
        )
        logger.info(
            f"[Evaluation] WinRate: {win_rate:.1f}%, ELO: {self.agent_elo:.1f} ({diff:+.1f})"
        )

        return self.agent_elo, win_rate

    def _get_expected_score(self, rating_a, rating_b):
        """勝率の期待値を計算"""
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
