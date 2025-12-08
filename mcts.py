import math

import numpy as np
import torch
import torch.nn.functional as F

from config import mcts_config
from contrast_game import P2, ContrastGame, flip_action
from logger import get_logger

logger = get_logger(__name__)


class MCTS:
    """モンテカルロ木探索 (MCTS) の実装

    AlphaZeroスタイルのMCTSで、ニューラルネットワークによる
    価値評価と政策予測を組み合わせて探索を行います。
    """

    def __init__(
        self,
        network: torch.nn.Module,
        device: torch.device,
        alpha=None,
        c_puct=None,
        epsilon=None,
        verbose=False,
    ):
        """
        Args:
            network: 価値と政策を予測するニューラルネットワーク
            device: 計算デバイス
            alpha: ディリクレノイズのパラメータ (Noneの場合はconfig.pyから取得)
            c_puct: PUCTアルゴリズムの探索係数 (Noneの場合はconfig.pyから取得)
            epsilon: ノイズの混合比率 (Noneの場合はconfig.pyから取得)
            verbose: 詳細なログ出力を有効化
        """
        # config.pyからデフォルト値を取得
        if alpha is None:
            alpha = mcts_config.DIRICHLET_ALPHA
        if c_puct is None:
            c_puct = mcts_config.C_PUCT
        if epsilon is None:
            epsilon = mcts_config.DIRICHLET_EPSILON
        self.network = network
        self.device = device
        self.alpha = alpha
        self.c_puct = c_puct
        self.eps = epsilon
        self.verbose = verbose

        # 状態の識別キー: (pieces_bytes, tiles_bytes, counts_bytes, player_int, move_count)
        self.P = {}  # Prior probability
        self.N = {}  # Visit count
        self.W = {}  # Total action value

        # 統計情報
        self.search_stats = {
            "total_simulations": 0,
            "total_expansions": 0,
            "avg_depth": 0.0,
            "max_depth": 0,
        }

    def game_to_key(self, game: ContrastGame):
        """
        ContrastGameの状態を一意なハッシュ可能オブジェクト(タプル)に変換
        修正: move_countを含めることで、盤面が同一でも手数が違えば別状態として扱い、循環(無限再帰)を防ぐ
        """
        return (
            game.pieces.tobytes(),
            game.tiles.tobytes(),
            game.tile_counts.tobytes(),
            game.current_player,
            game.move_count,  # <--- 重要: これを追加
        )

    def search(self, root_game: ContrastGame, num_simulations: int):
        """ルートノードからMCTS探索を実行

        Args:
            root_game: 探索を開始するゲーム状態
            num_simulations: 実行するシミュレーション回数

        Returns:
            (mcts_policy, action_values):
                - mcts_policy: 各アクションの訪問確率分布
                - action_values: 各アクションのQ値
        """
        root_key = self.game_to_key(root_game)

        # 未展開ならルートを展開
        if root_key not in self.P:
            self._expand(root_game)

        # 辞書アクセスに修正 (None対策)
        if root_key not in self.P:
            if self.verbose:
                logger.warning("Root expansion failed for game state")
            return {}, {}

        valid_actions = list(self.P[root_key].keys())

        # 合法手がない場合
        if not valid_actions:
            if self.verbose:
                logger.warning("No valid actions at root")
            return {}, {}

        if self.verbose:
            logger.debug(
                f"MCTS search started: {num_simulations} simulations, "
                f"{len(valid_actions)} legal actions"
            )

        # ルートノードにディリクレノイズを付加
        dirichlet_noise = np.random.dirichlet([self.alpha] * len(valid_actions))
        for i, action in enumerate(valid_actions):
            self.P[root_key][action] = (1 - self.eps) * self.P[root_key][
                action
            ] + self.eps * dirichlet_noise[i]

        # シミュレーション実行
        for sim_idx in range(num_simulations):
            # ルートからの探索を開始 (コピーを使用)
            self._evaluate(root_game.copy())
            self.search_stats["total_simulations"] += 1

        # 訪問回数に基づいたPolicyを返す
        root_visits = sum(self.N[root_key].values())
        if root_visits == 0:
            # 万が一訪問が0回の場合(通常ありえないが)は一様分布を返す
            return {a: 1.0 / len(valid_actions) for a in valid_actions}, {}

        mcts_policy = {a: self.N[root_key][a] / root_visits for a in valid_actions}

        # 各アクションの評価値 (Q値) を計算
        action_values = {}
        for action in valid_actions:
            n = self.N[root_key][action]
            if n > 0:
                action_values[action] = self.W[root_key][action] / n
            else:
                action_values[action] = 0.0

        if self.verbose:
            best_action = max(mcts_policy, key=lambda x: mcts_policy[x])
            logger.debug(
                f"MCTS search completed: Best action visit rate={mcts_policy[best_action]:.3f}, "
                f"Q-value={action_values[best_action]:.3f}"
            )

        return mcts_policy, action_values

    def _evaluate(self, game: ContrastGame) -> float:
        """
        再帰的な探索関数
        """
        key = self.game_to_key(game)

        # 1. ゲーム終了判定
        if game.game_over:
            if game.winner == 0:  # Draw
                return 0
            # current_playerが勝者なら1, 敗者なら-1
            # 注意: evaluateに入った時点の手番プレイヤー視点での価値
            return 1 if game.winner == game.current_player else -1

        # 2. 未展開ノードなら展開して値を返す
        if key not in self.P:
            value = self._expand(game)
            return value

        # 3. 展開済みならPUCTでアクション選択
        valid_actions = list(self.P[key].keys())

        if not valid_actions:
            # 展開済みだが合法手がない（ゲーム終了扱い漏れなど）
            return 0

        sum_n = sum(self.N[key].values())
        sqrt_sum_n = math.sqrt(sum_n)

        best_score = -float("inf")
        best_action = -1

        for action in valid_actions:
            p = self.P[key][action]
            n = self.N[key][action]
            w = self.W[key][action]

            q = w / n if n > 0 else 0
            u = self.c_puct * p * sqrt_sum_n / (1 + n)

            score = q + u

            if score > best_score:
                best_score = score
                best_action = action

        # 4. 次の状態へ遷移 & 再帰 (Simulation step)
        # 以前の修正: 引数を1つにする
        game.step(best_action)

        # 相手の手番での価値が返ってくるため反転させる
        v = -self._evaluate(game)

        # 5. バックプロパゲーション
        self.W[key][best_action] += v
        self.N[key][best_action] += 1

        return v

    def _expand(self, game: ContrastGame) -> float:
        """ニューラルネットで推論し、Prior ProbabilityとValueを計算して保存

        Args:
            game: 展開するゲーム状態

        Returns:
            ノードの価値評価値
        """
        key = self.game_to_key(game)

        # encode_state内でP2なら自動的に反転される
        input_tensor = (
            torch.from_numpy(game.encode_state()).unsqueeze(0).to(self.device)
        )

        self.network.eval()
        with torch.no_grad():
            move_logits, tile_logits, value = self.network(input_tensor)

        value = value.item()

        legal_actions = game.get_all_legal_actions()

        if not legal_actions:
            self.P[key] = {}
            self.N[key] = {}
            self.W[key] = {}
            if self.verbose:
                logger.debug(f"Expanded terminal node with value={value:.3f}")
            return value

        self.search_stats["total_expansions"] += 1

        m_logits = move_logits[0].cpu().numpy()
        t_logits = tile_logits[0].cpu().numpy()

        temp_logits = []
        action_mapping = []

        # P2の場合は、実アクション(legal_actions)を「反転」させてから
        # ネットワークの出力(反転済みの盤面に対する推論)を参照する
        should_flip = game.current_player == P2

        for action_hash in legal_actions:
            # ネットワークに問い合わせるためのハッシュ
            query_hash = flip_action(action_hash) if should_flip else action_hash

            # デコード (51 = ContrastGame.ACTION_SIZE_TILE)
            move_idx = query_hash // 51
            tile_idx = query_hash % 51

            combined_logit = m_logits[move_idx] + t_logits[tile_idx]
            temp_logits.append(combined_logit)
            action_mapping.append(action_hash)  # 辞書には実ハッシュを保存

        temp_logits = np.array(temp_logits)
        probs = F.softmax(torch.tensor(temp_logits), dim=0).numpy()

        self.P[key] = {}
        self.N[key] = {}
        self.W[key] = {}

        for action_hash, prob in zip(action_mapping, probs):
            self.P[key][action_hash] = prob
            self.N[key][action_hash] = 0
            self.W[key][action_hash] = 0

        if self.verbose:
            logger.debug(
                f"Expanded node with {len(action_mapping)} actions, value={value:.3f}"
            )

        return value

    def get_stats(self) -> dict:
        """探索統計情報を取得

        Returns:
            統計情報の辞書
        """
        return self.search_stats.copy()

    def reset_stats(self) -> None:
        """統計情報をリセット"""
        self.search_stats = {
            "total_simulations": 0,
            "total_expansions": 0,
            "avg_depth": 0.0,
            "max_depth": 0,
        }
