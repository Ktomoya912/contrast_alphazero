import math

import numpy as np
import torch
import torch.nn.functional as F

from contrast_game import ContrastGame


class MCTS:
    def __init__(
        self,
        network: torch.nn.Module,
        device: torch.device,
        alpha=0.3,
        c_puct=1.0,
        epsilon=0.25,
    ):
        self.network = network
        self.device = device
        self.alpha = alpha
        self.c_puct = c_puct
        self.eps = epsilon

        # 状態の識別キー: (pieces_bytes, tiles_bytes, counts_bytes, player_int, move_count)
        self.P = {}
        self.N = {}
        self.W = {}

    def game_to_key(self, game):
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
        root_key = self.game_to_key(root_game)

        # 未展開ならルートを展開
        if root_key not in self.P:
            self._expand(root_game)

        # 辞書アクセスに修正 (None対策)
        if root_key not in self.P:
            return {}

        valid_actions = list(self.P[root_key].keys())

        # 合法手がない場合
        if not valid_actions:
            return {}

        # ルートノードにディリクレノイズを付加
        dirichlet_noise = np.random.dirichlet([self.alpha] * len(valid_actions))
        for i, action in enumerate(valid_actions):
            self.P[root_key][action] = (1 - self.eps) * self.P[root_key][
                action
            ] + self.eps * dirichlet_noise[i]

        # シミュレーション実行
        for _ in range(num_simulations):
            # ルートからの探索を開始 (コピーを使用)
            self._evaluate(root_game.copy())

        # 訪問回数に基づいたPolicyを返す
        root_visits = sum(self.N[root_key].values())
        if root_visits == 0:
            # 万が一訪問が0回の場合(通常ありえないが)は一様分布を返す
            return {a: 1.0 / len(valid_actions) for a in valid_actions}

        mcts_policy = {a: self.N[root_key][a] / root_visits for a in valid_actions}

        return mcts_policy

    def _evaluate(self, game):
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

    def _expand(self, game):
        """
        ニューラルネットで推論し、Prior ProbabilityとValueを計算して保存する
        """
        key = self.game_to_key(game)

        input_tensor = (
            torch.from_numpy(game.encode_state()).unsqueeze(0).to(self.device)
        )

        self.network.eval()
        with torch.no_grad():
            move_logits, tile_logits, value = self.network(input_tensor)

        value = value.item()  # Scalar

        legal_actions = game.get_all_legal_actions()  # List[int]

        if not legal_actions:
            self.P[key] = {}
            self.N[key] = {}
            self.W[key] = {}
            return value

        m_logits = move_logits[0].cpu().numpy()  # shape (625,)
        t_logits = tile_logits[0].cpu().numpy()  # shape (51,)

        temp_logits = []
        action_mapping = []

        for action_hash in legal_actions:
            # デコード (51 = ACTION_SIZE_TILE)
            move_idx = action_hash // 51
            tile_idx = action_hash % 51

            combined_logit = m_logits[move_idx] + t_logits[tile_idx]
            temp_logits.append(combined_logit)
            action_mapping.append(action_hash)

        temp_logits = np.array(temp_logits)
        probs = F.softmax(torch.tensor(temp_logits), dim=0).numpy()

        self.P[key] = {}
        self.N[key] = {}
        self.W[key] = {}

        for action_hash, prob in zip(action_mapping, probs):
            self.P[key][action_hash] = prob
            self.N[key][action_hash] = 0
            self.W[key][action_hash] = 0

        return value
