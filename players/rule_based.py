import numpy as np

from contrast_game import (
    P1,
    P2,
    TILE_BLACK,
    TILE_GRAY,
    ContrastGame,
    decode_action,
)

from .base import BasePlayer


class RuleBasedPlayer(BasePlayer):
    """ルールベースのAIプレイヤー（高度な戦略版）

    C++版のKontrastBotV2を移植した戦略的なAIプレイヤー。
    7段階の優先順位に基づいて最適な手を選択します。
    """

    def __init__(self, player_id: int):
        super().__init__(player_id)
        self.opponent_id = P2 if player_id == P1 else P1

        # プレイヤーごとの方向定義
        # P1(1): 下(4)から上(0)へ攻める -> dir = -1
        # P2(2): 上(0)から下(4)へ�める -> dir = 1
        self.direction = -1 if player_id == P1 else 1
        self.goal_row = 0 if player_id == P1 else 4
        self.home_row = 4 if player_id == P1 else 0
        self.board_height = 5
        self.board_width = 5

    def get_action(self, game: ContrastGame):
        """戦略的な行動選択

        優先順位:
        1. 即時勝利
        2. 即時敗北の阻止
        3. パリティ制御
        4. 敵の横列形成阻止
        5. 先行駒の優先
        6. 側面攻撃
        7. スコアベース選択

        Args:
            game: 現在のゲーム状態

        Returns:
            選択されたアクションのハッシュ値（player_idに応じた正しい座標系）
        """
        legal_hashes = game.get_all_legal_actions()
        if not legal_hashes:
            return None

        # アクションハッシュを扱いやすいオブジェクト(辞書)リストに変換
        moves = self._decode_moves(game, legal_hashes)

        # 1. 即時勝利 (Check Immediate Win)
        for m in moves:
            if self._check_immediate_win(game, m):
                return m["hash"]

        # 2. 即時敗北の阻止 (Block Immediate Threat)
        block_move = self._block_immediate_threat(game, moves)
        if block_move:
            return block_move["hash"]

        # 3. パリティと小競り合いの制御 (Parity Skirmish Control)
        parity_move = self._parity_skirmish_control(game, moves)
        if parity_move:
            return parity_move["hash"]

        # 4. 敵の横列形成の阻止 (Interdict Row Formation)
        interdict_move = self._interdict_row_formation(game, moves)
        if interdict_move:
            return interdict_move["hash"]

        # 5. 先行する駒を優先 (Prioritize Lead Piece)
        lead_move = self._prioritize_lead_piece(game, moves)
        if lead_move:
            return lead_move["hash"]

        # 6. 直進する敵の側面攻撃 (Outflank Straight Runner)
        outflank_move = self._outflank_straight_runner(game, moves)
        if outflank_move:
            return outflank_move["hash"]

        # 7. スコアベースのフォールバック (Fallback by Score)
        return self._fallback_by_score(game, moves)["hash"]

    # --- 内部ヘルパーメソッド ---

    def _decode_moves(self, game, legal_hashes):
        """ハッシュリストを詳細情報のリストに変換"""
        decoded_moves = []
        for h in legal_hashes:
            move_idx, tile_idx = decode_action(h)
            from_idx = move_idx // 25
            to_idx = move_idx % 25
            fx, fy = from_idx % 5, from_idx // 5
            tx, ty = to_idx % 5, to_idx // 5

            place_tile = tile_idx > 0
            tile_type = None
            t_x, t_y = -1, -1

            if place_tile:
                if tile_idx <= 25:
                    tile_type = TILE_BLACK
                    idx = tile_idx - 1
                else:
                    tile_type = TILE_GRAY
                    idx = tile_idx - 26
                t_x, t_y = idx % 5, idx // 5

            decoded_moves.append(
                {
                    "hash": h,
                    "sx": fx,
                    "sy": fy,
                    "dx": tx,
                    "dy": ty,
                    "place_tile": place_tile,
                    "tile": tile_type,
                    "tx": t_x,
                    "ty": t_y,
                }
            )
        return decoded_moves

    def _row_progress(self, move):
        """ゴールの方向への進捗度"""
        delta = move["dy"] - move["sy"]
        return delta if self.player_id == P2 else -delta

    def _distance_to_nearest_empty_goal(self, game, x, y, player):
        target = 4 if player == P2 else 0
        best = 1000

        for gx in range(5):
            if game.pieces[target, gx] == 0:
                dist = abs(x - gx) + abs(y - target)
                best = min(best, dist)

        if best == 1000:
            best = abs(y - target)
        return best

    def _min_distance_to_empty_goal(self, game, player):
        best = 1000
        pieces = np.argwhere(game.pieces == player)
        for py, px in pieces:
            d = self._distance_to_nearest_empty_goal(game, px, py, player)
            best = min(best, d)
        return best

    def _project_row(self, y):
        """自分の視点での「進んだ距離」に変換 (0=Home, 4=Goal)"""
        return y if self.player_id == P2 else (self.board_height - 1 - y)

    def _collect_column_info(self, game):
        """各列の戦況（最前線の味方、敵、ギャップ）を分析"""
        cols = []

        for x in range(self.board_width):
            info = {
                "x": x,
                "has_friend": False,
                "friend_row": -1,
                "friend_proj": -1,
                "has_enemy_front": False,
                "enemy_front_row": -1,
                "enemy_front_proj": -1,
                "has_enemy_ahead": False,
                "enemy_ahead_row": -1,
                "gap": -1,
            }

            # 1. 味方の最前線を探す
            if self.player_id == P2:
                for y in range(self.board_height):
                    if game.pieces[y, x] == self.player_id:
                        info["has_friend"] = True
                        info["friend_row"] = y
            else:
                for y in range(self.board_height - 1, -1, -1):
                    if game.pieces[y, x] == self.player_id:
                        info["has_friend"] = True
                        info["friend_row"] = y

            if info["has_friend"]:
                info["friend_proj"] = self._project_row(info["friend_row"])

            # 2. 敵の最前線を探す
            if self.player_id == P2:
                for y in range(self.board_height):
                    if game.pieces[y, x] == self.opponent_id:
                        info["has_enemy_front"] = True
                        info["enemy_front_row"] = y
                        info["enemy_front_proj"] = self._project_row(y)
                        break
            else:
                for y in range(self.board_height - 1, -1, -1):
                    if game.pieces[y, x] == self.opponent_id:
                        info["has_enemy_front"] = True
                        info["enemy_front_row"] = y
                        info["enemy_front_proj"] = self._project_row(y)
                        break

            # 3. 味方最前線と敵とのギャップ
            if info["has_friend"]:
                y = info["friend_row"] + self.direction
                while 0 <= y < self.board_height:
                    occ = game.pieces[y, x]
                    if occ == self.opponent_id:
                        info["has_enemy_ahead"] = True
                        info["enemy_ahead_row"] = y
                        break
                    if occ != 0:
                        break
                    y += self.direction

                if info["has_enemy_ahead"]:
                    if self.player_id == P2:
                        info["gap"] = max(
                            0, info["enemy_ahead_row"] - info["friend_row"] - 1
                        )
                    else:
                        info["gap"] = max(
                            0, info["friend_row"] - info["enemy_ahead_row"] - 1
                        )

            cols.append(info)
        return cols

    # --- 戦略メソッド ---

    def _check_immediate_win(self, game: ContrastGame, move):
        """1. 即時勝利のチェック"""
        if move["dy"] == self.goal_row:
            return True
        return False

    def _block_immediate_threat(self, game: ContrastGame, moves):
        """2. 相手の即時勝利（あと1手）を防ぐ"""
        if self._min_distance_to_empty_goal(game, self.opponent_id) > 1:
            return None

        for m in moves:
            g_copy = game.copy()
            g_copy.step(m["hash"])

            if self._min_distance_to_empty_goal(g_copy, self.opponent_id) > 1:
                return m
        return None

    def _parity_skirmish_control(self, game, moves):
        """3. パリティ（偶奇）による主導権争い"""
        columns = self._collect_column_info(game)

        total_gap = 0
        counted = 0
        widest_col = -1
        widest_gap = -1

        for col in columns:
            if col["gap"] >= 0:
                total_gap += col["gap"]
                counted += 1
                if col["gap"] > widest_gap:
                    widest_gap = col["gap"]
                    widest_col = col["x"]

        if counted == 0:
            return None

        wants_forward = total_gap % 2 == 1

        if wants_forward:
            best = None
            best_score = -float("inf")

            for m in moves:
                if m["place_tile"]:
                    continue
                if m["dx"] != m["sx"]:
                    continue
                if self._row_progress(m) <= 0:
                    continue

                col = columns[m["sx"]]
                if not col["has_friend"] or col["friend_row"] != m["sy"]:
                    continue

                score = self._row_progress(m) * 120
                if col["gap"] >= 0:
                    score += (col["gap"] + 1) * 25

                if col["has_enemy_ahead"]:
                    remaining = 0
                    if self.player_id == P2:
                        remaining = max(0, col["enemy_ahead_row"] - m["dy"] - 1)
                    else:
                        remaining = max(0, m["dy"] - col["enemy_ahead_row"] - 1)
                    score += max(0, 60 - remaining * 15)

                if score > best_score:
                    best_score = score
                    best = m
            return best
        else:
            best = None
            best_score = -float("inf")

            for m in moves:
                if not m["place_tile"]:
                    continue
                tx = m["tx"]
                if not (0 <= tx < len(columns)):
                    continue

                col = columns[tx]
                if not col["has_friend"] or not col["has_enemy_front"]:
                    continue

                desired_row = col["enemy_front_row"] - self.direction

                if not (0 <= desired_row < self.board_height):
                    continue
                if m["ty"] != desired_row:
                    continue

                score = 140
                if col["gap"] >= 0:
                    score += col["gap"] * 12
                if tx == widest_col:
                    score += 30
                if m["tile"] == TILE_GRAY:
                    score += 30
                if m["tile"] == TILE_BLACK:
                    score += 20

                if score > best_score:
                    best_score = score
                    best = m
            return best

    def _interdict_row_formation(self, game: ContrastGame, moves):
        """4. 敵が横一列に並ぶのを阻止する"""
        columns = self._collect_column_info(game)
        targets = []

        for i in range(self.board_width):
            col = columns[i]
            if not col["has_enemy_front"]:
                continue

            irregular = False
            if i > 0 and columns[i - 1]["has_enemy_front"]:
                if (
                    abs(columns[i - 1]["enemy_front_proj"] - col["enemy_front_proj"])
                    >= 2
                ):
                    irregular = True
            if i + 1 < self.board_width and columns[i + 1]["has_enemy_front"]:
                if (
                    abs(columns[i + 1]["enemy_front_proj"] - col["enemy_front_proj"])
                    >= 2
                ):
                    irregular = True

            if irregular:
                targets.append(i)

        if not targets:
            fallback_col = -1
            closest = float("inf")
            for col in columns:
                if not col["has_enemy_front"]:
                    continue
                if col["enemy_front_proj"] < closest:
                    closest = col["enemy_front_proj"]
                    fallback_col = col["x"]
            if fallback_col != -1:
                targets.append(fallback_col)

        if not targets:
            return None

        best = None
        best_score = 0

        for m in moves:
            if not m["place_tile"]:
                continue

            score = 0
            for idx in targets:
                if abs(m["tx"] - idx) > 1:
                    continue

                target_col = columns[idx]
                if not target_col["has_enemy_front"]:
                    continue

                row_diff = abs(m["ty"] - target_col["enemy_front_row"])
                current_score = max(0, 80 - row_diff * 15)

                ahead = False
                if self.player_id == P2:
                    ahead = m["ty"] >= target_col["enemy_front_row"]
                else:
                    ahead = m["ty"] <= target_col["enemy_front_row"]

                if ahead:
                    current_score += 20

                score = max(score, current_score)

            if score == 0:
                continue

            if m["tile"] == TILE_GRAY:
                score += 25
            if m["tile"] == TILE_BLACK:
                score += 15

            if score > best_score:
                best_score = score
                best = m

        return best

    def _prioritize_lead_piece(self, game: ContrastGame, moves):
        """5. 先頭の駒を優先して進める"""
        columns = self._collect_column_info(game)
        best = None
        best_score = 0

        for m in moves:
            if m["place_tile"]:
                continue
            if self._row_progress(m) <= 0:
                continue
            if m["dx"] != m["sx"]:
                continue
            if m["sx"] != 0 and m["sx"] != self.board_width - 1:
                continue

            col = columns[m["sx"]]
            score = self._row_progress(m) * 110

            if col["has_friend"] and col["friend_row"] == m["sy"]:
                score += 30

            score += self._project_row(m["dy"]) * 5

            if score > best_score:
                best_score = score
                best = m

        return best

    def _outflank_straight_runner(self, game: ContrastGame, moves):
        """6. 直進してくる敵を側面から抜く"""
        columns = self._collect_column_info(game)

        closest_enemy = float("inf")
        for col in columns:
            if col["has_enemy_front"]:
                closest_enemy = min(closest_enemy, col["enemy_front_proj"])

        if closest_enemy == float("inf"):
            return None

        best = None
        best_score = 0

        for m in moves:
            if m["place_tile"]:
                continue
            if self._row_progress(m) <= 0:
                continue

            col = columns[m["sx"]]
            if not col["has_enemy_front"]:
                continue

            if col["enemy_front_proj"] > closest_enemy + 1:
                continue

            desired_row = col["enemy_front_row"] - self.direction

            after_gap = 0
            if self.player_id == P2:
                after_gap = max(0, col["enemy_front_row"] - m["dy"] - 1)
            else:
                after_gap = max(0, m["dy"] - col["enemy_front_row"] - 1)

            score = 100 - after_gap * 35
            if col["has_friend"] and col["friend_row"] == m["sy"]:
                score += 30

            dist_to_desired = abs(m["dy"] - desired_row)
            score += max(0, 40 - dist_to_desired * 15)

            if score > best_score:
                best_score = score
                best = m

        return best

    def _fallback_by_score(self, game: ContrastGame, moves):
        """7. 汎用的なスコアリングによる選択"""
        best = moves[0]
        best_score = -float("inf")

        for m in moves:
            score = self._row_progress(m) * 80

            dist = self._distance_to_nearest_empty_goal(
                game, m["dx"], m["dy"], self.player_id
            )
            score -= dist * 15

            if not m["place_tile"]:
                if game.pieces[m["dy"], m["dx"]] == self.opponent_id:
                    score += 50
            else:
                if m["tile"] == TILE_GRAY:
                    score += 30
                elif m["tile"] == TILE_BLACK:
                    score += 15
                if abs(m["tx"] - m["sx"]) <= 1:
                    score += 10

            if score > best_score:
                best_score = score
                best = m

        return best
