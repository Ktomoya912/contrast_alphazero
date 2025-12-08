"""Contrastゲームの実装

このモジュールは、Contrastボードゲームのルールとロジックを実装しています。
"""

import copy
from collections import deque
from typing import List, Tuple

import numpy as np

from config import game_config

# --- Game Constants ---
# Players
P1 = 1
P2 = 2
OPPONENT = {P1: P2, P2: P1}

# Tiles
TILE_WHITE = 0
TILE_BLACK = 1
TILE_GRAY = 2

# Action Space (from config)
BOARD_SIZE = game_config.BOARD_SIZE
NUM_BOARD_POSITIONS = game_config.NUM_BOARD_POSITIONS
NUM_MOVES = game_config.NUM_MOVES
NUM_TILE_ACTIONS = game_config.NUM_TILE_ACTIONS
NUM_TILES = NUM_TILE_ACTIONS

# Initial resources (from config)
INITIAL_BLACK_TILES = game_config.INITIAL_BLACK_TILES
INITIAL_GRAY_TILES = game_config.INITIAL_GRAY_TILES

# History size (from config)
HISTORY_SIZE = game_config.HISTORY_SIZE

# Directions (Pre-computed)
# 0: White (Orthogonal), 1: Black (Diagonal), 2: Gray (All)
DIRS = [
    np.array([(0, -1), (0, 1), (-1, 0), (1, 0)], dtype=np.int8),  # White
    np.array([(-1, -1), (1, -1), (-1, 1), (1, 1)], dtype=np.int8),  # Black
    np.array(  # Gray
        [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (1, -1), (-1, 1), (1, 1)],
        dtype=np.int8,
    ),
]


class ContrastGame:
    """Contrastゲームの状態管理クラス

    このクラスは、ゲームの盤面状態、駒の配置、タイルの管理、
    合法手の生成、ゲーム進行などを担当します。
    """

    ACTION_SIZE_TILE = NUM_TILES

    def __init__(self, board_size: int = BOARD_SIZE) -> None:
        """ゲームの初期化

        Args:
            board_size: 盤面のサイズ（デフォルト: 5）
        """
        self.size = board_size

        # 盤面状態 (int8 array)
        # pieces: 0=Empty, 1=P1, 2=P2
        # tiles: 0=White, 1=Black, 2=Gray
        self.pieces = np.zeros((self.size, self.size), dtype=np.int8)
        self.tiles = np.zeros((self.size, self.size), dtype=np.int8)

        # 持ちタイル数 [Player-1][TileType] (Player index is 0 or 1 for array access)
        # index 0: P1, index 1: P2
        # index 0: Black, index 1: Gray
        self.tile_counts = np.array(
            [
                [INITIAL_BLACK_TILES, INITIAL_GRAY_TILES],
                [INITIAL_BLACK_TILES, INITIAL_GRAY_TILES],
            ],
            dtype=np.int8,
        )

        self.current_player = P1
        self.game_over = False
        self.winner = 0
        self.move_count = 0

        # 履歴管理 (HISTORY_SIZE手分)
        # 高速化のため、dictではなくtuple(pieces, tiles, tile_counts)を保存
        self.history: deque = deque(maxlen=HISTORY_SIZE)

        # 盤面繰り返し判定用（50手以降の盤面ハッシュを記録）
        self.position_history: dict = {}  # {board_hash: count}
        self.repetition_threshold = 5  # 同じ盤面が10回出現したら引き分け

        self.setup_initial_position()

    def setup_initial_position(self) -> None:
        """初期配置をセットアップ

        P1は下段(y=4)、P2は上段(y=0)に配置されます。
        """
        self.pieces.fill(0)
        self.tiles.fill(0)
        # P1 at y=4, P2 at y=0
        self.pieces[BOARD_SIZE - 1, :] = P1
        self.pieces[0, :] = P2

        self.tile_counts = np.array(
            [
                [INITIAL_BLACK_TILES, INITIAL_GRAY_TILES],
                [INITIAL_BLACK_TILES, INITIAL_GRAY_TILES],
            ],
            dtype=np.int8,
        )
        self.current_player = P1
        self.move_count = 0
        self.game_over = False
        self.winner = 0

        self.history.clear()
        self._save_history()

    def _save_history(self) -> None:
        """現在の状態を履歴に追加

        状態のディープコピーを作成して履歴に追加します。
        """
        self.history.appendleft(
            (self.pieces.copy(), self.tiles.copy(), self.tile_counts.copy())
        )

    def copy(self) -> "ContrastGame":
        """シミュレーション用の軽量コピー

        Returns:
            ContrastGame: ゲーム状態のコピー
        """
        new_game = ContrastGame(self.size)
        new_game.pieces = self.pieces.copy()
        new_game.tiles = self.tiles.copy()
        new_game.tile_counts = self.tile_counts.copy()
        new_game.current_player = self.current_player
        new_game.game_over = self.game_over
        new_game.winner = self.winner
        new_game.move_count = self.move_count
        # historyは重いので、必要ならコピーするが、
        # MCTSのSimulation内だけで完結するなら直近の状態だけで良い場合も多い。
        # ここでは正確性のため浅くコピーしておく
        new_game.history = copy.copy(self.history)
        return new_game

    def get_valid_moves(self, x: int, y: int) -> List[Tuple[int, int]]:
        """指定された位置の駒の合法的な移動先を取得

        Args:
            x: X座標 (0-4)
            y: Y座標 (0-4)

        Returns:
            合法的な移動先座標のリスト [(x, y), ...]
        """
        # 自分の駒でなければ移動不可
        if self.pieces[y, x] != self.current_player:
            return []

        moves = []
        tile_type = self.tiles[y, x]
        directions = DIRS[tile_type]

        # Python loop is faster than numpy checking for small fixed steps
        # But we can use direct array access
        p_arr = self.pieces

        for dx, dy in directions:
            nx, ny = x + dx, y + dy

            # Bounds check loop
            while 0 <= nx < 5 and 0 <= ny < 5:
                target = p_arr[ny, nx]

                if target == 0:
                    # Empty -> Valid Move
                    moves.append((nx, ny))
                    break  # Stop sliding
                elif target == self.current_player:
                    # Friend -> Jump over
                    nx += dx
                    ny += dy
                    continue
                else:
                    # Enemy -> Blocked
                    break

        return moves

    def get_all_legal_actions(self) -> List[int]:
        """現在のプレイヤーの全合法手を取得

        Returns:
            合法手のアクションハッシュのリスト
            各ハッシュは move_idx * NUM_TILES + tile_idx の形式
        """
        if self.game_over:
            return []

        legal_hashes = []  # intのリスト

        # 1. 自分の駒の位置
        py, px = np.where(self.pieces == self.current_player)

        # 2. 持ちタイル情報
        p_idx = self.current_player - 1
        has_black = self.tile_counts[p_idx, 0] > 0
        has_gray = self.tile_counts[p_idx, 1] > 0

        # 3. 白タイルの場所 (配置候補)
        # 事前に座標リスト化しておく
        white_ty, white_tx = np.where(self.tiles == TILE_WHITE)
        white_spots = list(zip(white_tx, white_ty))

        # 定数のローカル変数化 (ループ内アクセス高速化)
        SIZE_TILE = self.ACTION_SIZE_TILE
        pieces = self.pieces  # numpy array access

        for i in range(len(px)):
            cx, cy = px[i], py[i]

            # 移動先取得
            moves = self.get_valid_moves(cx, cy)

            for mx, my in moves:
                # Base Move Index (0~624)
                move_idx = (cy * 5 + cx) * 25 + (my * 5 + mx)

                # Base Hash (move_idx * 51)
                base_hash = move_idx * SIZE_TILE

                # A. Move Only (Tile=0) -> base_hash + 0
                legal_hashes.append(base_hash)

                if not (has_black or has_gray):
                    continue

                # B. Move + Place Tile
                for tx, ty in white_spots:
                    # 移動先には置けない
                    if tx == mx and ty == my:
                        continue

                    # 既存のコマチェック (移動元のコマは無視してOK)
                    if pieces[ty, tx] != 0 and not (tx == cx and ty == cy):
                        continue

                    # Note: ここで append(int) を行うのは append(tuple) より高速です
                    if has_black:
                        # Tile Idx: 1 + (ty*5 + tx)
                        t_idx = 1 + (ty * 5 + tx)
                        legal_hashes.append(base_hash + t_idx)

                    if has_gray:
                        # Tile Idx: 26 + (ty*5 + tx)
                        t_idx = 26 + (ty * 5 + tx)
                        legal_hashes.append(base_hash + t_idx)

        return legal_hashes

    # --- Step & Update ---

    def step(self, action_hash: int | None) -> Tuple[bool, int]:
        """アクションを実行してゲーム状態を更新

        Args:
            action_hash: エンコードされたアクションハッシュ

        Returns:
            (game_over, winner): ゲーム終了フラグと勝者ID
        """
        if action_hash is None or self.game_over:
            return self.game_over, self.winner
        # デコード処理 (高速な整数演算のみ)
        move_idx = action_hash // self.ACTION_SIZE_TILE
        tile_idx = action_hash % self.ACTION_SIZE_TILE

        # Parse Move
        from_idx = move_idx // 25
        to_idx = move_idx % 25
        fx, fy = from_idx % 5, from_idx // 5
        tx, ty = to_idx % 5, to_idx // 5

        # Parse Tile
        place_tile = False
        t_color = 0
        t_x, t_y = 0, 0

        if tile_idx > 0:
            place_tile = True
            if tile_idx <= 25:
                # Black
                t_color = TILE_BLACK
                idx = tile_idx - 1
            else:
                # Gray
                t_color = TILE_GRAY
                idx = tile_idx - 26

            t_x, t_y = idx % 5, idx // 5

        # --- Execute Move (In-place) ---
        self.pieces[ty, tx] = self.pieces[fy, fx]
        self.pieces[fy, fx] = 0

        if place_tile:
            self.tiles[t_y, t_x] = t_color
            p_idx = self.current_player - 1
            c_idx = 0 if t_color == TILE_BLACK else 1
            self.tile_counts[p_idx, c_idx] -= 1

        self._check_win_fast()

        # 手番を交代する前に、次のプレイヤーの合法手をチェック
        self.current_player = OPPONENT[self.current_player]

        # 敗北条件チェック: 次のプレイヤーに合法手がない場合
        if not self.game_over:
            legal_actions = self.get_all_legal_actions()
            if len(legal_actions) == 0:
                # 合法手がないプレイヤーが敗北
                self.game_over = True
                self.winner = OPPONENT[self.current_player]

        self.move_count += 1
        self._save_history()

        # 盤面繰り返し判定（50手以降のみ）
        if not self.game_over and self.move_count >= 50:
            board_hash = self._get_board_hash()
            self.position_history[board_hash] = (
                self.position_history.get(board_hash, 0) + 1
            )

            if self.position_history[board_hash] >= self.repetition_threshold:
                # 同じ盤面が規定回数繰り返されたら引き分け
                self.game_over = True
                self.winner = 0

        return self.game_over, self.winner

    def _check_win_fast(self) -> None:
        """勝利条件を高速チェック

        P1が上段(y=0)に到達、またはP2が下段(y=4)に到達で勝利。
        """
        # Numpy check is fast
        if np.any(self.pieces[0, :] == P1):
            self.game_over = True
            self.winner = P1
        elif np.any(self.pieces[4, :] == P2):
            self.game_over = True
            self.winner = P2

    def _get_board_hash(self) -> int:
        """盤面のハッシュ値を計算（繰り返し判定用）

        Returns:
            盤面状態を表すハッシュ値
        """
        # pieces, tiles, current_playerを組み合わせてハッシュ化
        # tobytes()は高速で、同じ配列なら同じバイト列を返す
        return hash((self.pieces.tobytes(), self.tiles.tobytes(), self.current_player))

    # --- Encoding ---

    def encode_state(self) -> np.ndarray:
        """ニューラルネットワーク入力用の状態テンソルを生成

        P2の場合は盤面を180度回転させ、P1視点に正規化します。

        Returns:
            (90, 5, 5)の入力テンソル
            - 0-7: 現在プレイヤーの駒位置 (履歴)
            - 8-15: 相手の駒位置 (履歴)
            - 16-23: 黒タイル (履歴)
            - 24-31: グレータイル (履歴)
            - 56-63: 自分の黒タイル数 (履歴)
            - 64-71: 自分のグレータイル数 (履歴)
            - 72-79: 相手の黒タイル数 (履歴)
            - 80-87: 相手のグレータイル数 (履歴)
            - 88: 現在のプレイヤー識別子
            - 89: 手数
        """
        input_tensor = np.zeros((90, 5, 5), dtype=np.float32)

        # 履歴取得 (足りない分はパディング)
        hist_len = len(self.history)

        current_pid = self.current_player
        opp_pid = OPPONENT[current_pid]

        # Player Index for tile counts
        my_idx = current_pid - 1
        opp_idx = opp_pid - 1

        # 【追加】P2なら盤面を回転させるフラグ
        should_flip = current_pid == P2

        for i in range(8):
            if i < hist_len:
                p_grid, t_grid, t_counts = self.history[i]
            else:
                p_grid, t_grid, t_counts = self.history[-1]

            # 【追加】P2視点なら盤面を180度回転
            if should_flip:
                p_grid = np.rot90(p_grid, 2)
                t_grid = np.rot90(t_grid, 2)

            # Plane offsets
            input_tensor[i] = p_grid == current_pid
            input_tensor[8 + i] = p_grid == opp_pid
            input_tensor[16 + i] = t_grid == TILE_BLACK
            input_tensor[24 + i] = t_grid == TILE_GRAY

            # Tile Counts (値は回転不要、埋めるだけ)
            input_tensor[56 + i].fill(t_counts[my_idx, 0] / 3.0)
            input_tensor[64 + i].fill(t_counts[my_idx, 1] / 1.0)
            input_tensor[72 + i].fill(t_counts[opp_idx, 0] / 3.0)
            input_tensor[80 + i].fill(t_counts[opp_idx, 1] / 1.0)

        # 88: Color (P2の場合は回転しているので、常にP1視点として扱えるため常に1でも良いが、
        # AlphaZeroの慣例的には手番プレーヤーIDを入れることもある。
        # ここでは元の実装通り「自分=1」とする)
        input_tensor[88].fill(1.0)

        # 89: Move Count
        input_tensor[89].fill(self.move_count / game_config.MAX_STEPS_PER_GAME)

        return input_tensor


def encode_action(move_idx: int, tile_idx: int) -> int:
    """
    (move_idx, tile_idx) -> unique hash (int)
    Range: 0 ~ 31874 (625 * 51 - 1)
    """
    return move_idx * NUM_TILES + tile_idx


def decode_action(action_hash: int) -> Tuple[int, int]:
    """
    unique hash (int) -> (move_idx, tile_idx)
    """
    move_idx = action_hash // NUM_TILES
    tile_idx = action_hash % NUM_TILES
    return move_idx, tile_idx


def flip_location(idx: int) -> int:
    """盤面インデックス(0-24)を180度回転させる: i -> 24-i"""
    return 24 - idx


def flip_action(action_hash: int) -> int:
    """アクションハッシュを180度回転した視点に変換する"""
    move_idx = action_hash // NUM_TILES
    tile_idx = action_hash % NUM_TILES

    # Moveの変換
    from_idx = move_idx // 25
    to_idx = move_idx % 25

    new_from = flip_location(from_idx)
    new_to = flip_location(to_idx)
    new_move_idx = new_from * 25 + new_to

    # Tileの変換
    new_tile_idx = 0
    if tile_idx > 0:
        if tile_idx <= 25:  # Black
            pos = tile_idx - 1
            new_pos = flip_location(pos)
            new_tile_idx = new_pos + 1
        else:  # Gray
            pos = tile_idx - 26
            new_pos = flip_location(pos)
            new_tile_idx = new_pos + 26

    return new_move_idx * NUM_TILES + new_tile_idx
