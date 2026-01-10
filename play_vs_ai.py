import argparse

from config import mcts_config, path_config, play_mcts_config
from contrast_game import (
    P1,
    P2,
    TILE_BLACK,
    TILE_GRAY,
    TILE_WHITE,
    ContrastGame,
    decode_action,
)
from logger import get_logger, setup_logger
from players import (
    AlphaZeroPlayer,
    BasePlayer,
    HumanPlayer,
    RandomPlayer,
    RuleBasedPlayer,
)

logger = get_logger(__name__)


class HumanVsAI:
    def __init__(
        self, model_path, num_simulations=None, player1_type="human", player2_type="ai"
    ):
        """
        Args:
            model_path: 学習済みモデルのパス
            num_simulations: MCTSのシミュレーション回数 (Noneの場合はplay_mcts_configから取得)
            player1_type: プレイヤー1のタイプ ("human", "ai", "random", "rule")
            player2_type: プレイヤー2のタイプ ("human", "ai", "random", "rule")
        """
        # config.pyからデフォルト値を取得（プレイ時の設定を使用）
        if num_simulations is None:
            num_simulations = play_mcts_config.NUM_SIMULATIONS

        self.player1_type = player1_type
        self.player2_type = player2_type
        self.num_simulations = num_simulations
        self.action_history = []

        # ゲーム初期化
        self.game = ContrastGame()

        # プレイヤーの初期化
        self.players: dict[int, BasePlayer] = {}

        for player_id, player_type in [(P1, player1_type), (P2, player2_type)]:
            if player_type == "human":
                self.players[player_id] = HumanPlayer(player_id)
            elif player_type == "ai":
                self.players[player_id] = AlphaZeroPlayer(
                    player_id, model_path, num_simulations
                )
            elif player_type == "random":
                self.players[player_id] = RandomPlayer(player_id)
            elif player_type == "rule":
                self.players[player_id] = RuleBasedPlayer(player_id)
            else:
                raise ValueError(f"Unknown player type: {player_type}")

        logger.info(f"プレイヤー1: {player1_type}, プレイヤー2: {player2_type}")

    def display_board(self):
        """盤面を表示"""
        print("\n" + "=" * 50)
        print("現在の盤面:")
        print("=" * 50)

        # タイルの表示
        tile_symbols = {TILE_WHITE: "□", TILE_BLACK: "■", TILE_GRAY: "▦"}

        # 列ラベル (a-e)
        print("   ", end="")
        for x in range(5):
            print(f" {chr(ord('a') + x)} ", end="")
        print()

        # 行は5から1へ（下から上）
        for y in range(5):
            row_label = 5 - y  # 5, 4, 3, 2, 1
            print(f" {row_label} ", end="")
            for x in range(5):
                piece = self.game.pieces[y, x]
                tile = self.game.tiles[y, x]

                if piece == P1:
                    symbol = f"[1{tile_symbols[tile]}]"
                elif piece == P2:
                    symbol = f"[2{tile_symbols[tile]}]"
                else:
                    symbol = f" {tile_symbols[tile]} "

                print(symbol, end="")
            print()

        print("\n持ちタイル:")
        print(
            f"  プレイヤー1: 黒={self.game.tile_counts[0, 0]}, グレー={self.game.tile_counts[0, 1]}"
        )
        print(
            f"  プレイヤー2: 黒={self.game.tile_counts[1, 0]}, グレー={self.game.tile_counts[1, 1]}"
        )
        print(f"\n手数: {self.game.move_count}")
        print("=" * 50)

    def format_position(self, x, y):
        """内部座標(x, y)を位置文字列に変換

        Args:
            x, y: 内部座標 (0-4, 0-4)
        Returns:
            'a1'-'e5'形式の文字列
        """
        col = chr(ord("a") + x)
        row = 5 - y
        return f"{col}{row}"

    def get_action_for_player(self, player_id):
        """プレイヤーから行動を取得"""
        player = self.players[player_id]
        unpacked = player.get_action(self.game)
        if isinstance(unpacked, tuple):
            action, value = unpacked
        else:
            action = unpacked
            value = None
        self.print_hash(action, player_id, value)

        if action is not None:
            self.action_history.append((action, player_id, value))

        return action

    def play(self):
        """ゲームをプレイ"""
        logger.info(
            f"ゲーム開始: プレイヤー1={self.player1_type}, プレイヤー2={self.player2_type}"
        )

        self.display_board()

        while not self.game.game_over:
            action = self.get_action_for_player(self.game.current_player)

            if action is None:
                logger.error("無効なアクションです")
                break

            # アクション実行
            done, winner = self.game.step(action)

            self.display_board()

            if done:
                break

        # 結果表示
        print("\n" + "=" * 50)
        print("ゲーム終了!")
        print("=" * 50)

        if self.game.winner == 0:
            print("引き分けです")
        elif self.game.winner == P1:
            print(f"🎉 プレイヤー1 ({self.player1_type}) の勝利です！")
        else:
            print(f"🎉 プレイヤー2 ({self.player2_type}) の勝利です！")

        print(f"総手数: {self.game.move_count}")
        print("=" * 50)
        print("行動履歴:")
        for idx, (action, player, value) in enumerate(self.action_history):
            print(f"手数 {idx + 1}: ", end="")
            self.print_hash(action, player, value)

    def print_hash(self, action: int, player_id: int, value: float | None):
        move_idx, tile_idx = decode_action(action)
        from_idx = move_idx // 25
        to_idx = move_idx % 25
        fx, fy = from_idx % 5, from_idx // 5
        tx, ty = to_idx % 5, to_idx // 5

        from_pos = self.format_position(fx, fy)
        to_pos = self.format_position(tx, ty)
        action_str = f"プレイヤー{player_id} の行動: {from_pos},{to_pos}"

        if tile_idx > 0:
            if tile_idx <= 25:
                tile_color = "b"
                tile_type_jp = "黒タイル"
                idx_tile = tile_idx - 1
            else:
                tile_color = "g"
                tile_type_jp = "グレータイル"
                idx_tile = tile_idx - 26

            tile_x, tile_y = idx_tile % 5, idx_tile // 5
            tile_pos = self.format_position(tile_x, tile_y)
            action_str += f" {tile_pos}{tile_color} ({tile_type_jp})"

        if value is not None:
            action_str += f" 評価値: {value:.3f}"

        print(action_str)


def main():
    parser = argparse.ArgumentParser(description="学習済みモデルと対戦")
    parser.add_argument(
        "--model",
        type=str,
        default=path_config.FINAL_MODEL_PATH,
        help=f"学習済みモデルのパス (デフォルト: {path_config.FINAL_MODEL_PATH})",
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=play_mcts_config.NUM_SIMULATIONS,
        help=f"MCTSのシミュレーション回数 (デフォルト: {play_mcts_config.NUM_SIMULATIONS})",
    )
    parser.add_argument(
        "--player1",
        type=str,
        choices=["human", "ai", "random", "rule"],
        default="human",
        help="プレイヤー1のタイプ (human/ai/random/rule, デフォルト: human)",
    )
    parser.add_argument(
        "--player2",
        type=str,
        choices=["human", "ai", "random", "rule"],
        default="ai",
        help="プレイヤー2のタイプ (human/ai/random/rule, デフォルト: ai)",
    )

    args = parser.parse_args()

    # ロギング設定
    setup_logger()

    # ゲーム開始
    game = HumanVsAI(
        model_path=args.model,
        num_simulations=args.simulations,
        player1_type=args.player1,
        player2_type=args.player2,
    )

    try:
        game.play()
    except KeyboardInterrupt:
        print("\n\nゲームを中断しました")
        logger.info("Game interrupted by user")


if __name__ == "__main__":
    main()
