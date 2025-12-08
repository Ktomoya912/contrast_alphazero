from contrast_game import TILE_BLACK, TILE_GRAY, TILE_WHITE, ContrastGame

from .base import BasePlayer


class HumanPlayer(BasePlayer):
    def get_action(self, game: ContrastGame):
        """人間から行動を入力

        入力形式: <移動前>,<移動後> <配置座標><タイルカラー>
        例:
          b1,b2 b3g  (b1からb2へ移動、b3にグレータイルを配置)
          a5,a4 b1b  (a5からa4へ移動、b1に黒タイルを配置)
          c5,c4      (c5からc4へ移動、タイル配置なし)
        """
        print(f"\nあなたの番です (プレイヤー{game.current_player})")
        print("入力形式: <移動前>,<移動後> <配置座標><タイルカラー>")
        print("例: b1,b2 b3g (b1→b2へ移動、b3にグレータイル配置)")
        print("    c5,c4 (タイル配置なし)")

        p_idx = game.current_player - 1
        has_black = game.tile_counts[p_idx, 0] > 0
        has_gray = game.tile_counts[p_idx, 1] > 0
        print(
            f"持ちタイル: 黒(b)={game.tile_counts[p_idx, 0]}, グレー(g)={game.tile_counts[p_idx, 1]}"
        )

        while True:
            try:
                user_input = input("\n行動を入力: ").strip()

                if not user_input:
                    print("エラー: 入力が空です")
                    continue

                # スペースで分割: [移動部分, タイル部分(optional)]
                parts = user_input.split()

                if len(parts) == 0:
                    print("エラー: 入力が空です")
                    continue

                # 移動部分をパース
                move_part = parts[0]
                if "," not in move_part:
                    print(
                        "エラー: 移動は'<移動前>,<移動後>'の形式で入力してください (例: b1,b2)"
                    )
                    continue

                from_pos, to_pos = move_part.split(",")

                # 座標変換
                fx, fy = self.parse_position(from_pos)
                tx, ty = self.parse_position(to_pos)

                # 移動元の駒チェック
                if game.pieces[fy, fx] != game.current_player:
                    print(f"エラー: {from_pos}に自分の駒がありません")
                    continue

                # 移動先の有効性チェック
                valid_moves = game.get_valid_moves(fx, fy)
                if not valid_moves:
                    print(f"エラー: {from_pos}の駒は移動できません")
                    continue

                if (tx, ty) not in valid_moves:
                    # 移動可能な場所を新形式で表示
                    valid_pos_str = [
                        self.format_position(vx, vy) for vx, vy in valid_moves
                    ]
                    print(
                        f"エラー: {to_pos}には移動できません。移動可能: {', '.join(valid_pos_str)}"
                    )
                    continue

                # タイル配置部分をパース
                tile_type = 0
                tile_x, tile_y = 0, 0

                if len(parts) >= 2:
                    tile_part = parts[1]

                    if len(tile_part) < 3:
                        print(
                            "エラー: タイル配置は'<座標><色>'の形式で入力してください (例: b3g)"
                        )
                        continue

                    tile_pos = tile_part[:2]
                    tile_color = tile_part[2].lower()

                    if tile_color not in ["b", "g"]:
                        print(
                            "エラー: タイルの色はb(黒)またはg(グレー)を指定してください"
                        )
                        continue

                    # タイルの色を決定
                    if tile_color == "b":
                        if not has_black:
                            print("エラー: 黒タイルの持ち駒がありません")
                            continue
                        tile_type = TILE_BLACK
                    else:  # 'g'
                        if not has_gray:
                            print("エラー: グレータイルの持ち駒がありません")
                            continue
                        tile_type = TILE_GRAY

                    # タイル配置座標を変換
                    tile_x, tile_y = self.parse_position(tile_pos)

                    # タイル配置の有効性チェック
                    if game.tiles[tile_y, tile_x] != TILE_WHITE:
                        print(f"エラー: {tile_pos}は白タイルではありません")
                        continue

                    if tile_x == tx and tile_y == ty:
                        print("エラー: 移動先にはタイルを配置できません")
                        continue

                    if game.pieces[tile_y, tile_x] != 0 and not (
                        tile_x == fx and tile_y == fy
                    ):
                        print(f"エラー: {tile_pos}には駒があります（移動元以外）")
                        continue

                # アクションハッシュを生成
                move_idx = (fy * 5 + fx) * 25 + (ty * 5 + tx)

                if tile_type == 0:
                    tile_idx = 0
                elif tile_type == TILE_BLACK:
                    tile_idx = 1 + (tile_y * 5 + tile_x)
                else:  # TILE_GRAY
                    tile_idx = 26 + (tile_y * 5 + tile_x)

                action_hash = move_idx * 51 + tile_idx
                return action_hash

            except ValueError as e:
                print(f"エラー: {e}")
                continue
            except KeyboardInterrupt:
                raise

    def parse_position(self, pos_str):
        """位置文字列(例: 'b3')を内部座標(x, y)に変換

        Args:
            pos_str: 'a1'-'e5'形式の文字列
        Returns:
            (x, y): 内部座標 (0-4, 0-4)
        """
        if len(pos_str) != 2:
            raise ValueError("座標は2文字で指定してください (例: b3)")

        col = pos_str[0].lower()
        row = pos_str[1]

        if col not in "abcde":
            raise ValueError("列はa-eで指定してください")
        if row not in "12345":
            raise ValueError("行は1-5で指定してください")

        x = ord(col) - ord("a")  # a=0, b=1, ..., e=4
        y = 5 - int(row)  # 1=4, 2=3, 3=2, 4=1, 5=0 (下から上)

        return x, y
