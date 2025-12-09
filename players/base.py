class BasePlayer:
    def __init__(self, player_id: int):
        self.player_id = player_id

    def get_action(self, game) -> int:
        """ゲームの状態に基づいて行動を選択するメソッド

        Args:
            game: 現在のゲーム状態を表すContrastGameオブジェクト

        Returns:
            選択された行動のインデックス (int)
        """
        raise NotImplementedError("get_actionメソッドはサブクラスで実装してください")

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
