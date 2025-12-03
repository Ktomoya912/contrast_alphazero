# コントラスト (Contrast) - AIボードゲーム

5x5の盤面で、タイルの色によって移動方向が変化する戦略的な2人対戦ボードゲームです。

## ゲームルール

### 基本情報
- **盤面**: 5x5グリッド
- **プレイヤー**: 2人
- **各プレイヤーのコマ**: 5個
- **持ちタイル**: 黒タイル3枚、グレータイル1枚
- **ターン制**: 交互に1手ずつ行動、駒を動かした後にタイルを配置可能
- **移動制限**: 相手のコマがいるマスには移動不可、自分の駒が隣にある場合、その駒を飛び越えて移動可能
- **タイル制限**: タイルの再配置は認められない
- **勝利条件**: 相手の陣地（後列）に最初に到達

### タイルシステム
- **白タイル** (□): 縦横方向に1マス移動可能
- **黒タイル** (■): 斜め方向に1マス移動可能
- **グレータイル** (▦): 全8方向に1マス移動可能

### 初期配置
```
   0  1  2  3  4
0 [2][2][2][2][2]  ← Player 2 (ゴールはy=4)
1 [ ][ ][ ][ ][ ]
2 [ ][ ][ ][ ][ ]
3 [ ][ ][ ][ ][ ]
4 [1][1][1][1][1]  ← Player 1 (ゴールはy=0)
```

### ゲームの流れ
1. 各ターン、プレイヤーは以下を実行:
   - (オプション) 白タイルを黒/グレータイルに変更
   - 自分のコマを1つ移動
2. **相手のコマがいるマスには移動できません**
3. タイル配置数: 黒タイル×3、グレータイル×1 (各プレイヤー)

## ディレクトリ構造

```
contrast/
├── contrast_game.py      # ゲームエンジン本体
├── ai_model.py           # PyTorch CNNモデル
├── td_learning.py        # TD学習アルゴリズム
├── train_ai.py           # AI学習スクリプト
├── play_with_ai.py       # 人間 vs AI対戦
├── interactive_game.py   # 人間 vs 人間対戦
├── gui_game.py          # GUIバージョン (要tkinter)
├── models/              # 学習済みモデル
├── logs/                # 学習ログ
├── tests/               # テストコード
└── docs/                # ドキュメント
```

## セットアップ

### 必要要件
- Python 3.8+
- PyTorch 2.0+
- NumPy

### インストール (uv使用)
```bash
# uvで依存関係をインストール
uv sync

# または pip
pip install -r requirements.txt
```

## 使い方

### 1. AIを学習させる
```bash
# 基本的な学習 (1000エピソード)
uv run python train_ai.py --episodes 1000

# CUDA使用 (GPUがある場合)
uv run python train_ai.py --episodes 2000 --cuda

# 詳細オプション
uv run python train_ai.py --help
```

### 2. AIと対戦する
```bash
uv run python play_with_ai.py
```

### 3. 人間同士で対戦する
```bash
uv run python interactive_game.py
```

### 4. テストを実行
```bash
uv run python tests/test_rules.py
```

## AIについて

### アーキテクチャ
- **モデル**: 畳み込みニューラルネットワーク (CNN)
- **入力**: 12チャンネル × 5×5
  - コマ配置 (自分/相手)
  - タイル色 (白/黒/グレー)
  - 手番情報
  - ゴールまでの距離
  - 残りタイル数
- **出力**: 盤面評価値 [-1, 1]
- **パラメータ数**: 約380万

### 学習方法
- **TD学習** (Temporal Difference Learning)
- **セルフプレイ**: AIが自分自身と対戦
- **ε-greedy方策**: 探索と活用のバランス
- 詳細: [`docs/TRAINING_EXPLANATION.md`](docs/TRAINING_EXPLANATION.md)

## 主要な改善点

### v2 (最新)
- ✅ ルール修正: 相手のコマを取れない仕様に修正
- ✅ 入力情報の大幅改善 (5ch → 12ch)
- ✅ ネットワークサイズ拡大 (80万 → 380万パラメータ)
- ✅ 現在プレイヤー視点での正規化
- ✅ P1/P2の勝率バランス修正

## 参考資料

- ゲームルール詳細: [`docs/README.md`](docs/README.md)
- AI学習の仕組み: [`docs/TRAINING_EXPLANATION.md`](docs/TRAINING_EXPLANATION.md)
- API仕様: [`docs/AI_README.md`](docs/AI_README.md)

## ライセンス

MIT License

## 作者

iceto
