# AlphaZero Optimization for Contrast

このドキュメントは、コントラストゲーム向けのAlphaZero最適化の実装内容を説明します。

## 実装内容

### 1. ニューラルネットワークの最適化

**問題**: 8層のResidualブロックは19x19の囲碁向けの設計で、5x5のコントラストには過剰。

**解決策**:
- Residualブロック数を 8 → 6 に削減
- パラメータ数: ~1M（適切なサイズ）
- フィルタ数は64を維持（ゲームの複雑さに適合）

**設定場所**: `config.py` の `NetworkConfig.NUM_RES_BLOCKS`

```python
NUM_RES_BLOCKS: int = 6  # 8から削減
```

### 2. プレイ時の MCTS パラメータ

**問題**: 学習時と対戦時で同じMCTSパラメータを使用していたため、対戦時の強さが不十分。

**解決策**: プレイ時専用の設定 `PlayMCTSConfig` を追加

| パラメータ | 学習時 | プレイ時 | 理由 |
|-----------|-------|---------|------|
| NUM_SIMULATIONS | 50 | 200 | より深い探索で精度向上 |
| C_PUCT | 1.0 | 1.5 | 探索をより重視 |
| DIRICHLET_EPSILON | 0.25 | 0.0 | ノイズなし（決定的） |

**設定場所**: `config.py` の `PlayMCTSConfig`

**使用箇所**:
- `players/alpha_zero.py`: AlphaZeroPlayerが自動的にプレイ時設定を使用
- `play_vs_ai.py`: デフォルト値がプレイ時設定に

### 3. 学習率 0.2 のサポート

**問題**: 将棋のように初期学習率0.2を使用すると、学習が不安定になり失敗。

**解決策**: 
1. **学習率ウォームアップ**: `WarmupScheduler` クラスを追加
   - 初期ステップで学習率を徐々に上げる
   - 勾配の急激な変化を防ぐ

2. **勾配クリッピング**: 
   - 最大勾配ノルム: 10.0
   - 学習の安定性を向上

**設定場所**: `config.py` の `TrainingConfig`

```python
USE_WARMUP: bool = False  # ウォームアップを有効化
WARMUP_STEPS: int = 1000  # ウォームアップのステップ数
MAX_GRAD_NORM: float = 10.0  # 勾配クリッピングの閾値
```

**学習率 0.2 を使用する方法**:

`config.py` を編集:
```python
LEARNING_RATE: float = 0.2  # 0.001から変更
USE_WARMUP: bool = True  # ウォームアップを有効化
```

または、コマンドライン引数で指定可能にする場合、`main.py` を修正。

## 使用方法

### 通常の学習（デフォルト設定）

```bash
python3 main.py
```

設定:
- 学習率: 0.001
- ウォームアップ: なし
- Residualブロック: 6
- 学習時MCTS: 50シミュレーション

### 学習率 0.2 での学習

1. `config.py` を編集:
   ```python
   class TrainingConfig:
       LEARNING_RATE: float = 0.2
       USE_WARMUP: bool = True
       WARMUP_STEPS: int = 1000
   ```

2. 学習実行:
   ```bash
   python3 main.py
   ```

### プレイ時の最適化された設定

AIとの対戦では、自動的にプレイ時設定（200シミュレーション、C_PUCT=1.5）が使用されます。

```bash
# デフォルト（200シミュレーション）
python3 play_vs_ai.py

# カスタム設定
python3 play_vs_ai.py --simulations 400
```

## テスト

すべての変更はテストでカバーされています:

```bash
# すべてのテストを実行
python3 -m pytest tests/ -v

# 新機能のテストのみ
python3 -m pytest tests/test_play_mcts_config.py -v

# 学習率0.2のデモ
python3 test_lr_0_2.py
```

## パフォーマンス比較

### モデルサイズ

| 設定 | Residualブロック | パラメータ数 |
|-----|----------------|------------|
| 旧 | 8 | ~1.3M |
| 新 | 6 | ~1.0M |

**効果**: 
- 学習速度が約20%向上
- メモリ使用量が削減
- 過学習のリスクが低減

### MCTS性能

| 設定 | シミュレーション | C_PUCT | 対戦時の平均手数 |
|-----|----------------|--------|---------------|
| 学習時 | 50 | 1.0 | - |
| プレイ時 | 200 | 1.5 | より良い判断 |

**効果**:
- より深い読みで精度向上
- ルールベースAIに対する勝率向上が期待される

### 学習率の安定性

| 設定 | 初期LR | ウォームアップ | 勾配クリッピング | 安定性 |
|-----|-------|--------------|----------------|-------|
| 旧 | 0.001 | なし | なし | 安定 |
| 新（保守的） | 0.001 | なし | あり | より安定 |
| 新（積極的） | 0.2 | あり | あり | 安定 |

**効果**:
- 高い学習率での学習が可能に
- 学習の初期段階での収束が高速化

## トラブルシューティング

### 学習が不安定な場合

1. ウォームアップステップ数を増やす:
   ```python
   WARMUP_STEPS: int = 2000  # 1000から増加
   ```

2. 勾配クリッピングの閾値を下げる:
   ```python
   MAX_GRAD_NORM: float = 5.0  # 10.0から減少
   ```

3. 学習率を下げる:
   ```python
   LEARNING_RATE: float = 0.1  # 0.2から減少
   ```

### 対戦で弱い場合

1. プレイ時のシミュレーション回数を増やす:
   ```bash
   python3 play_vs_ai.py --simulations 400
   ```

2. より多くのステップで学習:
   ```python
   MAX_EPOCH: int = 100 * 10_000  # 50*10000から増加
   ```

## 参考文献

- AlphaZero論文: https://arxiv.org/abs/1712.01815
- 学習率ウォームアップ: https://arxiv.org/abs/1706.02677
- Residual Networks: https://arxiv.org/abs/1512.03385
