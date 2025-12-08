"""
ContrastDualPolicyNet と関連関数の網羅的なテストコード

テストカバレッジ:
- ResidualBlockの動作
- ContrastDualPolicyNetの初期化
- forward()の出力形状検証
- loss_function()の動作確認
- バッチ処理の検証
- GPU/CPU互換性テスト
- モデルの保存/読み込みテスト
- エッジケース
"""

import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from model import ContrastDualPolicyNet, ResidualBlock, loss_function


class TestResidualBlock(unittest.TestCase):
    """ResidualBlockのテスト"""

    def test_residual_block_initialization(self):
        """ResidualBlockが正しく初期化されるか確認"""
        block = ResidualBlock(channels=64)

        self.assertIsInstance(block.conv1, nn.Conv2d)
        self.assertIsInstance(block.bn1, nn.BatchNorm2d)
        self.assertIsInstance(block.conv2, nn.Conv2d)
        self.assertIsInstance(block.bn2, nn.BatchNorm2d)

    def test_residual_block_forward_shape(self):
        """ResidualBlockのforward()が正しい形状を返すか確認"""
        block = ResidualBlock(channels=64)
        block.eval()

        x = torch.randn(2, 64, 5, 5)
        output = block(x)

        # 入力と同じ形状が返される
        self.assertEqual(output.shape, (2, 64, 5, 5))

    def test_residual_block_preserves_shape(self):
        """ResidualBlockが入力形状を保持するか確認"""
        block = ResidualBlock(channels=32)
        block.eval()

        batch_size = 4
        channels = 32
        height = 5
        width = 5

        x = torch.randn(batch_size, channels, height, width)
        output = block(x)

        self.assertEqual(output.shape, x.shape)


class TestContrastDualPolicyNetInitialization(unittest.TestCase):
    """ContrastDualPolicyNetの初期化テスト"""

    def test_model_initialization_default(self):
        """デフォルトパラメータで初期化できるか確認"""
        model = ContrastDualPolicyNet()

        self.assertIsNotNone(model)
        self.assertEqual(model.board_size, 5)
        self.assertEqual(model.move_size, 625)
        self.assertEqual(model.tile_size, 51)

    def test_model_initialization_custom_parameters(self):
        """カスタムパラメータで初期化できるか確認"""
        model = ContrastDualPolicyNet(
            input_channels=100, board_size=5, num_res_blocks=6, num_filters=128
        )

        self.assertIsNotNone(model)
        self.assertEqual(model.board_size, 5)

    def test_model_has_all_components(self):
        """モデルが全ての必要なコンポーネントを持っているか確認"""
        model = ContrastDualPolicyNet()

        # Initial Block
        self.assertIsInstance(model.conv_input, nn.Conv2d)
        self.assertIsInstance(model.bn_input, nn.BatchNorm2d)

        # Residual Blocks
        self.assertIsInstance(model.res_blocks, nn.ModuleList)
        self.assertGreater(len(model.res_blocks), 0)

        # Move Head
        self.assertIsInstance(model.move_conv, nn.Conv2d)
        self.assertIsInstance(model.move_bn, nn.BatchNorm2d)
        self.assertIsInstance(model.move_fc, nn.Linear)

        # Tile Head
        self.assertIsInstance(model.tile_conv, nn.Conv2d)
        self.assertIsInstance(model.tile_bn, nn.BatchNorm2d)
        self.assertIsInstance(model.tile_fc, nn.Linear)

        # Value Head
        self.assertIsInstance(model.value_conv, nn.Conv2d)
        self.assertIsInstance(model.value_bn, nn.BatchNorm2d)
        self.assertIsInstance(model.value_fc1, nn.Linear)
        self.assertIsInstance(model.value_fc2, nn.Linear)


class TestContrastDualPolicyNetForward(unittest.TestCase):
    """ContrastDualPolicyNetのforward()テスト"""

    def test_forward_output_shapes(self):
        """forward()が正しい形状を返すか確認"""
        model = ContrastDualPolicyNet()
        model.eval()

        batch_size = 4
        x = torch.randn(batch_size, 90, 5, 5)

        move_logits, tile_logits, value = model(x)

        # 形状の確認
        self.assertEqual(move_logits.shape, (batch_size, 625))
        self.assertEqual(tile_logits.shape, (batch_size, 51))
        self.assertEqual(value.shape, (batch_size, 1))

    def test_forward_value_range(self):
        """forward()のvalue出力が[-1, 1]の範囲にあるか確認"""
        model = ContrastDualPolicyNet()
        model.eval()

        x = torch.randn(2, 90, 5, 5)
        _, _, value = model(x)

        # tanhを使っているので[-1, 1]の範囲
        self.assertTrue(torch.all(value >= -1.0))
        self.assertTrue(torch.all(value <= 1.0))

    def test_forward_single_batch(self):
        """バッチサイズ1でforward()が動作するか確認"""
        model = ContrastDualPolicyNet()
        model.eval()

        x = torch.randn(1, 90, 5, 5)
        move_logits, tile_logits, value = model(x)

        self.assertEqual(move_logits.shape, (1, 625))
        self.assertEqual(tile_logits.shape, (1, 51))
        self.assertEqual(value.shape, (1, 1))

    def test_forward_large_batch(self):
        """大きいバッチサイズでforward()が動作するか確認"""
        model = ContrastDualPolicyNet()
        model.eval()

        batch_size = 128
        x = torch.randn(batch_size, 90, 5, 5)

        move_logits, tile_logits, value = model(x)

        self.assertEqual(move_logits.shape, (batch_size, 625))
        self.assertEqual(tile_logits.shape, (batch_size, 51))
        self.assertEqual(value.shape, (batch_size, 1))

    def test_forward_deterministic_in_eval_mode(self):
        """eval()モードで同じ入力に対して同じ出力が返されるか確認"""
        model = ContrastDualPolicyNet()
        model.eval()

        x = torch.randn(2, 90, 5, 5)

        with torch.no_grad():
            m1, t1, v1 = model(x)
            m2, t2, v2 = model(x)

        # 同じ出力が返される
        self.assertTrue(torch.allclose(m1, m2))
        self.assertTrue(torch.allclose(t1, t2))
        self.assertTrue(torch.allclose(v1, v2))


class TestLossFunction(unittest.TestCase):
    """loss_function()のテスト"""

    def test_loss_function_basic(self):
        """loss_function()が正しく動作するか確認"""
        batch_size = 4

        move_logits = torch.randn(batch_size, 625)
        tile_logits = torch.randn(batch_size, 51)
        value_pred = torch.randn(batch_size, 1)

        # ターゲット (クラスインデックス)
        move_targets = torch.randint(0, 625, (batch_size,))
        tile_targets = torch.randint(0, 51, (batch_size,))
        value_targets = torch.randn(batch_size, 1)

        total_loss, (value_loss, move_loss, tile_loss) = loss_function(
            move_logits,
            tile_logits,
            value_pred,
            move_targets,
            tile_targets,
            value_targets,
        )

        # 損失が計算される
        self.assertIsInstance(total_loss, torch.Tensor)
        self.assertIsInstance(value_loss, float)
        self.assertIsInstance(move_loss, float)
        self.assertIsInstance(tile_loss, float)

        # 総損失が正の値
        self.assertGreater(total_loss.item(), 0)

    def test_loss_function_components_sum(self):
        """loss_function()の各損失が正しく合算されるか確認"""
        batch_size = 2

        move_logits = torch.randn(batch_size, 625)
        tile_logits = torch.randn(batch_size, 51)
        value_pred = torch.randn(batch_size, 1)

        move_targets = torch.randint(0, 625, (batch_size,))
        tile_targets = torch.randint(0, 51, (batch_size,))
        value_targets = torch.randn(batch_size, 1)

        total_loss, (value_loss, move_loss, tile_loss) = loss_function(
            move_logits,
            tile_logits,
            value_pred,
            move_targets,
            tile_targets,
            value_targets,
        )

        # 総損失が各損失の合計に近い (誤差を考慮)
        expected_total = value_loss + move_loss + tile_loss
        self.assertAlmostEqual(total_loss.item(), expected_total, places=5)

    def test_loss_function_gradient_flow(self):
        """loss_function()の勾配が正しく流れるか確認"""
        batch_size = 2

        move_logits = torch.randn(batch_size, 625, requires_grad=True)
        tile_logits = torch.randn(batch_size, 51, requires_grad=True)
        value_pred = torch.randn(batch_size, 1, requires_grad=True)

        move_targets = torch.randint(0, 625, (batch_size,))
        tile_targets = torch.randint(0, 51, (batch_size,))
        value_targets = torch.randn(batch_size, 1)

        total_loss, _ = loss_function(
            move_logits,
            tile_logits,
            value_pred,
            move_targets,
            tile_targets,
            value_targets,
        )

        # 勾配計算
        total_loss.backward()

        # 勾配が計算される
        self.assertIsNotNone(move_logits.grad)
        self.assertIsNotNone(tile_logits.grad)
        self.assertIsNotNone(value_pred.grad)

    def test_loss_function_with_perfect_prediction(self):
        """完璧な予測でのloss_function()の動作確認"""
        batch_size = 2

        # 正解クラスに高いlogitを設定
        move_targets = torch.tensor([0, 1])
        tile_targets = torch.tensor([0, 1])

        move_logits = torch.zeros(batch_size, 625)
        move_logits[0, 0] = 10.0
        move_logits[1, 1] = 10.0

        tile_logits = torch.zeros(batch_size, 51)
        tile_logits[0, 0] = 10.0
        tile_logits[1, 1] = 10.0

        # 完璧な価値予測
        value_targets = torch.tensor([[1.0], [-1.0]])
        value_pred = value_targets.clone()

        total_loss, (value_loss, move_loss, tile_loss) = loss_function(
            move_logits,
            tile_logits,
            value_pred,
            move_targets,
            tile_targets,
            value_targets,
        )

        # 完璧な予測なので損失は小さい
        self.assertLess(total_loss.item(), 1.0)
        self.assertLess(value_loss, 0.01)


class TestModelSaveLoad(unittest.TestCase):
    """モデルの保存/読み込みテスト"""

    def test_save_and_load_model(self):
        """モデルの保存と読み込みが正しく動作するか確認"""
        model1 = ContrastDualPolicyNet()
        model1.eval()

        # ダミー入力で出力を取得
        x = torch.randn(2, 90, 5, 5)
        with torch.no_grad():
            m1, t1, v1 = model1(x)

        # 一時ファイルに保存
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pth") as tmp:
            tmp_path = tmp.name
            torch.save(model1.state_dict(), tmp_path)

        try:
            # 新しいモデルをロード
            model2 = ContrastDualPolicyNet()
            model2.load_state_dict(torch.load(tmp_path, map_location="cpu"))
            model2.eval()

            # 同じ入力で出力を確認
            with torch.no_grad():
                m2, t2, v2 = model2(x)

            # 出力が一致する
            self.assertTrue(torch.allclose(m1, m2))
            self.assertTrue(torch.allclose(t1, t2))
            self.assertTrue(torch.allclose(v1, v2))

        finally:
            # 一時ファイルを削除
            Path(tmp_path).unlink(missing_ok=True)

    def test_model_state_dict_keys(self):
        """state_dict()が全ての必要なキーを含むか確認"""
        model = ContrastDualPolicyNet()
        state_dict = model.state_dict()

        # 重要なキーが含まれているか確認
        required_keys = [
            "conv_input.weight",
            "bn_input.weight",
            "move_fc.weight",
            "tile_fc.weight",
            "value_fc2.weight",
        ]

        for key in required_keys:
            self.assertIn(key, state_dict.keys())


class TestModelDeviceCompatibility(unittest.TestCase):
    """GPU/CPU互換性テスト"""

    def test_model_on_cpu(self):
        """モデルがCPUで動作するか確認"""
        device = torch.device("cpu")
        model = ContrastDualPolicyNet().to(device)
        model.eval()

        x = torch.randn(2, 90, 5, 5).to(device)
        m, t, v = model(x)

        self.assertEqual(m.device.type, "cpu")
        self.assertEqual(t.device.type, "cpu")
        self.assertEqual(v.device.type, "cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_model_on_gpu(self):
        """モデルがGPUで動作するか確認（CUDA利用可能時のみ）"""
        device = torch.device("cuda")
        model = ContrastDualPolicyNet().to(device)
        model.eval()

        x = torch.randn(2, 90, 5, 5).to(device)
        m, t, v = model(x)

        self.assertEqual(m.device.type, "cuda")
        self.assertEqual(t.device.type, "cuda")
        self.assertEqual(v.device.type, "cuda")

    def test_model_device_transfer(self):
        """モデルのデバイス間転送が正しく動作するか確認"""
        model = ContrastDualPolicyNet()

        # CPU -> CPU
        model_cpu = model.to(torch.device("cpu"))
        self.assertIsNotNone(model_cpu)

        # GPU利用可能な場合のみテスト
        if torch.cuda.is_available():
            model_gpu = model.to(torch.device("cuda"))
            model_back_to_cpu = model_gpu.to(torch.device("cpu"))
            self.assertIsNotNone(model_back_to_cpu)


class TestModelEdgeCases(unittest.TestCase):
    """エッジケースのテスト"""

    def test_model_with_zero_input(self):
        """ゼロ入力でモデルが動作するか確認"""
        model = ContrastDualPolicyNet()
        model.eval()

        x = torch.zeros(2, 90, 5, 5)
        m, t, v = model(x)

        # 出力が生成される
        self.assertIsNotNone(m)
        self.assertIsNotNone(t)
        self.assertIsNotNone(v)

    def test_model_with_extreme_input(self):
        """極端な値の入力でモデルが動作するか確認"""
        model = ContrastDualPolicyNet()
        model.eval()

        # 非常に大きい値
        x_large = torch.ones(2, 90, 5, 5) * 100
        m, t, v = model(x_large)

        self.assertFalse(torch.isnan(m).any())
        self.assertFalse(torch.isnan(t).any())
        self.assertFalse(torch.isnan(v).any())

    def test_loss_function_with_batch_size_one(self):
        """バッチサイズ1でloss_function()が動作するか確認"""
        batch_size = 1

        move_logits = torch.randn(batch_size, 625)
        tile_logits = torch.randn(batch_size, 51)
        value_pred = torch.randn(batch_size, 1)

        move_targets = torch.randint(0, 625, (batch_size,))
        tile_targets = torch.randint(0, 51, (batch_size,))
        value_targets = torch.randn(batch_size, 1)

        total_loss, _ = loss_function(
            move_logits,
            tile_logits,
            value_pred,
            move_targets,
            tile_targets,
            value_targets,
        )

        self.assertIsInstance(total_loss, torch.Tensor)
        self.assertGreater(total_loss.item(), 0)


class TestModelTraining(unittest.TestCase):
    """モデルの学習動作テスト"""

    def test_model_gradient_update(self):
        """モデルのパラメータが勾配で更新されるか確認"""
        model = ContrastDualPolicyNet()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # 初期パラメータを保存
        initial_params = [p.clone() for p in model.parameters()]

        # ダミーデータで学習
        x = torch.randn(2, 90, 5, 5)
        move_targets = torch.randint(0, 625, (2,))
        tile_targets = torch.randint(0, 51, (2,))
        value_targets = torch.randn(2, 1)

        optimizer.zero_grad()
        m, t, v = model(x)
        total_loss, _ = loss_function(
            m, t, v, move_targets, tile_targets, value_targets
        )
        total_loss.backward()
        optimizer.step()

        # パラメータが更新されているか確認
        updated = False
        for initial_p, current_p in zip(initial_params, model.parameters()):
            if not torch.allclose(initial_p, current_p):
                updated = True
                break

        self.assertTrue(updated, "パラメータが更新されませんでした")

    def test_model_training_mode(self):
        """train()とeval()モードが正しく切り替わるか確認"""
        model = ContrastDualPolicyNet()

        # train()モード
        model.train()
        self.assertTrue(model.training)

        # eval()モード
        model.eval()
        self.assertFalse(model.training)


if __name__ == "__main__":
    unittest.main(verbosity=2)
