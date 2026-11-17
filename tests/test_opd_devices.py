"""Device-budget tests use mocked CUDA and empty models, never GPU results."""

import unittest
from unittest.mock import patch

from transformers import Qwen3Config

from src.opd import plan_teacher_device_map, teacher_memory_limits

GIB = 2**30


class TeacherDeviceTests(unittest.TestCase):
    def memory_limits(self, memory, device="cuda:0"):
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.device_count", return_value=len(memory)),
            patch("torch.cuda.mem_get_info", side_effect=lambda index: memory[index]),
        ):
            return teacher_memory_limits(device)

    def test_single_80gb_preserves_training_headroom(self):
        self.assertEqual(self.memory_limits([(78 * GIB, 79 * GIB)]), {0: 66 * GIB})

    def test_two_40gb_reserve_more_room_for_student_on_zero(self):
        memory = [(int(39 * GIB), int(39.5 * GIB))] * 2
        self.assertEqual(self.memory_limits(memory), {0: 27 * GIB, 1: 36 * GIB})

    def test_single_40gb_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "two almost-empty"):
            self.memory_limits([(39 * GIB, 40 * GIB)])

    def test_occupied_second_gpu_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "Restart kernels"):
            self.memory_limits([(39 * GIB, 40 * GIB), (20 * GIB, 40 * GIB)])

    def test_cpu_is_rejected_before_querying_cuda(self):
        with self.assertRaisesRegex(RuntimeError, "allocated CUDA"):
            teacher_memory_limits("cpu")

    def test_plan_rejects_cpu_and_disk_fallback(self):
        config = Qwen3Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
        )
        for destination in ("cpu", "disk"):
            with (
                self.subTest(destination=destination),
                patch("accelerate.infer_auto_device_map", return_value={"": destination}),
                self.assertRaisesRegex(RuntimeError, "offload was rejected"),
            ):
                plan_teacher_device_map(config, {0: GIB, 1: GIB})

    def test_full_qwen32b_shape_plan_fits_two_weight_budgets(self):
        # Public pinned configuration dimensions, without allocating real weights.
        config = Qwen3Config(
            vocab_size=151936,
            hidden_size=5120,
            intermediate_size=25600,
            num_hidden_layers=64,
            num_attention_heads=64,
            num_key_value_heads=8,
            head_dim=128,
            tie_word_embeddings=False,
        )
        with patch("accelerate.utils.modeling.get_max_memory", side_effect=lambda limits: limits):
            device_map = plan_teacher_device_map(config, {0: 27 * GIB, 1: 36 * GIB})
        self.assertEqual(set(device_map.values()), {0, 1})
        self.assertEqual(device_map["model.embed_tokens"], 0)
        self.assertEqual(device_map["lm_head"], 1)
        # Each complete decoder block stays together, preserving residual paths.
        self.assertEqual(sum(key.startswith("model.layers.") for key in device_map), 64)


if __name__ == "__main__":
    unittest.main()
