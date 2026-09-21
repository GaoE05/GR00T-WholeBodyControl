"""N4 follow-up: singleton safety without changing released batch updates."""
import copy
import importlib.util
from pathlib import Path
import unittest
import torch

FILE = Path(__file__).resolve().parents[1] / "gear_sonic/utils/running_mean_std.py"
spec = importlib.util.spec_from_file_location("rms_test_module", FILE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
RunningMeanStd = module.RunningMeanStd


class SingletonGuardTests(unittest.TestCase):
    def test_singleton_train_calls_stay_finite_and_do_not_update(self):
        rms = RunningMeanStd((3,))
        before = copy.deepcopy(rms.state_dict())
        x = torch.tensor([[1.0, 2.0, -3.0]])
        for _ in range(3):
            self.assertTrue(torch.isfinite(rms(x)).all())
        for key, value in rms.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]), key)

    def test_regular_batch_is_bitwise_legacy_update(self):
        torch.manual_seed(19)
        x = torch.randn(2048, 5) * 3 + 4
        rms = RunningMeanStd((5,))
        expected_output = ((x - rms.running_mean) / (rms.running_var + rms.epsilon).sqrt()).clamp(-5, 5)
        expected = rms._update_mean_var_count_from_moments(
            rms.running_mean,
            rms.running_var,
            rms.count,
            x.mean(rms.axis),
            x.var(rms.axis),
            x.size(0),
        )
        output = rms(x)
        self.assertTrue(torch.equal(output, expected_output))
        self.assertTrue(torch.equal(rms.running_mean, expected[0]))
        self.assertTrue(torch.equal(rms.running_var, expected[1]))
        self.assertTrue(torch.equal(rms.count, expected[2]))

    def test_reshaped_time_batch_uses_actual_flattened_count(self):
        rms = RunningMeanStd((3,))
        rms(torch.tensor([[[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]]]))
        self.assertEqual(rms.count.item(), 3)
        self.assertTrue(torch.isfinite(rms.running_var).all())

    def test_eval_and_frozen_legacy_state_remain_bitwise_unchanged(self):
        rms = RunningMeanStd((3,))
        state = copy.deepcopy(rms.state_dict())
        state["running_mean"] = torch.tensor([1.0, -2.0, 0.5])
        state["running_var"] = torch.tensor([0.2, 3.0, 2.0])
        state["count"] = torch.tensor(4097.0)
        state._metadata[""]["version"] = 1
        rms.load_state_dict(state, strict=True)
        x = torch.tensor([[0.4, -1.0, 0.1]])
        expected = ((x - state["running_mean"]) / (state["running_var"] + rms.epsilon).sqrt()).clamp(-5, 5)
        for use_eval in (True, False):
            rms.train(not use_eval)
            rms.frozen = not use_eval
            for _ in range(3):
                self.assertTrue(torch.equal(rms(x), expected))
            for key, value in rms.state_dict().items():
                self.assertTrue(torch.equal(value, state[key]), key)
        self.assertEqual(rms.state_dict()._metadata[""]["version"], 3)


if __name__ == "__main__":
    unittest.main()
