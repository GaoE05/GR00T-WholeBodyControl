"""N4: population-moment oracle and exact frozen legacy compatibility."""
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


def expected_with_prior(x):
    x = x.double().reshape(-1, x.shape[-1])
    count = len(x) + 1
    mean = x.sum(0) / count
    # Existing prior: count=1, mean=0, variance=1, unchanged by this fix.
    var = (x.square().sum(0) + 1) / count - mean.square()
    return mean, var


class PopulationStatsTests(unittest.TestCase):
    def test_singleton_remains_finite_on_following_call(self):
        r = RunningMeanStd((3,))
        x = torch.tensor([[1., 2., -3.]])
        self.assertTrue(torch.isfinite(r(x)).all())
        self.assertTrue(torch.isfinite(r.running_var).all())
        self.assertTrue(torch.isfinite(r(x)).all())
        mean, var = expected_with_prior(x.repeat(2, 1))
        torch.testing.assert_close(r.running_mean.double(), mean)
        torch.testing.assert_close(r.running_var.double(), var)

    def test_batch_partition_invariance_matches_population_oracle(self):
        torch.manual_seed(19)
        x = torch.randn(17, 5) * 3 + 4
        whole = RunningMeanStd((5,)); whole(x)
        chunks = RunningMeanStd((5,))
        for part in (x[:1], x[1:7], x[7:]): chunks(part)
        mean, var = expected_with_prior(x)
        for r in (whole, chunks):
            torch.testing.assert_close(r.running_mean.double(), mean, atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(r.running_var.double(), var, atol=2e-6, rtol=1e-6)
            self.assertEqual(r.count.item(), 18)

    def test_constant_and_sequence_batches(self):
        for x in (torch.full((8, 3), 2.), torch.arange(24.).reshape(2, 4, 3)):
            r = RunningMeanStd((3,)); output = r(x)
            self.assertEqual(output.shape, x.shape)
            mean, var = expected_with_prior(x)
            torch.testing.assert_close(r.running_mean.double(), mean)
            torch.testing.assert_close(r.running_var.double(), var)
            self.assertTrue(torch.isfinite(r(x)).all())

    def test_legacy_state_load_frozen_output_and_buffers_bitwise_unchanged(self):
        r = RunningMeanStd((3,))
        state = copy.deepcopy(r.state_dict())
        state['running_mean'] = torch.tensor([1., -2., .5])
        state['running_var'] = torch.tensor([.2, 3., 2.])
        state['count'] = torch.tensor(4097.)
        state._metadata['']['version'] = 1
        r.load_state_dict(state, strict=True)
        x = torch.tensor([[.4, -1., .1]])
        expected = ((x-state['running_mean']) / (state['running_var']+r.epsilon).sqrt()).clamp(-5, 5)
        for use_eval in (True, False):
            r.train(not use_eval); r.frozen = not use_eval
            for _ in range(3):
                self.assertTrue(torch.equal(r(x), expected))
            for key, value in r.state_dict().items():
                self.assertTrue(torch.equal(value, state[key]))
        self.assertEqual(r.state_dict()._metadata['']['version'], 2)

if __name__ == '__main__': unittest.main()
