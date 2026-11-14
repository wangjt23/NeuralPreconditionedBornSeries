import unittest

from rte.analyze_collision_transport_replications import (
    bootstrap_mean_interval, hierarchical_interval, paired_summary,
)


class ReplicationAnalysisTests(unittest.TestCase):
    def test_empty_and_constant_intervals(self):
        self.assertEqual(bootstrap_mean_interval([], 1), (None, None))
        self.assertEqual(hierarchical_interval([]), (None, None))
        self.assertEqual(bootstrap_mean_interval([2., 2.], 1), (2., 2.))
        self.assertEqual(hierarchical_interval([{'seed': 1, 'delta_seconds': 2.}]), (2., 2.))

    def test_hierarchical_interval_retains_between_seed_variation(self):
        rows = [{'seed': seed, 'delta_seconds': value}
                for seed, value in [(1, -10.), (2, 0.), (3, 10.)] for _ in range(32)]
        low, high = hierarchical_interval(rows)
        self.assertLess(low, -6.)
        self.assertGreater(high, 6.)
        self.assertEqual((low, high), hierarchical_interval(rows))

    def test_failures_remain_visible_without_speedup_credit(self):
        rows = [dict(seed=1, baseline='HSS', delta_seconds=-.1, delta_iterations=-1,
                     speedup_baseline_over_ct=2., ct_failed=False, baseline_failed=False),
                dict(seed=1, baseline='HSS', delta_seconds=None, delta_iterations=None,
                     speedup_baseline_over_ct=None, ct_failed=True, baseline_failed=False)]
        result = paired_summary(rows, 1)
        self.assertEqual(result['media'], 2)
        self.assertEqual(result['paired_successes'], 1)
        self.assertEqual(result['ct_faster_media'], 1)
        self.assertEqual(result['ct_failed_media'], 1)
        self.assertEqual(result['mean_delta_seconds_ct_minus_baseline'], -.1)
        rows[0].update(delta_seconds=None, ct_failed=True)
        result = paired_summary(rows, 'pooled')
        self.assertEqual(result['paired_successes'], 0)
        self.assertIsNone(result['bootstrap95_mean_delta_low'])
        self.assertIsNone(result['median_speedup_baseline_over_ct'])


if __name__ == '__main__':
    unittest.main()
