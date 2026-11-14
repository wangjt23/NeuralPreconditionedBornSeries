import unittest

import pandas as pd

from rte.report_source_initialization import paired_comparison, pool_cases, summary


class SourceReportTests(unittest.TestCase):
    def test_repetitions_are_not_independent_cases(self):
        rows = [dict(size=64, index=i, method=m, repeat=r, seconds=t,
            iterations=5, failed=0, true_residual=1e-6)
            for i in range(3) for m, value in [('base', 2.), ('neural', 1.)]
            for r, t in enumerate([value, value, value*5])]
        pooled = pool_cases(pd.DataFrame(rows))
        self.assertEqual(len(pooled), 6)
        self.assertTrue((pooled.repeats == 3).all())
        table = summary(pooled).set_index('method')
        self.assertEqual(table.loc['base', 'media'], 3)
        self.assertEqual(table.loc['base', 'mean_seconds'], 2.)
        pair = paired_comparison(pooled, 'base', 'neural', 'unit').iloc[0]
        self.assertEqual(pair.aggregate_savings, .5)
        self.assertEqual(pair.savings_ci_low, .5)
        self.assertEqual(pair.savings_ci_high, .5)
        self.assertEqual(pair.paired_wins, 3)

    def test_failed_method_is_retained_and_missing_pair_rejected(self):
        rows = [dict(size=64, index=i, method=m, repeat=0, seconds=1.,
            iterations=100 if m == 'neural' else 5, failed=int(m == 'neural'),
            true_residual=.1 if m == 'neural' else 1e-6)
            for i in range(2) for m in ('base', 'neural')]
        pooled = pool_cases(pd.DataFrame(rows))
        pair = paired_comparison(pooled, 'base', 'neural', 'unit').iloc[0]
        self.assertEqual(pair.candidate_failed, 2)
        with self.assertRaises(ValueError):
            paired_comparison(pooled.iloc[:-1], 'base', 'neural', 'unit')


if __name__ == '__main__':
    unittest.main()
