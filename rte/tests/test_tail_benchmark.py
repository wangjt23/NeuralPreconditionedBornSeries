import pytest

from rte.benchmark_tail_solver import paired_comparison, select, summarize


def test_case_weighted_summary_selection_and_pairing():
    rows = []
    for index in range(3):
        for method, seconds in [('c', 2.+index), ('n', 1.+index/2)]:
            for repeat in range(3):
                rows.append(dict(method=method, index=index, seconds=seconds+(100 if repeat == 2 else 0),
                    failed=0, true_residual=1e-6, setup_seconds=.2, setup_plus_seconds=seconds+.2,
                    iterations=4, classical_calls=4, injection_calls=0))
    cases, summary = summarize(rows)
    assert len(cases) == 6
    assert select(summary, ['c', 'n']) == 'n'
    assert select(summary, ['c']) == 'c'
    result = paired_comparison(cases, 'c', 'n')
    assert result['independent_media'] == 3
    assert result['saving_percent'] == pytest.approx(50)
    assert result['saving_percent_ci95'] == pytest.approx([50, 50])
    summary[1]['failed'] = 1
    assert select(summary, ['c', 'n']) == 'c'
    with pytest.raises(RuntimeError):
        select(summary, ['n'])
