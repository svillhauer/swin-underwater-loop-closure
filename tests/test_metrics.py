import math

import numpy as np
import pytest

from loopclosure import metrics


def test_confusion_and_threshold_metrics():
    y = np.array([1, 1, 0, 0, 0])
    s = np.array([0.9, 0.2, 0.8, 0.1, 0.4])
    m = metrics.at_threshold(y, s, 0.5)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 2)
    assert m["precision"] == 0.5 and m["recall"] == 0.5 and m["f1"] == 0.5
    assert math.isclose(m["balanced_accuracy"], (0.5 + 2 / 3) / 2)


@pytest.mark.parametrize("s, expected", [([0.1, 0.2, 0.8, 0.9], 1.0), ([0.9, 0.8, 0.2, 0.1], 0.0),
                                         ([0.5, 0.5, 0.5, 0.5], 0.5)])
def test_roc_auc_extremes_and_ties(s, expected):
    assert metrics.roc_auc([0, 0, 1, 1], s) == expected


def test_roc_auc_matches_pairwise_definition():
    rng = np.random.default_rng(0)
    y = rng.random(300) < 0.2
    s = rng.random(300) + y * 0.3
    pos, neg = s[y], s[~y]
    brute = np.mean((pos[:, None] > neg[None]) + 0.5 * (pos[:, None] == neg[None]))
    assert math.isclose(metrics.roc_auc(y, s), brute, rel_tol=1e-12)


def test_average_precision_known_value():
    # ranking: 1, 0, 1  -> precision at each recall step: 1/1, 2/3
    assert math.isclose(metrics.average_precision([1, 0, 1], [0.9, 0.8, 0.7]), 0.5 * 1 + 0.5 * 2 / 3)


def test_average_precision_of_random_scores_is_the_base_rate():
    rng = np.random.default_rng(1)
    y = rng.random(200_000) < 0.05
    assert abs(metrics.average_precision(y, rng.random(len(y))) - 0.05) < 0.005


def test_best_f1_picks_the_optimal_threshold():
    b = metrics.best_f1([0, 1, 1, 0], [0.1, 0.7, 0.6, 0.65])
    assert b["f1"] == 1.0 or math.isclose(b["f1"], 0.8)
    assert b["threshold"] in (0.6, 0.65, 0.7)
