"""Tests for metric helpers."""

import math

import torch

from mech_uspto.training.metrics import MetricsComputer, binary_pr_auc


def test_pr_auc_perfect_ranking_is_one():
    scores = torch.tensor([0.9, 0.8, 0.2, 0.1])
    targets = torch.tensor([1, 1, 0, 0])
    assert math.isclose(binary_pr_auc(scores, targets), 1.0, abs_tol=1e-6)


def test_pr_auc_inverted_ranking_is_low():
    scores = torch.tensor([0.1, 0.2, 0.8, 0.9])
    targets = torch.tensor([1, 1, 0, 0])
    assert binary_pr_auc(scores, targets) < 0.5


def test_pr_auc_no_positives_returns_zero():
    scores = torch.tensor([0.5, 0.5, 0.5])
    targets = torch.tensor([0, 0, 0])
    assert binary_pr_auc(scores, targets) == 0.0


def test_pr_auc_empty_returns_zero():
    assert binary_pr_auc(torch.empty(0), torch.empty(0)) == 0.0


def _all_no_change_batch(num_classes: int, B: int = 2, N: int = 4):
    """Build a batch where every target pair is 'no change' (Δ=0).

    For ``num_classes=k``, the no-change index is ``k // 2``.
    The model 'perfectly' predicts no-change everywhere.
    """
    no_change_idx = num_classes // 2
    targets = torch.full((B, N, N), no_change_idx, dtype=torch.long)
    # Logits with a huge spike on the no-change index → argmax = no_change_idx everywhere.
    logits = torch.zeros(B, N, N, num_classes)
    logits[..., no_change_idx] = 10.0
    mask = torch.ones(B, N, dtype=torch.bool)
    return logits, targets, mask


def test_mechanism_metrics_no_reactions_present_3class():
    """If all targets are no-change and model predicts no-change, there are no positives."""
    logits, targets, mask = _all_no_change_batch(num_classes=3)
    m = MetricsComputer.get_mechanism_metrics(logits, targets, mask)
    assert m["tp"] == 0 and m["fp"] == 0 and m["fn"] == 0


def test_mechanism_metrics_no_reactions_present_7class():
    """Regression: with 7 classes the no-change index is 3, NOT 1.

    Before the fix this test failed because the metric hardcoded ``!= 1`` and
    treated 99.97% of pairs as 'reaction centers', producing F1 ≈ majority-class
    frequency on the first epoch of every end-to-end training run.
    """
    logits, targets, mask = _all_no_change_batch(num_classes=7)
    m = MetricsComputer.get_mechanism_metrics(logits, targets, mask)
    assert m["tp"] == 0, "no positives expected when all targets are no-change"
    assert m["fp"] == 0, (
        "no false positives expected when all preds are no-change — if non-zero, "
        "metric is using the wrong no_change index"
    )
    assert m["fn"] == 0


def test_mechanism_metrics_detects_true_reaction_7class():
    """7-class: one off-center target, model predicts it correctly → exactly 1 tp."""
    B, N, C = 1, 3, 7
    no_change_idx = C // 2  # 3
    targets = torch.full((B, N, N), no_change_idx, dtype=torch.long)
    # Off-diagonal upper-triangle pair (0, 1) is a real reaction with class 5 (Δ=+2).
    targets[0, 0, 1] = 5
    # Perfect model.
    logits = torch.zeros(B, N, N, C)
    logits[..., no_change_idx] = 10.0
    logits[0, 0, 1, no_change_idx] = 0.0
    logits[0, 0, 1, 5] = 10.0
    mask = torch.ones(B, N, dtype=torch.bool)
    m = MetricsComputer.get_mechanism_metrics(logits, targets, mask)
    assert m["tp"] == 1
    assert m["fp"] == 0
    assert m["fn"] == 0
    assert math.isclose(m["f1"], 1.0, abs_tol=1e-4)


def test_per_class_counters_and_pr_auc_7class():
    """Per-class counters distinguish 'never predicted class c' from 'tried but wrong'.

    Setup: 2x4 batch. Pair (0,1) has true class 5 (Δ=+2). Model predicts class 5
    on that pair AND on one other (FP), and never predicts class 4. This is the
    canonical 'distinguish P=0' regression: class 4 should have n_preds=0
    (signalling 'no signal') while class 5 has n_preds>0.
    """
    B, N, C = 1, 4, 7
    no_change_idx = C // 2  # 3
    targets = torch.full((B, N, N), no_change_idx, dtype=torch.long)
    targets[0, 0, 1] = 5  # one true reaction at (0,1) with Δ=+2

    logits = torch.zeros(B, N, N, C)
    logits[..., no_change_idx] = 10.0
    # Correct prediction at (0, 1):
    logits[0, 0, 1, no_change_idx] = 0.0
    logits[0, 0, 1, 5] = 10.0
    # FP at (2, 3): also predicted class 5 but target is no-change.
    logits[0, 2, 3, no_change_idx] = 0.0
    logits[0, 2, 3, 5] = 10.0
    mask = torch.ones(B, N, dtype=torch.bool)

    m = MetricsComputer.get_mechanism_metrics(logits, targets, mask)
    assert m["n_preds_per_class"][5] == 2, "class 5 predicted twice"
    assert m["n_preds_per_class"][4] == 0, "class 4 never predicted (PR-AUC tells ranking)"
    assert m["n_targets_per_class"][5] == 1, "one true class-5 target"
    assert m["n_targets_per_class"][no_change_idx] == 5, "remaining upper-tri pairs are no-change"
    # Class 5: perfect ranking (highest P(5) is on the true positive).
    assert math.isclose(m["pr_auc_per_class"][5], 1.0, abs_tol=1e-4)
    # Class 4: no positives → AP undefined → 0.0 by convention.
    assert m["pr_auc_per_class"][4] == 0.0


def test_pr_auc_uses_full_non_no_change_mass_7class():
    """Regression: PR-AUC scoring must use P(any non-no-change), not P(first)+P(last).

    Build a batch where:
      - target reaction is class 4 (Δ=+1) on pair (0,1)
      - model puts ALL its non-no-change mass on class 4 for that pair
        and almost ZERO mass on classes 0 and 6
    The old buggy scorer (probs[:, 0] + probs[:, -1]) would give the positive
    pair a tiny score and rank it BELOW negative pairs whose noise happened to
    favor class 0 or 6. The fixed scorer (1 - probs[:, 3]) correctly ranks it
    near the top, yielding PR-AUC ≈ 1.
    """
    B, N, C = 1, 3, 7
    no_change_idx = C // 2  # 3
    targets = torch.full((B, N, N), no_change_idx, dtype=torch.long)
    targets[0, 0, 1] = 4  # Δ=+1 (a "middle" rare class, not first or last)

    logits = torch.zeros(B, N, N, C)
    # Negative pairs: nearly all mass on no-change, with a tiny tilt toward classes 0 and 6.
    logits[..., no_change_idx] = 5.0
    logits[..., 0] = 1.0
    logits[..., -1] = 1.0
    # Positive pair (0,1): split between no-change and class 4 ONLY.
    # Almost no probability on classes 0 or 6.
    logits[0, 0, 1, :] = -10.0
    logits[0, 0, 1, no_change_idx] = 1.0
    logits[0, 0, 1, 4] = 2.0  # the true class wins

    mask = torch.ones(B, N, dtype=torch.bool)
    m = MetricsComputer.get_mechanism_metrics(logits, targets, mask)

    # With the fix, positive pair has rxn_scores ≈ 1 - softmax_no_change, large.
    # Negative pairs have rxn_scores ≈ 1 - softmax_no_change (no-change dominates), small.
    # → positive ranked first → PR-AUC = 1.0.
    assert m["pr_auc"] > 0.95, (
        f"PR-AUC should be ~1 when the true rare class is correctly ranked, "
        f"got {m['pr_auc']:.3f}. If close to 0, the scorer is still using only "
        f"probs[:, 0] + probs[:, -1] and missing P(Δ=±1), P(Δ=±2)."
    )


def test_n_rxn_preds_zero_when_model_collapsed():
    """Class-collapsed model (always predicts no-change) reports 0 rare predictions."""
    logits, targets, mask = _all_no_change_batch(num_classes=7)
    # Add some real reactions to targets so n_rxn_targets > 0.
    targets[0, 0, 1] = 5  # Δ=+2
    targets[0, 1, 2] = 4  # Δ=+1
    m = MetricsComputer.get_mechanism_metrics(logits, targets, mask)
    assert m["n_rxn_targets"] >= 2, "sanity: targets contain at least 2 reactions"
    assert m["n_rxn_preds"] == 0, "class-collapsed model should predict NO rare classes"
    assert m["tp"] == 0
    # When n_rxn_preds == 0, precision is undefined-by-convention 0, recall is 0.
    assert m["precision"] == 0.0
    assert m["recall"] == 0.0


def test_precision_recall_improve_with_one_correct_tp_7class():
    """Sanity check: ONE correct rare-class TP → P>0 and R>0.

    Setup: 5 reactions in targets, model predicts no-change everywhere EXCEPT
    one pair where it correctly predicts the rare class. Precision should be
    perfect (1 TP, 0 FP → P=1.0) and recall should be 1/5=0.2.
    """
    B, N, C = 1, 5, 7
    no_change_idx = C // 2  # 3
    # 5 reaction pairs in upper triangle: (0,1), (0,2), (0,3), (1,2), (1,3).
    rxn_pairs = [(0, 1, 5), (0, 2, 4), (0, 3, 2), (1, 2, 5), (1, 3, 4)]
    targets = torch.full((B, N, N), no_change_idx, dtype=torch.long)
    for i, j, c in rxn_pairs:
        targets[0, i, j] = c

    # Model: predict no-change everywhere except (0,1) where it correctly predicts 5.
    logits = torch.zeros(B, N, N, C)
    logits[..., no_change_idx] = 10.0
    logits[0, 0, 1, no_change_idx] = 0.0
    logits[0, 0, 1, 5] = 10.0

    mask = torch.ones(B, N, dtype=torch.bool)
    m = MetricsComputer.get_mechanism_metrics(logits, targets, mask)

    assert m["n_rxn_preds"] == 1, "model should have made exactly 1 rare-class prediction"
    assert m["tp"] == 1, "that 1 rare prediction was correct"
    assert m["fp"] == 0
    assert m["fn"] == 4, "4 reactions were predicted as no-change"
    assert math.isclose(m["precision"], 1.0, abs_tol=1e-6), (
        f"P should be 1.0 (1 TP / (1 TP + 0 FP)), got {m['precision']}"
    )
    assert math.isclose(m["recall"], 0.2, abs_tol=1e-6), (
        f"R should be 0.2 (1 TP / (1 TP + 4 FN)), got {m['recall']}"
    )


def test_precision_zero_when_model_tries_but_misses():
    """Distinguish 'never tried' from 'tried but wrong': the latter has FP>0."""
    B, N, C = 1, 3, 7
    no_change_idx = C // 2  # 3
    targets = torch.full((B, N, N), no_change_idx, dtype=torch.long)
    targets[0, 0, 1] = 5  # Δ=+2

    # Model predicts Δ=+3 (class 6) everywhere — including (0,1) which should be class 5.
    logits = torch.zeros(B, N, N, C)
    logits[..., 6] = 10.0

    mask = torch.ones(B, N, dtype=torch.bool)
    m = MetricsComputer.get_mechanism_metrics(logits, targets, mask)

    assert m["n_rxn_preds"] > 0, "model predicted rare classes (it's just wrong)"
    assert m["tp"] == 0
    assert m["fp"] > 0
    assert m["precision"] == 0.0, "P=0 with FP>0 means 'tried but missed', different from collapse"


def test_pooled_pr_auc_differs_from_averaged_per_batch():
    """Regression: epoch-level PR-AUC must pool scores across all batches, not average per-batch APs.

    Two batches:
      - Batch A: positives ranked perfectly (per-batch AP = 1.0)
      - Batch B: positives ranked at the bottom (per-batch AP ≈ 0.4)
    The MEAN of per-batch APs (~0.7) is mathematically meaningless; the only
    correct epoch summary is to concatenate (scores, targets) across batches
    and compute AP once. With the score distributions chosen below, the pooled
    AP is ~0.43 — well below the (wrong) average of 0.7.
    """
    from mech_uspto.training.metrics import pooled_pr_auc

    # Batch A: positives at top.
    scores_a = torch.tensor([0.9, 0.85, 0.1, 0.05])
    targets_a = torch.tensor([1.0, 1.0, 0.0, 0.0])
    # Batch B: positives at bottom (low scores) — per-batch AP is poor.
    scores_b = torch.tensor([0.95, 0.92, 0.2, 0.15])
    targets_b = torch.tensor([0.0, 0.0, 1.0, 1.0])

    ap_a = binary_pr_auc(scores_a, targets_a)
    ap_b = binary_pr_auc(scores_b, targets_b)
    avg_of_per_batch = (ap_a + ap_b) / 2

    # Pooled: when concatenated, the 4 positives are interleaved with negatives
    # that outrank them → pooled AP is much lower than the average of per-batch APs.
    pooled, _ = pooled_pr_auc(
        scores_list=[scores_a, scores_b],
        targets_list=[targets_a, targets_b],
        probs_list=[],
        class_targets_list=[],
        num_classes=0,
    )
    # Sanity: per-batch APs are very different.
    assert ap_a > 0.99
    assert ap_b < 0.6
    assert avg_of_per_batch > 0.7
    # The bug: averaged per-batch APs ≈ 0.7, but pooled AP must be < 0.6.
    assert pooled < avg_of_per_batch - 0.1, (
        f"pooled AP ({pooled:.3f}) should be substantially lower than "
        f"average-of-per-batch APs ({avg_of_per_batch:.3f}); if they match, "
        f"the engine is still averaging APs instead of pooling scores"
    )
    # And it should match sklearn's behavior of computing AP on the concatenated arrays.
    expected = binary_pr_auc(torch.cat([scores_a, scores_b]), torch.cat([targets_a, targets_b]))
    assert math.isclose(pooled, expected, abs_tol=1e-6)


def test_pooled_pr_auc_per_class_matches_concatenated():
    """Per-class pooled PR-AUC must equal AP on concatenated (probs[:,c], target==c)."""
    from mech_uspto.training.metrics import pooled_pr_auc

    C = 3
    # Two batches, 4 samples each, 3 classes.
    probs_a = torch.tensor([[0.1, 0.8, 0.1], [0.2, 0.7, 0.1], [0.6, 0.3, 0.1], [0.5, 0.4, 0.1]])
    class_targets_a = torch.tensor([1, 1, 0, 0])
    probs_b = torch.tensor([[0.1, 0.1, 0.8], [0.2, 0.1, 0.7], [0.1, 0.6, 0.3], [0.1, 0.5, 0.4]])
    class_targets_b = torch.tensor([2, 2, 1, 1])

    _, per_class = pooled_pr_auc(
        scores_list=[],
        targets_list=[],
        probs_list=[probs_a, probs_b],
        class_targets_list=[class_targets_a, class_targets_b],
        num_classes=C,
    )

    pooled_probs = torch.cat([probs_a, probs_b])
    pooled_ct = torch.cat([class_targets_a, class_targets_b])
    for c in range(C):
        expected = binary_pr_auc(pooled_probs[:, c], (pooled_ct == c).to(torch.float32))
        assert math.isclose(per_class[c], expected, abs_tol=1e-6), (
            f"class {c}: per-class pooled PR-AUC {per_class[c]:.4f} != expected {expected:.4f}"
        )


def test_get_mechanism_metrics_exposes_pooled_buffers_for_epoch_aggregation():
    """Engine relies on metrics returning detached CPU buffers under reserved keys."""
    B, N, C = 1, 3, 7
    no_change_idx = C // 2
    targets = torch.full((B, N, N), no_change_idx, dtype=torch.long)
    targets[0, 0, 1] = 5
    logits = torch.zeros(B, N, N, C)
    logits[..., no_change_idx] = 10.0
    logits[0, 0, 1, no_change_idx] = 0.0
    logits[0, 0, 1, 5] = 10.0
    mask = torch.ones(B, N, dtype=torch.bool)
    m = MetricsComputer.get_mechanism_metrics(logits, targets, mask)

    for k in ("_pooled_scores", "_pooled_targets", "_pooled_probs", "_pooled_class_targets"):
        assert k in m, f"missing {k} (needed for epoch-level PR-AUC pooling)"
        assert isinstance(m[k], torch.Tensor)
        assert m[k].device.type == "cpu"
        assert not m[k].requires_grad

    # Sizes consistent with the upper-triangle count (3 atoms → 3 pairs).
    assert m["_pooled_scores"].numel() == 3
    assert m["_pooled_targets"].numel() == 3
    assert m["_pooled_probs"].shape == (3, C)
    assert m["_pooled_class_targets"].numel() == 3
