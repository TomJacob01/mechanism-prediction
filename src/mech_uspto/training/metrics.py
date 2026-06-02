"""Metrics for delta-matrix prediction."""

import torch


def binary_pr_auc(scores: torch.Tensor, targets: torch.Tensor) -> float:
    """Average-precision (area under the precision-recall curve) for a binary task.

    Computed via the standard step-function integration used by
    ``sklearn.metrics.average_precision_score``: ``AP = Σ (R_n − R_{n-1}) · P_n``.

    Args:
        scores: 1-D tensor of per-sample positive-class scores.
        targets: 1-D bool/0-1 tensor of the same length.

    Returns:
        Average precision in ``[0, 1]``. Returns ``0.0`` when there are no
        positives (undefined PR curve).
    """
    if scores.numel() == 0:
        return 0.0

    targets = targets.to(torch.float32).flatten()
    scores = scores.flatten().to(torch.float32)

    n_pos = int(targets.sum().item())
    if n_pos == 0:
        return 0.0

    # Sort descending by score; ties broken arbitrarily (stable enough for AP).
    order = torch.argsort(scores, descending=True)
    sorted_targets = targets[order]

    tp_cum = torch.cumsum(sorted_targets, dim=0)
    ranks = torch.arange(1, sorted_targets.numel() + 1, device=scores.device, dtype=torch.float32)
    precision = tp_cum / ranks
    recall = tp_cum / n_pos

    # AP = Σ (R_n − R_{n-1}) · P_n with R_0 = 0.
    recall_prev = torch.cat([torch.zeros(1, device=scores.device), recall[:-1]])
    ap = ((recall - recall_prev) * precision).sum().item()
    return float(ap)


def _metrics_for_sample(
    mol_logits: torch.Tensor,
    mol_targets: torch.Tensor,
) -> tuple[
    int,
    int,
    int,
    int,
    int,
    int,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    list[int],
    list[int],
]:
    """Per-sample reaction-center metrics over upper-triangle pairs.

    Returns ``(tp, fp, fn, n_rxn_preds, exact_match_top1, total_rxns, scores,
    target_bin, probs, targets, n_preds_per_class, n_targets_per_class)``.

    ``exact_match_top1`` is 1 iff ``argmax(logits, dim=-1)`` equals
    ``mol_targets`` at *every* pair in the sample — i.e., the model's top-1
    full Δ-matrix exactly matches the ground truth. This is the
    PMechDB / Bradshaw-2018 mechanism-prediction metric (cf. FUTURE_TASKS #2)
    and replaces the previous pair-level ``topk_acc`` which was mathematically
    capped at ~5% in end_to_end mode (~55 reactions per sample / k=3).

    The extra last four entries support per-class PR-AUC pooling and the
    "did the model predict this class at all?" diagnostic that distinguishes
    ``P=0 because everything was wrong`` from ``P=undefined because nothing
    was predicted``.
    """
    mol_preds = torch.argmax(mol_logits, dim=-1)

    # "No change" (Δ=0) is the middle index after target shifting:
    #   3-class {-1,0,1} → {0,1,2}  → no-change = 1
    #   7-class {-3..3}  → {0..6}   → no-change = 3
    # In general: no_change_idx = num_classes // 2.
    num_classes = mol_logits.shape[-1]
    no_change_idx = num_classes // 2
    is_rxn_target = mol_targets != no_change_idx
    is_rxn_pred = mol_preds != no_change_idx

    tp = (is_rxn_pred & is_rxn_target & (mol_preds == mol_targets)).sum().item()
    fp = (is_rxn_pred & (~is_rxn_target | (mol_preds != mol_targets))).sum().item()
    fn = (is_rxn_target & (~is_rxn_pred | (mol_preds != mol_targets))).sum().item()
    n_rxn_preds = is_rxn_pred.sum().item()

    mol_total_rxns = is_rxn_target.sum().item()

    probs = torch.softmax(mol_logits, dim=-1)
    # "Any reaction" score = total probability mass NOT on the no-change class.
    # Equivalent to summing P(c) over all c != no_change_idx. This works for
    # any num_classes; previously the code used probs[:, 0] + probs[:, -1]
    # which was correct only for 3-class (where first+last == all-rare) and
    # silently broken for 7-class (ignored P(Δ=±1) and P(Δ=±2)).
    rxn_scores = 1.0 - probs[:, no_change_idx]

    # Top-1 exact match: every pair's argmax == target. Whole-sample 0/1.
    # Top-k for k>1 requires a beam search over candidate Δ-matrices
    # (Lawler-style heap enumeration) — deferred follow-up to FUTURE_TASKS #2.
    exact_match_top1 = int((mol_preds == mol_targets).all().item())

    # Per-class counters: how many pairs the model assigned to each class, and
    # how many true pairs of each class exist. Lets callers distinguish
    # "didn't try class c" (n_preds[c] == 0) from "tried but wrong".
    n_preds_per_class = torch.bincount(mol_preds, minlength=num_classes).tolist()
    n_targets_per_class = torch.bincount(mol_targets, minlength=num_classes).tolist()

    return (
        tp,
        fp,
        fn,
        n_rxn_preds,
        exact_match_top1,
        mol_total_rxns,
        rxn_scores.detach(),
        is_rxn_target.detach(),
        probs.detach(),
        mol_targets.detach(),
        n_preds_per_class,
        n_targets_per_class,
    )


class MetricsComputer:
    """Compute precision / recall / F1 / top-K for delta predictions."""

    @staticmethod
    def get_mechanism_metrics(
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, float]:
        """Reaction-center metrics over the upper-triangle of each batch sample."""
        B, N, _, num_classes = logits.shape
        mask_2d = mask.unsqueeze(1) * mask.unsqueeze(2)
        triu_indices = torch.triu_indices(N, N, offset=1, device=logits.device)

        tp, fp, fn, n_rxn_preds = 0, 0, 0, 0
        # Whole-sample top-1 Δ-matrix exact-match (PMechDB-style). Accumulator
        # tracks hits and the number of samples that actually had upper-tri
        # pairs to score (samples with N≤1 contribute nothing).
        exact_match_hits, exact_match_total = 0, 0
        total_rxns = 0
        all_scores: list[torch.Tensor] = []
        all_targets: list[torch.Tensor] = []
        all_probs: list[torch.Tensor] = []
        all_class_targets: list[torch.Tensor] = []
        n_preds_per_class = [0] * num_classes
        n_targets_per_class = [0] * num_classes

        for b in range(B):
            m = mask_2d[b]
            valid = m[triu_indices[0], triu_indices[1]]
            idx0, idx1 = triu_indices[0][valid], triu_indices[1][valid]
            if len(idx0) == 0:
                continue

            mol_logits = logits[b, idx0, idx1]
            mol_targets = targets[b, idx0, idx1]
            (
                s_tp,
                s_fp,
                s_fn,
                s_n_rxn_preds,
                s_exact_match,
                s_total,
                s_scores,
                s_target_bin,
                s_probs,
                s_class_targets,
                s_n_preds,
                s_n_targets,
            ) = _metrics_for_sample(mol_logits, mol_targets)
            tp += s_tp
            fp += s_fp
            fn += s_fn
            n_rxn_preds += s_n_rxn_preds
            exact_match_hits += s_exact_match
            exact_match_total += 1
            total_rxns += s_total
            all_scores.append(s_scores)
            all_targets.append(s_target_bin)
            all_probs.append(s_probs)
            all_class_targets.append(s_class_targets)
            for c in range(num_classes):
                n_preds_per_class[c] += s_n_preds[c]
                n_targets_per_class[c] += s_n_targets[c]

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        exact_match_top1 = exact_match_hits / exact_match_total if exact_match_total > 0 else 0.0

        pr_auc_per_class = [0.0] * num_classes
        if all_scores:
            pooled_scores = torch.cat(all_scores).cpu()
            pooled_targets = torch.cat(all_targets).cpu()
            pooled_probs = torch.cat(all_probs).cpu()  # (M, num_classes)
            pooled_class_targets = torch.cat(all_class_targets).cpu()  # (M,)
            # Per-batch PR-AUC retained for at-a-glance progress prints; the
            # engine recomputes a *pooled* PR-AUC over the whole epoch using
            # the ``_pooled_*`` buffers below (averaging per-batch APs is
            # mathematically wrong — APs are not additive).
            pr_auc = binary_pr_auc(pooled_scores, pooled_targets)
            for c in range(num_classes):
                # One-vs-rest: score = P(class=c), positives = (target == c).
                pr_auc_per_class[c] = binary_pr_auc(
                    pooled_probs[:, c],
                    (pooled_class_targets == c).to(torch.float32),
                )
        else:
            pr_auc = 0.0
            pooled_scores = torch.empty(0)
            pooled_targets = torch.empty(0)
            pooled_probs = torch.empty(0, num_classes)
            pooled_class_targets = torch.empty(0, dtype=torch.long)

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "exact_match_top1": exact_match_top1,
            "exact_match_hits": exact_match_hits,
            "exact_match_total": exact_match_total,
            "pr_auc": pr_auc,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "n_rxn_preds": n_rxn_preds,
            "n_rxn_targets": total_rxns,
            "pr_auc_per_class": pr_auc_per_class,
            "n_preds_per_class": n_preds_per_class,
            "n_targets_per_class": n_targets_per_class,
            # Raw buffers (CPU, detached) for epoch-level pooled PR-AUC.
            # Keys prefixed with "_" so callers know they're for aggregation,
            # not direct reporting.
            "_pooled_scores": pooled_scores,
            "_pooled_targets": pooled_targets,
            "_pooled_probs": pooled_probs,
            "_pooled_class_targets": pooled_class_targets,
        }


def pooled_pr_auc(
    scores_list: list[torch.Tensor],
    targets_list: list[torch.Tensor],
    probs_list: list[torch.Tensor],
    class_targets_list: list[torch.Tensor],
    num_classes: int,
) -> tuple[float, list[float]]:
    """Pool batch-level buffers and compute (overall, per-class) PR-AUC once.

    AP is the area under a step function over the *full* score ranking, so it
    cannot be computed batch-by-batch and averaged. This helper concatenates
    all per-batch tensors and runs ``binary_pr_auc`` exactly once.

    Args:
        scores_list: per-batch 1-D tensors of "any non-no-change" scores.
        targets_list: per-batch 1-D tensors of binary (is-reaction) targets.
        probs_list: per-batch (M_b, num_classes) softmax-probability tensors.
        class_targets_list: per-batch 1-D tensors of integer class targets.
        num_classes: number of classes (used to size the per-class output).

    Returns:
        ``(overall_pr_auc, per_class_pr_auc)``. Empty inputs return zeros.
    """
    per_class = [0.0] * num_classes
    overall = 0.0
    if scores_list:
        overall = binary_pr_auc(torch.cat(scores_list), torch.cat(targets_list))
    if probs_list and num_classes > 0:
        pooled_probs = torch.cat(probs_list)
        pooled_ct = torch.cat(class_targets_list)
        for c in range(num_classes):
            per_class[c] = binary_pr_auc(pooled_probs[:, c], (pooled_ct == c).to(torch.float32))
    return overall, per_class

__all__ = [
    "binary_pr_auc",
    "MetricsComputer",
    "pooled_pr_auc",
]
