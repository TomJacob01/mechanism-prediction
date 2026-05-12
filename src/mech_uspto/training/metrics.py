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
    k: int,
) -> tuple[int, int, int, int, int, torch.Tensor, torch.Tensor]:
    """Per-sample (tp, fp, fn, topk_hits, total_rxns, scores, targets) over upper-triangle pairs.

    ``scores`` and ``targets`` are returned so the caller can pool them across
    the batch / epoch and compute a single PR-AUC.
    """
    mol_preds = torch.argmax(mol_logits, dim=-1)

    # Reaction center = anything not the "no change" class (index 1 for 3-class,
    # 2 for 5-class — but the original code conventions class 1 as "no change"
    # for 3-class targets after the shift {-1,0,1} → {0,1,2}). We keep the
    # original 3-class convention; for end-to-end the head is bigger but
    # the same comparison "!= 1" treats class index 1 as the no-change class
    # in both schemes after target shifting (1 maps to Δ=0 in stepwise and
    # to Δ=-1 in 5-class — see note in metrics docs).
    is_rxn_target = mol_targets != 1
    is_rxn_pred = mol_preds != 1

    tp = (is_rxn_pred & is_rxn_target & (mol_preds == mol_targets)).sum().item()
    fp = (is_rxn_pred & (~is_rxn_target | (mol_preds != mol_targets))).sum().item()
    fn = (is_rxn_target & (~is_rxn_pred | (mol_preds != mol_targets))).sum().item()

    mol_total_rxns = is_rxn_target.sum().item()

    probs = torch.softmax(mol_logits, dim=-1)
    # "Any reaction" score = combined probability of the rare classes (first + last).
    rxn_scores = probs[:, 0] + probs[:, -1]

    topk_hits = 0
    if mol_total_rxns > 0:
        k_to_use = min(k, len(rxn_scores))
        _, top_indices = torch.topk(rxn_scores, k_to_use)
        target_rxn_indices = torch.where(is_rxn_target)[0]
        for tr_idx in target_rxn_indices:
            if tr_idx in top_indices:
                topk_hits += 1

    return tp, fp, fn, topk_hits, mol_total_rxns, rxn_scores.detach(), is_rxn_target.detach()


class MetricsComputer:
    """Compute precision / recall / F1 / top-K for delta predictions."""

    @staticmethod
    def get_mechanism_metrics(
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
        k: int = 3,
    ) -> dict[str, float]:
        """Reaction-center metrics over the upper-triangle of each batch sample."""
        B, N, _, _ = logits.shape
        mask_2d = mask.unsqueeze(1) * mask.unsqueeze(2)
        triu_indices = torch.triu_indices(N, N, offset=1, device=logits.device)

        tp, fp, fn, topk_hits, total_rxns = 0, 0, 0, 0, 0
        all_scores: list[torch.Tensor] = []
        all_targets: list[torch.Tensor] = []

        for b in range(B):
            m = mask_2d[b]
            valid = m[triu_indices[0], triu_indices[1]]
            idx0, idx1 = triu_indices[0][valid], triu_indices[1][valid]
            if len(idx0) == 0:
                continue

            mol_logits = logits[b, idx0, idx1]
            mol_targets = targets[b, idx0, idx1]
            s_tp, s_fp, s_fn, s_topk, s_total, s_scores, s_target_bin = _metrics_for_sample(
                mol_logits, mol_targets, k
            )
            tp += s_tp
            fp += s_fp
            fn += s_fn
            topk_hits += s_topk
            total_rxns += s_total
            all_scores.append(s_scores)
            all_targets.append(s_target_bin)

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        topk_acc = topk_hits / (total_rxns + 1e-8)

        if all_scores:
            pooled_scores = torch.cat(all_scores)
            pooled_targets = torch.cat(all_targets)
            pr_auc = binary_pr_auc(pooled_scores, pooled_targets)
        else:
            pr_auc = 0.0

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "topk_acc": topk_acc,
            "pr_auc": pr_auc,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
