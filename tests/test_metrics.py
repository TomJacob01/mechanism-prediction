"""Tests for metric helpers."""

import math

import torch

from mech_uspto.training.metrics import binary_pr_auc


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
