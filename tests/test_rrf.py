"""Reciprocal rank fusion.

Regression: fusion used to key on `chunk_index` alone, which the chunker restarts per
document, so chunks from different files collapsed into one entry.
"""

from __future__ import annotations

import pytest

from api.retriever import RRF_K, rrf_merge
from tests.conftest import make_document, make_hit


def test_same_chunk_index_from_different_files_stays_separate() -> None:
    dense = [make_hit("tutorial/path-params.md", 0, score=0.9)]
    bm25 = [(make_document("advanced/security.md", 0), 4.2)]

    merged = rrf_merge(dense, bm25, top_k=10)

    assert len(merged) == 2, "chunks sharing a chunk_index across files must not be merged"
    assert {hit.chunk_key for hit in merged} == {
        ("tutorial/path-params.md", 0),
        ("advanced/security.md", 0),
    }


def test_same_chunk_from_both_arms_is_merged_and_scores_add() -> None:
    hit = make_hit("tutorial/path-params.md", 3, score=0.9)
    dense = [hit]
    bm25 = [(make_document("tutorial/path-params.md", 3), 4.2)]

    merged = rrf_merge(dense, bm25, top_k=10)

    assert len(merged) == 1
    # Rank 0 in both arms.
    assert merged[0].score == pytest.approx(2 * (1 / (RRF_K + 1)))


def test_scores_follow_the_rrf_formula_and_ranking_is_descending() -> None:
    dense = [make_hit("a.md", i) for i in range(3)]
    merged = rrf_merge(dense, [], top_k=3)

    assert [hit.chunk_key for hit in merged] == [("a.md", 0), ("a.md", 1), ("a.md", 2)]
    for rank, hit in enumerate(merged):
        assert hit.score == pytest.approx(1 / (RRF_K + rank + 1))
    assert merged[0].score > merged[1].score > merged[2].score


def test_a_document_ranked_by_both_arms_outranks_one_ranked_by_only_one() -> None:
    # "b.md" is second in each arm; "a.md" is first in the dense arm only.
    dense = [make_hit("a.md", 0), make_hit("b.md", 0)]
    bm25 = [(make_document("c.md", 0), 5.0), (make_document("b.md", 0), 4.0)]

    merged = rrf_merge(dense, bm25, top_k=3)

    assert merged[0].chunk_key == ("b.md", 0)


def test_top_k_truncates_the_fused_ranking() -> None:
    dense = [make_hit("a.md", i) for i in range(10)]
    assert len(rrf_merge(dense, [], top_k=4)) == 4


def test_empty_inputs_produce_no_hits() -> None:
    assert rrf_merge([], [], top_k=5) == []
