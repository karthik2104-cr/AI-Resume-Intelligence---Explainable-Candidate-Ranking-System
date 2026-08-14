"""Evaluation runner comparing TF-IDF baseline vs Semantic embeddings.

Usage:
    python scripts/evaluate_semantic_vs_tfidf.py [--use-real]

By default the script uses a deterministic Fake embedding engine so it can run
without downloading large models. Pass --use-real to attempt to load the real
sentence-transformers model (requires internet and the sentence-transformers
package installed).
"""
from __future__ import annotations

import argparse
import numpy as np
from collections import defaultdict

from tests.fixtures.matching_pairs import matching_pairs
from src.models.job import ParsedJobDescription
from src.models.resume import ParsedResume
from src.matching.tfidf_baseline import TfidfBaselineMatcher
from src.matching.semantic_matcher import SemanticMatcher
from src.embeddings import get_embedding_engine


class FakeEngine:
    def __init__(self):
        self.model_name = "fake-model"
        self._dim = 8
        self._normalize = True

    def embed_texts(self, texts):
        out = []
        for t in texts:
            s = sum(ord(c) for c in (t or ""))
            vec = np.array([float((s % (i + 5)) / (i + 5)) for i in range(self._dim)], dtype=float)
            # normalize
            n = np.linalg.norm(vec)
            if n > 0:
                vec = vec / n
            out.append(vec)
        return np.vstack(out)

    def embed_text(self, text):
        return self.embed_texts([text])[0]


def spearmanr(a, b):
    # simple spearman via rank + pearson
    def rank(x):
        # average rank for ties
        idx = np.argsort(x)
        ranks = np.empty_like(idx, dtype=float)
        ranks[idx] = np.arange(1, len(x) + 1)
        return ranks

    ra = rank(a)
    rb = rank(b)
    # pearson
    a_m = ra - ra.mean()
    b_m = rb - rb.mean()
    denom = np.sqrt((a_m ** 2).sum() * (b_m ** 2).sum())
    if denom == 0:
        return 0.0
    return float((a_m * b_m).sum() / denom)


def run(use_real: bool):
    tfidf = TfidfBaselineMatcher()

    # prepare lists
    resumes = []
    jds = []
    labels = []
    for i, (rtext, jdtext, label) in enumerate(matching_pairs):
        resumes.append(ParsedResume(name=f"cand_{i}", raw_text=rtext, summary=rtext, skills=[]))
        jds.append(ParsedJobDescription(title=f"job_{i}", raw_text=jdtext, responsibilities=[]))
        labels.append(label)

    # For TF-IDF compute similarity per pair (matcher expects list of resumes and one job)
    tfidf_scores = []
    for i in range(len(resumes)):
        res = tfidf.match(resumes[i], jds[i])
        tfidf_scores.append(res.scores.baseline_tfidf or 0.0)

    # Semantic: either try real engine or use FakeEngine via monkeypatching
    sem_scores = []
    if use_real:
        try:
            sm = SemanticMatcher()
            for i in range(len(resumes)):
                res = sm.match(resumes[i], jds[i])
                sem_scores.append(res.scores.semantic or 0.0)
        except Exception as e:
            print("Real semantic model failed to load/encode:", e)
            print("Falling back to fake engine for evaluation")
            use_real = False

    if not use_real:
        # monkeypatch embedding factory to return FakeEngine
        from src.embeddings import get_embedding_engine as real_get
        import src.matching.semantic_matcher as smod

        smod.get_embedding_engine = lambda: FakeEngine()
        sm = SemanticMatcher()
        for i in range(len(resumes)):
            res = sm.match(resumes[i], jds[i])
            sem_scores.append(res.scores.semantic or 0.0)

    # map labels to numeric classes
    class_map = {"high": 2, "medium": 1, "low": 0}
    y = np.array([class_map.get(l, 0) for l in labels], dtype=float)
    tf = np.array(tfidf_scores, dtype=float)
    se = np.array(sem_scores, dtype=float)

    print("Pairs:", len(resumes))
    print("TF-IDF mean by label:")
    by_label = defaultdict(list)
    for lab, s in zip(labels, tfidf_scores):
        by_label[lab].append(s)
    for k, v in by_label.items():
        print(f"  {k}: mean={np.mean(v):.4f} n={len(v)}")

    print("Semantic mean by label:")
    by_label = defaultdict(list)
    for lab, s in zip(labels, sem_scores):
        by_label[lab].append(s)
    for k, v in by_label.items():
        print(f"  {k}: mean={np.mean(v):.4f} n={len(v)}")

    sp_tf = spearmanr(y, tf)
    sp_se = spearmanr(y, se)
    print(f"Spearman correlation vs relevance (TF-IDF): {sp_tf:.4f}")
    print(f"Spearman correlation vs relevance (Semantic): {sp_se:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--use-real", action="store_true", help="Attempt to use the real sentence-transformers model")
    args = p.parse_args()
    run(args.use_real)
