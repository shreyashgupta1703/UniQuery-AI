"""Retrieval evaluation harness for localrag.

Turns vague "the retrieval is good" claims into measured numbers: Recall@k,
MRR, and hit-rate over a small hand-labeled QA set, for each available embedder.
See ``harness.py`` for the runnable report and ``qa.py`` for the labels.
"""
