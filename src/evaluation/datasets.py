"""Small controlled fixtures for demonstration benchmarking.

Provides a reproducible tiny dataset with clearly labeled relevance for a
single job description. This is intentionally small and synthetic — it is
for engineering demonstration only and not a substitute for real benchmarks.
"""
from typing import Dict, List, Tuple


def load_small_fixture() -> Tuple[str, Dict[str, str], Dict[str, int]]:
    """Return (job_description, candidate_texts, relevance_mapping).

    relevance_mapping maps candidate_id -> relevance (0=no, 1=partly, 2=high)
    """
    jd = (
        "Senior Machine Learning Engineer with experience building production "
        "ML pipelines, expertise in scikit-learn, PyTorch, model deployment, "
        "and strong Python coding skills."
    )

    candidates = {
        "cand_A": "Experienced data scientist. Built models with scikit-learn and TensorFlow. "
                  "Experienced in deployment and Python.",
        "cand_B": "Backend engineer focused on Java and microservices. Some exposure to Docker.",
        "cand_C": "Machine learning engineer. Used PyTorch and scikit-learn for research and production.",
        "cand_D": "Recent graduate. Coursework in machine learning, did thesis on NLP using transformers.",
        "cand_E": "Business analyst with heavy Excel use and reporting. No ML experience.",
    }

    relevance = {
        "cand_A": 2,  # highly relevant
        "cand_B": 0,  # irrelevant
        "cand_C": 2,  # highly relevant
        "cand_D": 1,  # partially relevant
        "cand_E": 0,  # irrelevant
    }

    return jd, candidates, relevance
