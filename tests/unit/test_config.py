"""Unit tests for configuration loading."""

from pathlib import Path

import pytest

from src.utils.config import Settings, get_settings, load_settings


def test_load_settings_from_yaml():
    settings = load_settings()
    assert isinstance(settings, Settings)
    assert settings.app.version == "2.0.0"
    assert settings.matching.baseline.vectorizer.max_features == 2000


def test_hybrid_weights_sum_reasonable():
    w = get_settings().matching.hybrid_weights
    total = w.skill_weight + w.experience_weight + w.semantic_weight
    total += w.education_weight + w.project_weight
    assert abs(total - 1.0) < 0.01


def test_resolve_path():
    settings = get_settings()
    path = settings.resolve_path("artifacts")
    assert path.name == "artifacts"
    assert path.parent.name == "v2"


def test_missing_config_raises():
    with pytest.raises(FileNotFoundError):
        load_settings(Path("/nonexistent/config.yaml"))
