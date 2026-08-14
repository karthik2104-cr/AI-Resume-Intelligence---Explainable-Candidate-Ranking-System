"""Configuration loading and validation for V2."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


V2_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = V2_ROOT / "configs" / "config.yaml"
SKILLS_CONFIG_PATH = V2_ROOT / "configs" / "skills.yaml"


class AppConfig(BaseModel):
    name: str
    version: str
    environment: str = "development"


class PathsConfig(BaseModel):
    artifacts_dir: str = "artifacts"
    data_raw_dir: str = "data/raw"
    data_processed_dir: str = "data/processed"
    data_evaluation_dir: str = "data/evaluation"
    experiments_dir: str = "artifacts/experiments"


class IngestionConfig(BaseModel):
    allowed_extensions: list[str] = Field(default_factory=lambda: [".pdf", ".docx", ".txt"])
    max_upload_size_mb: int = 10
    validate_magic_bytes: bool = True
    reject_empty_extraction: bool = True
    txt_encodings: list[str] = Field(
        default_factory=lambda: ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
    )


class PreprocessingConfig(BaseModel):
    lowercase: bool = True
    remove_urls: bool = True
    remove_non_ascii: bool = True
    normalize_whitespace: bool = True


class ParsingConfig(BaseModel):
    max_header_line_length: int = 60
    min_sections_for_high_quality: int = 2
    section_heading_keywords: dict[str, list[str]] = Field(
        default_factory=lambda: {
            "summary": ["summary", "profile", "objective", "about me", "professional summary"],
            "experience": ["experience", "work experience", "employment", "professional experience"],
            "education": ["education", "academic background", "qualifications"],
            "skills": ["skills", "technical skills", "core competencies", "expertise"],
            "projects": ["projects", "personal projects", "key projects"],
            "certifications": ["certifications", "certificates", "licenses"],
            "achievements": ["achievements", "awards", "honors"],
            "publications": ["publications"],
            "languages": ["languages", "language proficiency"],
        }
    )


class TfidfVectorizerConfig(BaseModel):
    max_features: int = 2000
    sublinear_tf: bool = True
    stop_words: str = "english"
    ngram_range: list[int] = Field(default_factory=lambda: [1, 1])


class BaselineMatchingConfig(BaseModel):
    vectorizer: TfidfVectorizerConfig = Field(default_factory=TfidfVectorizerConfig)
    similarity_metric: str = "cosine"


class HybridWeightsConfig(BaseModel):
    skill_weight: float = 0.35
    experience_weight: float = 0.25
    semantic_weight: float = 0.20
    education_weight: float = 0.10
    project_weight: float = 0.10


class MatchingConfig(BaseModel):
    baseline: BaselineMatchingConfig = Field(default_factory=BaselineMatchingConfig)
    hybrid_weights: HybridWeightsConfig = Field(default_factory=HybridWeightsConfig)


class EmbeddingsConfig(BaseModel):
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 32
    cache_embeddings: bool = True


class LLMConfig(BaseModel):
    enabled: bool = False
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    api_key_env: str = "OPENAI_API_KEY"
    timeout_seconds: int = 30
    max_retries: int = 2


class DatabaseConfig(BaseModel):
    url: str = "sqlite:///artifacts/resume_intelligence.db"


class RankingConfig(BaseModel):
    default_top_k: int = 10


class DuplicatesConfig(BaseModel):
    exact_hash_algorithm: str = "sha256"
    near_duplicate_threshold: float = 0.95


class ObservabilityConfig(BaseModel):
    log_level: str = "INFO"
    log_format: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


class ResponsibleAIConfig(BaseModel):
    exclude_from_scoring: list[str] = Field(
        default_factory=lambda: ["name", "email", "phone", "gender", "age", "photo"]
    )


class SkillsConfig(BaseModel):
    vocabulary: dict[str, list[str]] = Field(default_factory=dict)
    skill_categories: dict[str, str] = Field(default_factory=dict)
    soft_skills: list[str] = Field(default_factory=list)
    domains: list[str] = Field(default_factory=list)


class JobParsingConfig(BaseModel):
    max_header_line_length: int = 70
    section_aliases: dict[str, list[str]] = Field(default_factory=dict)
    required_phrases: list[str] = Field(default_factory=list)
    preferred_phrases: list[str] = Field(default_factory=list)
    seniority_levels: list[str] = Field(default_factory=list)
    employment_types: list[str] = Field(default_factory=list)
    work_modes: list[str] = Field(default_factory=list)
    soft_skills: list[str] = Field(default_factory=list)
    domains: list[str] = Field(default_factory=list)
    degree_aliases: dict[str, list[str]] = Field(default_factory=dict)


class Settings(BaseModel):
    """Root configuration object."""

    app: AppConfig
    paths: PathsConfig = Field(default_factory=PathsConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    parsing: ParsingConfig = Field(default_factory=ParsingConfig)
    skills: SkillsConfig = Field(default_factory=SkillsConfig)
    job_parsing: JobParsingConfig = Field(default_factory=JobParsingConfig)
    matching: MatchingConfig = Field(default_factory=MatchingConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    ranking: RankingConfig = Field(default_factory=RankingConfig)
    duplicates: DuplicatesConfig = Field(default_factory=DuplicatesConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    responsible_ai: ResponsibleAIConfig = Field(default_factory=ResponsibleAIConfig)

    def resolve_path(self, relative: str) -> Path:
        """Resolve a config-relative path against the V2 root."""
        return V2_ROOT / relative

    @property
    def artifacts_path(self) -> Path:
        return self.resolve_path(self.paths.artifacts_dir)


def _apply_env_overrides(raw: dict[str, Any]) -> dict[str, Any]:
    """Apply environment variable overrides to raw config dict."""
    if api_key := os.getenv("OPENAI_API_KEY"):
        raw.setdefault("llm", {})["enabled"] = bool(api_key) or raw.get("llm", {}).get("enabled", False)

    if db_url := os.getenv("DATABASE_URL"):
        raw.setdefault("database", {})["url"] = db_url

    if env := os.getenv("APP_ENVIRONMENT"):
        raw.setdefault("app", {})["environment"] = env

    if log_level := os.getenv("LOG_LEVEL"):
        raw.setdefault("observability", {})["log_level"] = log_level

    return raw


def load_settings(config_path: Path | None = None) -> Settings:
    """Load settings from YAML file with optional env overrides."""
    path = config_path or DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    if SKILLS_CONFIG_PATH.exists():
        with SKILLS_CONFIG_PATH.open("r", encoding="utf-8") as fh:
            skills_data = yaml.safe_load(fh) or {}
        if skills := skills_data.get("skills"):
            raw.setdefault("skills", {}).update(skills)
        if job_parsing := skills_data.get("job_parsing"):
            merged = raw.setdefault("job_parsing", {})
            for key, value in job_parsing.items():
                if key in ("soft_skills", "domains") and key in merged:
                    continue  # prefer skills.* canonical lists
                merged[key] = value

    raw = _apply_env_overrides(raw)
    return Settings.model_validate(raw)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings singleton."""
    return load_settings()
