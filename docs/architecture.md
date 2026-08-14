# V2 Architecture

> **Status:** Phase 6 — unified entity intelligence and skill gap. See [entity_intelligence.md](entity_intelligence.md).

## Overview

V2 follows a layered, interface-driven design. Each layer communicates through Pydantic models and abstract base classes, enabling independent testing and phased implementation.

## Document Ingestion (Phase 3)

```mermaid
flowchart TD
    File[Uploaded File] --> Factory[IngestionFactory]
    Factory -->|".pdf"| PDF[PdfIngester]
    Factory -->|".docx"| DOCX[DocxIngester]
    Factory -->|".txt"| TXT[TxtIngester]
    PDF --> Validate[DocumentValidator]
    DOCX --> Validate
    TXT --> Validate
    Validate --> Document[Document abstraction]
    Document --> Parser[Future Resume / JD Parser]
```

### Ingestion pipeline

1. **Factory routing** — `IngestionFactory` selects the ingester by file extension.
2. **Pre-validation** — extension, file size, magic bytes, MIME hint (configurable).
3. **Format extraction** — per-page PDF text, DOCX paragraphs/tables, TXT with encoding fallbacks.
4. **Post-validation** — reject empty extracted text when configured.
5. **Document model** — unified `Document` with `document_id`, `raw_text`, pages, metadata, warnings.

### Error taxonomy

| Error | When |
|-------|------|
| `UnsupportedFileTypeError` | Bad extension or magic bytes |
| `FileTooLargeError` | Exceeds `max_upload_size_mb` |
| `CorruptedFileError` | PDF/DOCX parse failure |
| `ExtractionFailureError` | Decode/extraction failure |
| `EmptyDocumentError` | Zero-byte file or empty extracted text |

Raw library exceptions are logged internally; callers receive domain-level errors.

## Resume Parsing (Phase 4)

```mermaid
flowchart TD
    Document[Document] --> Parser[HeuristicResumeParser]
    Parser --> Split[Section Splitter]
    Parser --> Contact[Contact Extractor]
    Split --> Entries[Entry Extractors]
    Contact --> Parsed[ParsedResume]
    Entries --> Parsed
    Parsed --> Quality[Quality Assessment]
```

1. **Section splitting** — detect headings (ALL CAPS, colons, markdown `##`) via configurable keywords.
2. **Contact extraction** — name, email, phone from header block (pre-first-section text).
3. **Entry extraction** — skills, experience, education, projects, certifications from section content.
4. **Quality assessment** — `high` / `medium` / `low` based on detected structure (not statistical confidence).

Pipeline: `ingest_document()` → `HeuristicResumeParser.parse()` → `ParsedResume`

## Job Description Parsing (Phase 5)

```mermaid
flowchart TD
    JD[Job Description Text] --> JParser[HeuristicJobDescriptionParser]
    JParser --> JSplit[JD Section Splitter]
    JParser --> JTitle[Title / Seniority Extractor]
    JSplit --> JReq[Requirement Classifier]
    JReq --> Skills[Shared Skill Normalizer]
    Skills --> ParsedJD[ParsedJobDescription]
    JTitle --> ParsedJD
```

See [job_parsing.md](job_parsing.md) for full documentation.

## Entity Intelligence & Skill Gap (Phase 6)

```mermaid
flowchart TD
    PR[ParsedResume] --> EE1[EntityExtractor]
    PJ[ParsedJobDescription] --> EE2[EntityExtractor]
    EE1 --> CP[Candidate Profile]
    EE2 --> JP[Job Profile]
    CP --> SG[Skill Gap Engine]
    JP --> SG
    SG --> SGR[SkillGapResult]
```

See [entity_intelligence.md](entity_intelligence.md) for full documentation.

## Dual Input Pipeline (Pre-Matching)

```mermaid
flowchart LR
    Doc[Document] --> RParser[Resume Parser]
    RParser --> Resume[ParsedResume]
    JDText[JD Text] --> JParser[JD Parser]
    JParser --> Job[ParsedJobDescription]
    Resume --> Match[Future Matching Engine]
    Job --> Match
```

## Data Flow (Target State)

```mermaid
flowchart LR
    Upload[File Upload] --> Ingest[Ingestion]
    Ingest --> Doc[Document Model]
    Doc --> RParse[Resume Parser]
    Doc --> JParse[JD Parser]
    RParse --> Resume[ParsedResume]
    JParse --> Job[ParsedJobDescription]
    Resume --> Match[Matching Engine]
    Job --> Match
    Match --> Score[Scoring Engine]
    Score --> Rank[Ranking Engine]
    Rank --> Explain[Explainability]
    Explain --> LLM[Optional LLM]
    Rank --> API[FastAPI Response]
    API --> UI[Streamlit UI]
```

## Implemented

- Configuration management (`src/utils/config.py`)
- Pydantic domain models (`src/models/`)
- Abstract interfaces for all major components
- TF-IDF baseline matcher and ranker
- **Document ingestion** — PDF, DOCX, TXT via `IngestionFactory`
- **Resume parsing** — `HeuristicResumeParser` with section/contact/entry extraction
- **JD parsing** — `HeuristicJobDescriptionParser` with required/preferred/**mentioned** skills
- **Entity extraction** — unified `EntityExtractor` + shared skill normalization
- **Skill gap analysis** — deterministic `compute_skill_gap()`
- Unit tests for baseline, preprocessing, config, and ingestion

## Planned Layers

Detailed implementation docs will be added as each phase completes.

| Component | Module | Phase |
|-----------|--------|-------|
| Ingestion | `src/ingestion/` | **3 — done** |
| Resume parsing | `src/parsing/resume_parser.py` | **4 — done** |
| JD parsing | `src/parsing/job_parser.py` | **5 — done** |
| Embeddings | `src/embeddings/` | 7 |
| Hybrid scoring | `src/scoring/` | 8 |
| Explainability | `src/explainability/` | 10 |
| LLM | `src/llm/` | 13 |
| API | `api/` | 14 |
| Frontend | `app/` | 15 |
| Database | `src/storage/` | 16 |

## Design Principles

1. **LLM does not score** — numerical scores come from deterministic engines
2. **Explainability from features** — no fabricated explanations
3. **Fallback architecture** — core matching works without LLM or vector index
4. **Configurable weights** — no magic numbers in business logic
5. **Legacy preserved** — V1 code untouched at repository root
