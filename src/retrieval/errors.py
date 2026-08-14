class RetrievalError(Exception):
    """Base retrieval exception."""


class CandidateNotFoundError(RetrievalError):
    pass


class InvalidEmbeddingError(RetrievalError):
    pass


class DuplicateCandidateError(RetrievalError):
    pass


class EmbeddingDimensionMismatchError(RetrievalError):
    pass
