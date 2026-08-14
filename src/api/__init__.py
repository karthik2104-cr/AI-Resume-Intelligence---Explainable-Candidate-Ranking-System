"""API package for v2 application.

Contains FastAPI application wiring and route registration. The API uses the
existing ScreeningService for all screening logic — route handlers remain
thin and perform validation + ingestion/parsing only.
"""
__all__ = ["main", "routes", "schemas"]
