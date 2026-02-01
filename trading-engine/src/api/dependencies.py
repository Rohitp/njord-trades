"""Shared dependencies for API endpoints.

This module centralizes FastAPI dependencies that are shared across multiple routers.
Currently provides database session dependency, but can be extended with:
- Authentication dependencies (get_current_user)
- Authorization dependencies (require_role)
- Rate limiting dependencies
- etc.
"""

from src.database.connection import get_session

# Alias for cleaner imports in routers
# All routers use get_db() instead of importing from database.connection
get_db = get_session

