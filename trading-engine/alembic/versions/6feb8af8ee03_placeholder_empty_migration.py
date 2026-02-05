"""placeholder empty migration

Revision ID: 6feb8af8ee03
Revises: 397ca6a8afc7
Create Date: 2026-02-05 08:20:00.000000

This migration intentionally does nothing. It exists to satisfy references
to the previously generated revision id 6feb8af8ee03, which was removed
because its schema changes were already captured in revision 397ca6a8afc7.
"""

from typing import Sequence, Union

from alembic import op  # noqa: F401 (kept for consistency)
import sqlalchemy as sa  # noqa: F401 (kept for consistency)


# revision identifiers, used by Alembic.
revision: str = "6feb8af8ee03"
down_revision: Union[str, None] = "397ca6a8afc7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """No-op upgrade."""
    pass


def downgrade() -> None:
    """No-op downgrade."""
    pass
