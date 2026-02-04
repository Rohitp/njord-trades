"""enable pgvector extension

Revision ID: ce4acf99c9b7
Revises: 4d0cad9906fb
Create Date: 2026-02-04 17:35:21.638976

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'ce4acf99c9b7'
down_revision: Union[str, None] = '4d0cad9906fb'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Enable pgvector extension for vector similarity search
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")


def downgrade() -> None:
    # Drop pgvector extension (only if no tables depend on it)
    op.execute("DROP EXTENSION IF EXISTS vector")
