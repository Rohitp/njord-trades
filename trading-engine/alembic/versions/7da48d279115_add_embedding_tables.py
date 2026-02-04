"""add embedding tables

Revision ID: 7da48d279115
Revises: ce4acf99c9b7
Create Date: 2026-02-04 17:52:19.790225

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from pgvector.sqlalchemy import Vector


# revision identifiers, used by Alembic.
revision: str = '7da48d279115'
down_revision: Union[str, None] = 'ce4acf99c9b7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create trade_embeddings table
    op.create_table(
        'trade_embeddings',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('trade_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('embedding', Vector(384), nullable=False),
        sa.Column('context_text', sa.Text(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['trade_id'], ['trades.id'], ondelete='CASCADE'),
    )
    op.create_index('ix_trade_embeddings_trade_id', 'trade_embeddings', ['trade_id'], unique=True)
    op.create_index(
        'ix_trade_embeddings_embedding',
        'trade_embeddings',
        ['embedding'],
        postgresql_using='ivfflat'
    )

    # Create market_condition_embeddings table
    op.create_table(
        'market_condition_embeddings',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('timestamp', sa.DateTime(timezone=True), nullable=False),
        sa.Column('embedding', Vector(384), nullable=False),
        sa.Column('context_text', sa.Text(), nullable=False),
        sa.Column('condition_metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    )
    op.create_index('ix_market_condition_embeddings_timestamp', 'market_condition_embeddings', ['timestamp'])
    op.create_index(
        'ix_market_condition_embeddings_embedding',
        'market_condition_embeddings',
        ['embedding'],
        postgresql_using='ivfflat'
    )

    # Create symbol_context_embeddings table
    op.create_table(
        'symbol_context_embeddings',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('symbol', sa.String(length=10), nullable=False),
        sa.Column('embedding', Vector(384), nullable=False),
        sa.Column('context_text', sa.Text(), nullable=False),
        sa.Column('context_type', sa.String(length=50), nullable=False),
        sa.Column('source_url', sa.Text(), nullable=True),
        sa.Column('timestamp', sa.DateTime(timezone=True), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    )
    op.create_index('ix_symbol_context_embeddings_symbol', 'symbol_context_embeddings', ['symbol'])
    op.create_index('ix_symbol_context_embeddings_timestamp', 'symbol_context_embeddings', ['timestamp'])
    op.create_index(
        'ix_symbol_context_embeddings_symbol_timestamp',
        'symbol_context_embeddings',
        ['symbol', 'timestamp']
    )
    op.create_index(
        'ix_symbol_context_embeddings_embedding',
        'symbol_context_embeddings',
        ['embedding'],
        postgresql_using='ivfflat'
    )


def downgrade() -> None:
    op.drop_index('ix_symbol_context_embeddings_embedding', table_name='symbol_context_embeddings')
    op.drop_index('ix_symbol_context_embeddings_symbol_timestamp', table_name='symbol_context_embeddings')
    op.drop_index('ix_symbol_context_embeddings_timestamp', table_name='symbol_context_embeddings')
    op.drop_index('ix_symbol_context_embeddings_symbol', table_name='symbol_context_embeddings')
    op.drop_table('symbol_context_embeddings')

    op.drop_index('ix_market_condition_embeddings_embedding', table_name='market_condition_embeddings')
    op.drop_index('ix_market_condition_embeddings_timestamp', table_name='market_condition_embeddings')
    op.drop_table('market_condition_embeddings')

    op.drop_index('ix_trade_embeddings_embedding', table_name='trade_embeddings')
    op.drop_index('ix_trade_embeddings_trade_id', table_name='trade_embeddings')
    op.drop_table('trade_embeddings')
