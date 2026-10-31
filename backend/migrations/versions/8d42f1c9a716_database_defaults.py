"""Database defaults and removal of unused columns.

Revision ID: 8d42f1c9a716
Revises: 41ece228ff73
"""

from alembic import op
import sqlalchemy as sa

revision = "8d42f1c9a716"
down_revision = "41ece228ff73"
branch_labels = None
depends_on = None


def upgrade():
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")
    op.execute("""
        CREATE FUNCTION uuid_generate_v7() RETURNS text
        LANGUAGE sql VOLATILE AS $$
          SELECT (
            lpad(to_hex(floor(extract(epoch FROM clock_timestamp()) * 1000)::bigint), 12, '0') ||
            '7' || substr(encode(gen_random_bytes(2), 'hex'), 2, 3) ||
            substr('89ab', floor(random() * 4)::int + 1, 1) ||
            substr(encode(gen_random_bytes(8), 'hex'), 2, 15)
          )::uuid::text
        $$
    """)
    for table in ("jobs", "runs", "artifacts"):
        op.alter_column(table, "id", server_default=sa.text("uuid_generate_v7()"))
    for table in ("jobs", "runs"):
        op.alter_column(table, "created_at", server_default=sa.text("now()"))
    op.alter_column("jobs", "state", server_default="queued")
    op.alter_column("jobs", "channel", server_default="web")
    op.alter_column("runs", "state", server_default="queued")
    op.alter_column("runs", "stage", server_default="queued")
    op.alter_column("runs", "parameters", server_default="{}")
    op.drop_column("jobs", "source_size")
    op.drop_column("runs", "schema_version")
    op.drop_column("artifacts", "size")
    op.drop_column("artifacts", "sha256")


def downgrade():
    op.add_column("jobs", sa.Column("source_size", sa.BigInteger(), nullable=False, server_default="0"))
    op.add_column("runs", sa.Column("schema_version", sa.String(length=16), nullable=True))
    op.add_column("artifacts", sa.Column("size", sa.BigInteger(), nullable=False, server_default="0"))
    op.add_column("artifacts", sa.Column("sha256", sa.String(length=64), nullable=False, server_default=""))
    op.alter_column("jobs", "source_size", server_default=None)
    op.alter_column("artifacts", "size", server_default=None)
    op.alter_column("artifacts", "sha256", server_default=None)
    for table in ("jobs", "runs", "artifacts"):
        op.alter_column(table, "id", server_default=None)
    for table in ("jobs", "runs"):
        op.alter_column(table, "created_at", server_default=None)
    op.alter_column("jobs", "state", server_default=None)
    op.alter_column("jobs", "channel", server_default=None)
    op.alter_column("runs", "state", server_default=None)
    op.alter_column("runs", "stage", server_default=None)
    op.alter_column("runs", "parameters", server_default=None)
    op.execute("DROP FUNCTION uuid_generate_v7()")
