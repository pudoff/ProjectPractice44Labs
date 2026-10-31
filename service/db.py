"""PostgreSQL-backed single-step queue. No additional broker required."""
import uuid
import psycopg

from .settings import db_dsn

DDL = """
CREATE TABLE IF NOT EXISTS jobs (
  id uuid PRIMARY KEY,
  original_name text NOT NULL,
  input_path text NOT NULL,
  output_path text NOT NULL,
  report_path text NOT NULL,
  status text NOT NULL CHECK (status IN ('queued','running','completed','failed')),
  error text,
  created_at timestamptz NOT NULL DEFAULT now(),
  started_at timestamptz,
  finished_at timestamptz
);
CREATE INDEX IF NOT EXISTS ix_jobs_status_created ON jobs(status, created_at);
"""

def connect():
    return psycopg.connect(db_dsn())

def setup():
    with connect() as conn:
        conn.execute(DDL)

def enqueue(job_id: uuid.UUID, original_name: str, in_path: str, out_path: str, rep_path: str):
    with connect() as conn:
        conn.execute(
            """INSERT INTO jobs(id, original_name, input_path, output_path, report_path, status)
               VALUES (%s, %s, %s, %s, %s, 'queued')""",
            (job_id, original_name, in_path, out_path, rep_path),
        )

def get_job(job_id):
    with connect() as conn:
        return conn.execute(
            """SELECT id::text, original_name, status, error, created_at,
                      started_at, finished_at, output_path, report_path
               FROM jobs WHERE id = %s""", (job_id,)
        ).fetchone()

def claim():
    with connect() as conn:
        return conn.execute(
            """WITH next AS (
                 SELECT id FROM jobs WHERE status = 'queued'
                 ORDER BY created_at, id FOR UPDATE SKIP LOCKED LIMIT 1
               )
               UPDATE jobs j SET status='running', started_at=now(), error=NULL
               FROM next WHERE j.id = next.id
               RETURNING j.id::text, j.input_path, j.output_path, j.report_path""",
        ).fetchone()

def finish(job_id, error=None):
    with connect() as conn:
        if error:
            conn.execute(
                "UPDATE jobs SET status='failed', error=%s, finished_at=now() WHERE id=%s",
                (str(error)[:2000], job_id))
        else:
            conn.execute(
                "UPDATE jobs SET status='completed', finished_at=now() WHERE id=%s",
                (job_id,))

def recover_on_worker_start():
    # One worker is supported; on restart, retry interrupted jobs.
    with connect() as conn:
        return conn.execute(
            "UPDATE jobs SET status='queued', started_at=NULL "
            "WHERE status='running' RETURNING id"
        ).fetchall()
