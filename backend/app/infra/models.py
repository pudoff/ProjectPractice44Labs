from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, UniqueConstraint, func, text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.infra.db import Base


class Job(Base):
    __tablename__ = "jobs"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, server_default=text("uuid_generate_v7()"))
    filename: Mapped[str] = mapped_column(String(255))
    source_sha256: Mapped[str] = mapped_column(String(64))
    source_ext: Mapped[str] = mapped_column(String(8))
    state: Mapped[str] = mapped_column(String(20), server_default="queued")
    channel: Mapped[str] = mapped_column(String(32), server_default="web")
    conversation_id: Mapped[str | None] = mapped_column(String(255))
    event_id: Mapped[str | None] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    runs: Mapped[list["Run"]] = relationship(back_populates="job", order_by="Run.number", lazy="selectin")
    __table_args__ = (UniqueConstraint("channel", "event_id", name="uq_channel_event"),)


class Run(Base):
    __tablename__ = "runs"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, server_default=text("uuid_generate_v7()"))
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"), index=True)
    number: Mapped[int] = mapped_column(Integer)
    state: Mapped[str] = mapped_column(String(20), server_default="queued", index=True)
    stage: Mapped[str] = mapped_column(String(40), server_default="queued")
    parameters: Mapped[str] = mapped_column(Text, server_default="{}")
    error_code: Mapped[str | None] = mapped_column(String(80))
    error_message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    job: Mapped[Job] = relationship(back_populates="runs", lazy="selectin")
    artifacts: Mapped[list["Artifact"]] = relationship(back_populates="run", lazy="selectin")
    __table_args__ = (UniqueConstraint("job_id", "number", name="uq_run_number"),)


class Artifact(Base):
    __tablename__ = "artifacts"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, server_default=text("uuid_generate_v7()"))
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(80))
    relative_path: Mapped[str] = mapped_column(Text)
    run: Mapped[Run] = relationship(back_populates="artifacts")
    __table_args__ = (UniqueConstraint("run_id", "name", name="uq_run_artifact"),)
