from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import String, Text, Integer, JSON, TIMESTAMP, func, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
import uuid

class Base(DeclarativeBase):
    pass

class Content(Base):
    __tablename__ = "content"
    content_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source: Mapped[str] = mapped_column(String(32), nullable=False)
    external_id: Mapped[str] = mapped_column(String(256), nullable=False)
    url: Mapped[str] = mapped_column(Text, nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    body: Mapped[str | None] = mapped_column(Text)
    authors: Mapped[list[str] | None] = mapped_column(JSON)
    categories: Mapped[list[str] | None] = mapped_column(JSON)
    language: Mapped[str | None] = mapped_column(String(8))
    published_at: Mapped[str] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    ingested_at: Mapped[str] = mapped_column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())
    as_of_ts: Mapped[str] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    revision_seq: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="active")
    metadata: Mapped[dict | None] = mapped_column(JSON)
    exact_dedupe_key: Mapped[str] = mapped_column(String(128), nullable=False)

    __table_args__ = (
        UniqueConstraint("source", "external_id", name="uq_source_external"),
        UniqueConstraint("exact_dedupe_key", name="uq_exact_dedupe"),
    )

class Outbox(Base):
    __tablename__ = "outbox"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    payload: Mapped[dict] = mapped_column(JSON, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="PENDING")
    retries: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[str] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now())

class OutboxDLQ(Base):
    __tablename__ = "outbox_dlq"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    payload: Mapped[dict] = mapped_column(JSON, nullable=False)
    last_error: Mapped[str | None] = mapped_column(Text)
    retries: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[str] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now())

class IngestAudit(Base):
    __tablename__ = "ingest_audit"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    content_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    source: Mapped[str] = mapped_column(String(32))
    external_id: Mapped[str] = mapped_column(String(256))
    reason: Mapped[str] = mapped_column(String(64))
    dedupe_key: Mapped[str | None] = mapped_column(String(128))
    details: Mapped[dict | None] = mapped_column(JSON)
    created_at: Mapped[str] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now())

class BackfillState(Base):
    __tablename__ = "backfill_state"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source: Mapped[str] = mapped_column(String(32), nullable=False)
    start_ts: Mapped[str] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    end_ts: Mapped[str] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="RUNNING")
    progress: Mapped[dict | None] = mapped_column(JSON)
    created_at: Mapped[str] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now())
    completed_at: Mapped[str | None] = mapped_column(TIMESTAMP(timezone=True))
