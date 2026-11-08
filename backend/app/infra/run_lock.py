"""Session advisory locks survive short queue transactions and die with the worker.

Use a dedicated connection, never a pooled session that could retain a lock after
returning to the pool. SQLite's in-process fallback is only for unit tests.
"""
import hashlib

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

from app.infra.db import engine


class RunLockLost(RuntimeError):
    pass


_local_locks: set[str] = set()


class RunLock:
    def __init__(self, run_id: str, connection: AsyncConnection | None = None):
        self.run_id = run_id
        self.connection = connection
        self.backend_pid = None
        self.lost = False
        self.key = int.from_bytes(hashlib.sha256(run_id.encode()).digest()[:8], "big", signed=True)

    @classmethod
    async def acquire(cls, run_id: str) -> "RunLock | None":
        lock = cls(run_id)
        if engine.dialect.name != "postgresql":
            if run_id in _local_locks:
                return None
            _local_locks.add(run_id)
            return lock
        connection = await engine.connect()
        try:
            acquired = await connection.scalar(text("SELECT pg_try_advisory_lock(:key)"), {"key": lock.key})
            await connection.commit()
            if acquired:
                lock.backend_pid = await connection.scalar(text("SELECT pg_backend_pid()"))
                await connection.commit()
                lock.connection = connection
                return lock
        except BaseException:
            await connection.invalidate()
            raise
        finally:
            if lock.connection is None:
                await connection.close()
        return None

    async def assert_owned(self):
        if self.connection is None:
            if self.run_id not in _local_locks:
                raise RunLockLost(self.run_id)
            return
        try:
            if self.connection.closed or self.connection.invalidated:
                raise RunLockLost(self.run_id)
            owned = await self.connection.scalar(text("""
                SELECT pg_backend_pid() = :pid AND EXISTS (
                    SELECT 1 FROM pg_locks WHERE pid = pg_backend_pid()
                    AND locktype = 'advisory' AND granted
                    AND classid::bigint = :hi AND objid::bigint = :lo AND objsubid = 1)
            """), {"pid": self.backend_pid, "hi": (self.key >> 32) & 0xffffffff,
                   "lo": self.key & 0xffffffff})
            await self.connection.commit()
            if not owned:
                raise RunLockLost(self.run_id)
        except Exception as exc:
            self.lost = True
            raise RunLockLost(self.run_id) from exc

    async def release(self):
        if self.connection is None:
            _local_locks.discard(self.run_id)
            return
        if self.lost:
            await self.connection.invalidate()
            await self.connection.close()
            return
        try:
            await self.connection.execute(text("SELECT pg_advisory_unlock(:key)"), {"key": self.key})
            await self.connection.commit()
        except BaseException:
            await self.connection.invalidate()
            raise
        finally:
            await self.connection.close()


claims: dict[str, RunLock] = {}
processing: set[str] = set()


async def release_claim(run_id: str):
    lock = claims.pop(run_id, None)
    if lock:
        await lock.release()
