from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.infra.models import Artifact, Job, Run


class JobRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get(self, job_id: str, *, lock: bool = False) -> Job | None:
        statement = select(Job).where(Job.id == job_id)
        if lock:
            statement = statement.with_for_update()
        return (await self.session.execute(statement)).scalar_one_or_none()

    async def by_event(self, channel: str, event_id: str) -> Job | None:
        statement = select(Job).where(Job.channel == channel, Job.event_id == event_id)
        return (await self.session.execute(statement)).scalar_one_or_none()

    async def artifact(self, run_id: str, name: str) -> Artifact | None:
        statement = select(Artifact).where(Artifact.run_id == run_id, Artifact.name == name)
        return (await self.session.execute(statement)).scalar_one_or_none()


class RunRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get(self, run_id: str) -> Run | None:
        return await self.session.get(Run, run_id)

    async def running_for_recovery(self) -> list[Run]:
        statement = select(Run).where(Run.state == "running").with_for_update(skip_locked=True)
        return list((await self.session.execute(statement)).scalars().all())

    async def next_queued(self) -> Run | None:
        statement = (select(Run).where(Run.state == "queued")
                     .order_by(Run.created_at, Run.number)
                     .with_for_update(skip_locked=True).limit(1))
        return (await self.session.execute(statement)).scalar_one_or_none()
