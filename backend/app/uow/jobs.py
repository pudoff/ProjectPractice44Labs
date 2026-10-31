from sqlalchemy.ext.asyncio import AsyncSession

from app.repositories.jobs import JobRepository


class JobUnitOfWork:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.jobs = JobRepository(session)

    async def commit(self) -> None:
        await self.session.commit()

    def add(self, item: object) -> None:
        self.session.add(item)

    async def flush(self) -> None:
        await self.session.flush()

    async def refresh_runs(self, job) -> None:
        await self.session.refresh(job, attribute_names=["runs"])
        for run in job.runs:
            await self.session.refresh(run, attribute_names=["artifacts"])

    async def rollback(self) -> None:
        await self.session.rollback()
