from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.pool import NullPool

from app.config import get_settings


class Base(DeclarativeBase):
    pass


database_url = get_settings().database_url
engine = create_async_engine(database_url, pool_pre_ping=True,
                             **({"poolclass": NullPool} if database_url.startswith("sqlite") else {}))
SessionLocal = async_sessionmaker(engine, expire_on_commit=False)


