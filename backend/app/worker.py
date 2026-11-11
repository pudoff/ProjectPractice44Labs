import asyncio
import logging

from sqlalchemy.exc import SQLAlchemyError

from app.config import get_settings
from app.infra.model_assets import configure_local_models
from app.tasks.job import process_run


async def main():
    logging.basicConfig(level=logging.INFO)
    configure_local_models()
    while True:
        try:
            if await process_run() is not None:
                continue
        except SQLAlchemyError:
            logging.exception("Queue transaction failed; retrying after rollback")
        await asyncio.sleep(get_settings().poll_seconds)


if __name__ == "__main__":
    asyncio.run(main())
