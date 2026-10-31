import asyncio
import logging

from app.config import get_settings
from app.tasks.job import claim_run, process_run, recover_interrupted_runs


async def main():
    logging.basicConfig(level=logging.INFO)
    await recover_interrupted_runs()
    while True:
        run_id = await claim_run()
        if run_id:
            await process_run(run_id)
        else:
            await asyncio.sleep(get_settings().poll_seconds)


if __name__ == "__main__":
    asyncio.run(main())
