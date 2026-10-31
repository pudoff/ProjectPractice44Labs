"""Single worker polling a durable PostgreSQL queue."""
import logging
import os
import time
from pathlib import Path

from . import db
from .models import paths
from .settings import data_dir

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger(__name__)

def main():
    # Check at startup, not after claiming a job: no implicit model download.
    paths(check=True)
    data_dir().mkdir(parents=True, exist_ok=True)
    db.setup()
    recovered = db.recover_on_worker_start()
    LOG.info("worker ready; recovered=%s", len(recovered))
    from runs.klapan_png2dxf.pipeline_b.pipeline import run
    interval = max(0.1, float(os.environ.get("WORKER_POLL_SECONDS", "2")))
    while True:
        job = db.claim()
        if job is None:
            time.sleep(interval)
            continue
        id_, input_file, output_file, report_file = job
        LOG.info("processing job=%s", id_)
        try:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            run(input_file, output_file, report_path=report_file, strict_ocr=True)
            db.finish(id_)
            LOG.info("completed job=%s", id_)
        except Exception as exc:
            LOG.exception("failed job=%s", id_)
            db.finish(id_, error=f"{type(exc).__name__}: {exc}")

if __name__ == "__main__":
    main()
