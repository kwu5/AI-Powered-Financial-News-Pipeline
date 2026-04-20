import logging
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from src.config import Settings
from src.pipeline import run_pipeline


logger = logging.getLogger(__name__)


def run_daily_summarization() -> None:
    """Execute the full daily pipeline. Log at every step. Never raise."""
    logger.info("Daily run starting")
    try:
        summary, count = run_pipeline()
        logger.info("Daily run complete: %d articles, %d chars summary", count, len(summary))
    except Exception:
        logger.exception("Daily run failed")


def start_scheduler() -> BackgroundScheduler:
    """Build a BackgroundScheduler with one daily job at DAILY_RUN_HOUR. Caller starts it."""
    settings = Settings()  # type: ignore
    scheduler = BackgroundScheduler()
    scheduler.add_job(
        run_daily_summarization,
        trigger=CronTrigger(hour=settings.DAILY_RUN_HOUR, minute=0),
        id="daily_summarization",
        replace_existing=True,
    )
    return scheduler
