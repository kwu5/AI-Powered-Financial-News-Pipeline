import argparse
import logging
import time

import uvicorn

from src.api.main import app
from src.pipeline import run_pipeline
from src.scheduler.jobs import start_scheduler


def main() -> None:
    # Logging config belongs at the entry point, not in library modules.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Daily Financial News Summarizer")
    parser.add_argument(
        "--mode",
        choices=["api", "run-once", "scheduler"],
        default="api",
        help="Which face of the app to run (default: api)",
    )
    parser.add_argument("--port", type=int, default=8000, help="Port for API mode (default: 8000)")
    args = parser.parse_args()

    if args.mode == "api":
        # Note: uvicorn.run blocks until SIGINT. The scheduler runs in a
        # daemon thread, so it dies with the process — no explicit shutdown needed.
        scheduler = start_scheduler()
        scheduler.start()
        uvicorn.run(app,host="0.0.0.0", port=args.port)
        

    elif args.mode == "run-once":
        summary, count = run_pipeline()
        logger.info("Done: %d articles, %d chars", count, len(summary))
        

    elif args.mode == "scheduler":
        scheduler = start_scheduler()
        scheduler.start()
        try:
            while True:     time.sleep(3600)
        except (KeyboardInterrupt, SystemExit):
            scheduler.shutdown()


if __name__ == "__main__":
    main()
