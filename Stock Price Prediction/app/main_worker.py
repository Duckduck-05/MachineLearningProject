# app/main_worker.py
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from datetime import datetime
import pytz
import os

# Use relative imports
from .config import (
    SCHEDULER_TIMEZONE, DATA_UPDATE_HOUR_ET, DATA_UPDATE_MINUTE_ET,
    PREDICTION_HOUR_ET, PREDICTION_MINUTE_ET
)
from .scheduler_tasks import (
    daily_data_ingestion_and_db_update_job,
    daily_prediction_trigger_job
)

if __name__ == "__main__":
    # Set AM_I_IN_A_DOCKER_CONTAINER for config.py if running worker locally for testing
    # and config.py uses this env var for path resolution.
    # if os.getenv("AM_I_IN_A_DOCKER_CONTAINER") is None and os.getenv("RUNNING_LOCALLY_FOR_TEST") == "true":
    #     print("MAIN_WORKER: Setting AM_I_IN_A_DOCKER_CONTAINER=false for local test run.")
    #     os.environ["AM_I_IN_A_DOCKER_CONTAINER"] = "false"

    scheduler = BlockingScheduler(timezone=pytz.timezone(SCHEDULER_TIMEZONE))
    print(f"MAIN_WORKER: Scheduler initialized with timezone {SCHEDULER_TIMEZONE}.")

    # Job 1: Daily Data Ingestion and DB Price Updates
    scheduler.add_job(
        daily_data_ingestion_and_db_update_job,
        trigger=CronTrigger(
            hour=DATA_UPDATE_HOUR_ET,
            minute=DATA_UPDATE_MINUTE_ET,
            timezone=SCHEDULER_TIMEZONE
        ),
        id='daily_data_ingestion_and_db_update_job_id', # Unique ID for the job
        name='Daily Full Data Ingestion and Price DB Update',
        replace_existing=True,
        misfire_grace_time=3600 # Allow 1h late run
    )
    print(f"MAIN_WORKER: Scheduled 'Daily Full Data Ingestion and Price DB Update' job "
          f"at {DATA_UPDATE_HOUR_ET:02d}:{DATA_UPDATE_MINUTE_ET:02d} {SCHEDULER_TIMEZONE}.")

    # Job 2: Trigger Daily Predictions for all models and tickers
    scheduler.add_job(
        daily_prediction_trigger_job,
        trigger=CronTrigger(
            hour=PREDICTION_HOUR_ET,
            minute=PREDICTION_MINUTE_ET,
            timezone=SCHEDULER_TIMEZONE
        ),
        id='daily_prediction_trigger_all_job_id', # Unique ID
        name='Daily Prediction Trigger (All Tickers & Models)',
        replace_existing=True,
        misfire_grace_time=3600
    )
    print(f"MAIN_WORKER: Scheduled 'Daily Prediction Trigger (All Tickers & Models)' job "
          f"at {PREDICTION_HOUR_ET:02d}:{PREDICTION_MINUTE_ET:02d} {SCHEDULER_TIMEZONE}.")

    print(f"MAIN_WORKER: [{datetime.now()}] Scheduler starting. Press Ctrl+C to exit.")
    # NO TRY-EXCEPT around scheduler.start() as per your strict requirement
    scheduler.start()
    # Code here will only be reached if scheduler.start() is non-blocking or finishes,
    # or if an unhandled exception stops it. BlockingScheduler typically blocks.
    # If using BackgroundScheduler, you'd need a loop like `while True: time.sleep(1)`
    # For BlockingScheduler, the finally block is less critical if Ctrl+C is the main exit.
    print("MAIN_WORKER: Scheduler has been stopped (this line might not be reached with BlockingScheduler and Ctrl+C).")
    if scheduler.running: # Check if it's still running for some reason before shutdown
        scheduler.shutdown()
    print("MAIN_WORKER: Scheduler explicitly shut down.")