# app/main_worker.py
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from datetime import datetime
import pytz
import os

from .config import (
    SCHEDULER_TIMEZONE, DATA_UPDATE_HOUR_ET, DATA_UPDATE_MINUTE_ET,
    PREDICTION_HOUR_ET, PREDICTION_MINUTE_ET
)
from .scheduler_tasks import (
    daily_data_ingestion_and_db_update_job,
    daily_prediction_trigger_job
)

if __name__ == "__main__":
    scheduler = BlockingScheduler(timezone=pytz.timezone(SCHEDULER_TIMEZONE))
    print(f"MAIN_WORKER: Scheduler initialized with timezone {SCHEDULER_TIMEZONE}.")

    # Data ingestion every 4 minutes
    # scheduler.add_job(
    #     daily_data_ingestion_and_db_update_job,
    #     trigger=IntervalTrigger(minutes=4),
    #     id='test_data_ingestion_job_id',
    #     name='TEST Data Ingestion Every 4 Minutes',
    #     replace_existing=True,
    #     misfire_grace_time=120
    # )
    # print(f"MAIN_WORKER: Scheduled TEST data ingestion job to run every 4 minutes.")

    # Job 1: Daily Data Ingestion and DB Price Updates
    scheduler.add_job(
        daily_data_ingestion_and_db_update_job,
        trigger=CronTrigger(
            hour=DATA_UPDATE_HOUR_ET,
            minute=DATA_UPDATE_MINUTE_ET,
            timezone=SCHEDULER_TIMEZONE
        ),
        id='daily_data_ingestion_and_db_update_job_id',
        name='Daily Full Data Ingestion and Price DB Update',
        replace_existing=True,
        misfire_grace_time=3600
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
        id='daily_prediction_trigger_all_job_id',
        name='Daily Prediction Trigger (All Tickers & Models)',
        replace_existing=True,
        misfire_grace_time=3600
    )
    print(f"MAIN_WORKER: Scheduled 'Daily Prediction Trigger (All Tickers & Models)' job "
          f"at {PREDICTION_HOUR_ET:02d}:{PREDICTION_MINUTE_ET:02d} {SCHEDULER_TIMEZONE}.")

    print(f"MAIN_WORKER: [{datetime.now()}] Scheduler starting. Press Ctrl+C to exit.")
    scheduler.start()
    print("MAIN_WORKER: Scheduler has been stopped (this line might not be reached with BlockingScheduler and Ctrl+C).")
    if scheduler.running:
        scheduler.shutdown()
    print("MAIN_WORKER: Scheduler explicitly shut down.")