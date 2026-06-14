"""Tiered update scheduler.

Manages different update frequencies for different data sources:
- Tier 1 (fast): every 15 min -- price, TA
- Tier 2 (medium): every 1h -- sentiment, funding rates, premiums
- Tier 3 (slow): every 6h -- on-chain, whale, miner
- Tier 4 (daily): every 24h -- macro, cycle, dominance
"""

import asyncio
from datetime import datetime, timezone

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from src.engine.predictor import PredictionEngine
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class TieredScheduler:
    """Manages tiered data collection and prediction scheduling."""

    def __init__(self, engine: PredictionEngine, config: dict | None = None):
        self.engine = engine
        self.config = config or {}
        self.scheduler = AsyncIOScheduler()
        self._running = False

    def setup(self) -> None:
        """Configure all scheduled jobs."""
        intervals = self.config.get("schedule", {})

        # Tier 1: Fast updates and prediction runs
        tier1_min = intervals.get("tier1_fast", 15)
        self.scheduler.add_job(
            self._tier1_update,
            IntervalTrigger(minutes=tier1_min),
            id="tier1",
            name="Price + TA + Predict",
        )

        # Tier 2: Medium frequency
        tier2_min = intervals.get("tier2_medium", 60)
        self.scheduler.add_job(
            self._tier2_update,
            IntervalTrigger(minutes=tier2_min),
            id="tier2",
            name="Sentiment + Derivatives + Institutional",
        )

        # Tier 3: Slow updates
        tier3_min = intervals.get("tier3_slow", 360)
        self.scheduler.add_job(
            self._tier3_update,
            IntervalTrigger(minutes=tier3_min),
            id="tier3",
            name="On-chain + Whale",
        )

        # Tier 4: Daily
        tier4_min = intervals.get("tier4_daily", 1440)
        self.scheduler.add_job(
            self._tier4_update,
            IntervalTrigger(minutes=tier4_min),
            id="tier4",
            name="Macro + Cycle",
        )

        logger.info(f"Scheduler configured: T1={tier1_min}m, T2={tier2_min}m, T3={tier3_min}m, T4={tier4_min}m")

    async def _tier1_update(self) -> None:
        """Fast update: price + TA + run prediction."""
        try:
            await self.engine.run_prediction()
        except Exception as e:
            logger.error(f"Tier 1 update failed: {e}")

    async def _tier2_update(self) -> None:
        """Medium update: sentiment, derivatives, institutional."""
        try:
            await asyncio.gather(
                self.engine.sentiment.collect(),
                self.engine.derivatives.collect(),
                self.engine.institutional.collect(),
                return_exceptions=True,
            )
            logger.info("Tier 2 signals updated")
        except Exception as e:
            logger.error(f"Tier 2 update failed: {e}")

    async def _tier3_update(self) -> None:
        """Slow update: on-chain, whale."""
        try:
            await asyncio.gather(
                self.engine.onchain.collect(),
                self.engine.whale.collect(),
                return_exceptions=True,
            )
            logger.info("Tier 3 signals updated")
        except Exception as e:
            logger.error(f"Tier 3 update failed: {e}")

    async def _tier4_update(self) -> None:
        """Daily update: macro, cycle."""
        try:
            await asyncio.gather(
                self.engine.macro.collect(),
                self.engine.cycle.collect(),
                return_exceptions=True,
            )
            logger.info("Tier 4 signals updated")
        except Exception as e:
            logger.error(f"Tier 4 update failed: {e}")

    async def start(self) -> None:
        """Start the scheduler and run initial collection."""
        if self._running:
            return

        self.setup()
        self.scheduler.start()
        self._running = True
        logger.info("Tiered scheduler started")

        # Run all tiers immediately on start
        await asyncio.gather(
            self._tier2_update(),
            self._tier3_update(),
            self._tier4_update(),
            return_exceptions=True,
        )
        # Then run first prediction
        await self._tier1_update()

    def stop(self) -> None:
        """Stop the scheduler."""
        if self._running:
            self.scheduler.shutdown(wait=False)
            self._running = False
            logger.info("Scheduler stopped")

    @property
    def is_running(self) -> bool:
        return self._running

    def get_status(self) -> dict:
        """Get current scheduler status."""
        jobs = []
        for job in self.scheduler.get_jobs():
            jobs.append({
                "id": job.id,
                "name": job.name,
                "next_run": str(job.next_run_time) if job.next_run_time else None,
            })
        return {"running": self._running, "jobs": jobs}
