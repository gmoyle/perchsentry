"""Data update coordinator for BirdBuddy."""
from __future__ import annotations

import logging
from datetime import timedelta

import aiohttp
import async_timeout

from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.update_coordinator import (
    DataUpdateCoordinator,
    UpdateFailed,
)

from .const import (
    DEFAULT_SCAN_INTERVAL,
    DOMAIN,
    EVENT_ANIMAL_DETECTED,
    EVENT_BIRD_DETECTED,
)

_LOGGER = logging.getLogger(__name__)


class BirdBuddyCoordinator(DataUpdateCoordinator):
    """Polls BirdBuddy's /api/ha and fires HA events on fresh detections."""

    def __init__(self, hass: HomeAssistant, host: str, port: int) -> None:
        super().__init__(
            hass,
            _LOGGER,
            name=DOMAIN,
            update_interval=timedelta(seconds=DEFAULT_SCAN_INTERVAL),
        )
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self._session = async_get_clientsession(hass)
        self._last_detection_at = None

    async def _async_update_data(self) -> dict:
        try:
            async with async_timeout.timeout(10):
                async with self._session.get(f"{self.base_url}/api/ha") as resp:
                    resp.raise_for_status()
                    data = await resp.json()
        except (aiohttp.ClientError, TimeoutError) as err:
            raise UpdateFailed(f"BirdBuddy unreachable: {err}") from err

        self._maybe_fire_event(data)
        return data

    def _maybe_fire_event(self, data: dict) -> None:
        """Fire a bus event when last_event_at advances to a new detection."""
        at = data.get("last_event_at")
        event = data.get("last_event")
        if not at or at == self._last_detection_at:
            return
        # Skip the first poll after setup so we don't replay a stale detection.
        if self._last_detection_at is not None and event in (
            "bird_detected",
            "animal_detected",
        ):
            image_url = data.get("last_image_url")
            payload = {
                "kind": "bird" if event == "bird_detected" else "animal",
                "species": data.get("last_species") or data.get("last_animal"),
                "confidence": data.get("last_confidence"),
                "image_url": (
                    f"{self.base_url}{image_url}" if image_url else None
                ),
            }
            self.hass.bus.async_fire(
                EVENT_BIRD_DETECTED
                if event == "bird_detected"
                else EVENT_ANIMAL_DETECTED,
                payload,
            )
        self._last_detection_at = at
