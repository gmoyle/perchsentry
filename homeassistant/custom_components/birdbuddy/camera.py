"""BirdBuddy camera — live MJPEG stream plus snapshot stills."""
from __future__ import annotations

from homeassistant.components.camera import Camera
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import (
    async_aiohttp_proxy_web,
    async_get_clientsession,
)
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import DOMAIN
from .entity import BirdBuddyEntity


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    coordinator = hass.data[DOMAIN][entry.entry_id]
    async_add_entities([BirdBuddyCamera(coordinator, entry.entry_id)])


class BirdBuddyCamera(BirdBuddyEntity, Camera):
    """A camera entity backed by BirdBuddy's MJPEG stream and /snapshot."""

    _attr_name = "Camera"

    def __init__(self, coordinator, entry_id: str) -> None:
        BirdBuddyEntity.__init__(self, coordinator, entry_id)
        Camera.__init__(self)
        self._attr_unique_id = f"{entry_id}_camera"
        self._mjpeg_url = f"{coordinator.base_url}/stream"
        self._snapshot_url = f"{coordinator.base_url}/snapshot"

    async def async_camera_image(
        self, width: int | None = None, height: int | None = None
    ) -> bytes | None:
        session = async_get_clientsession(self.hass)
        async with session.get(self._snapshot_url) as resp:
            resp.raise_for_status()
            return await resp.read()

    async def handle_async_mjpeg_stream(self, request):
        session = async_get_clientsession(self.hass)
        stream_coro = session.get(self._mjpeg_url)
        return await async_aiohttp_proxy_web(self.hass, request, stream_coro)
