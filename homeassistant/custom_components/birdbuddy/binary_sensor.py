"""BirdBuddy binary sensors — 'recently detected' flags for dashboards."""
from __future__ import annotations

import time

from homeassistant.components.binary_sensor import (
    BinarySensorDeviceClass,
    BinarySensorEntity,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import DETECTION_HOLD, DOMAIN
from .entity import BirdBuddyEntity


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    coordinator = hass.data[DOMAIN][entry.entry_id]
    async_add_entities(
        [
            BirdBuddyDetection(
                coordinator, entry.entry_id, "bird_detected", "Bird detected", "mdi:bird"
            ),
            BirdBuddyDetection(
                coordinator, entry.entry_id, "animal_detected", "Animal detected", "mdi:paw"
            ),
        ]
    )


class BirdBuddyDetection(BirdBuddyEntity, BinarySensorEntity):
    """On for DETECTION_HOLD seconds after a matching detection."""

    _attr_device_class = BinarySensorDeviceClass.MOTION

    def __init__(self, coordinator, entry_id, event_key, name, icon) -> None:
        super().__init__(coordinator, entry_id)
        self._event_key = event_key
        self._attr_name = name
        self._attr_icon = icon
        self._attr_unique_id = f"{entry_id}_{event_key}"

    @property
    def is_on(self) -> bool:
        data = self.coordinator.data or {}
        if data.get("last_event") != self._event_key:
            return False
        at = data.get("last_event_at") or 0
        return (time.time() - at) < DETECTION_HOLD

    @property
    def extra_state_attributes(self):
        data = self.coordinator.data or {}
        if data.get("last_event") != self._event_key:
            return None
        image_url = data.get("last_image_url")
        return {
            "species": data.get("last_species") or data.get("last_animal"),
            "confidence": data.get("last_confidence"),
            "image_url": (
                f"{self.coordinator.base_url}{image_url}" if image_url else None
            ),
        }
