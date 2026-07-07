"""Base entity for PerchSentry — shared device info + coordinator wiring."""
from __future__ import annotations

from homeassistant.helpers.device_info import DeviceInfo
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import DOMAIN
from .coordinator import PerchSentryCoordinator


class PerchSentryEntity(CoordinatorEntity[PerchSentryCoordinator]):
    """Common base: groups every entity under one PerchSentry device."""

    _attr_has_entity_name = True

    def __init__(self, coordinator: PerchSentryCoordinator, entry_id: str) -> None:
        super().__init__(coordinator)
        self._entry_id = entry_id
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry_id)},
            name="PerchSentry",
            manufacturer="PerchSentry",
            model="Raspberry Pi bird camera",
            configuration_url=coordinator.base_url,
        )
