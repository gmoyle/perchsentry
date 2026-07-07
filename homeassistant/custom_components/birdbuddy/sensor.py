"""BirdBuddy sensors."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from homeassistant.components.sensor import (
    SensorDeviceClass,
    SensorEntity,
    SensorEntityDescription,
    SensorStateClass,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import PERCENTAGE, UnitOfTemperature
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import DOMAIN
from .entity import BirdBuddyEntity


@dataclass(frozen=True, kw_only=True)
class BirdBuddySensorDescription(SensorEntityDescription):
    """A sensor description with a value extractor over the /api/ha payload."""

    value_fn: Callable[[dict], object]


def _confidence_pct(data: dict):
    conf = data.get("last_confidence")
    return round(conf * 100, 1) if conf is not None else None


SENSORS: tuple[BirdBuddySensorDescription, ...] = (
    BirdBuddySensorDescription(
        key="last_species",
        name="Last bird",
        icon="mdi:bird",
        value_fn=lambda d: d.get("last_species"),
    ),
    BirdBuddySensorDescription(
        key="last_animal",
        name="Last animal",
        icon="mdi:paw",
        value_fn=lambda d: d.get("last_animal"),
    ),
    BirdBuddySensorDescription(
        key="last_confidence",
        name="Last confidence",
        native_unit_of_measurement=PERCENTAGE,
        state_class=SensorStateClass.MEASUREMENT,
        icon="mdi:target",
        value_fn=_confidence_pct,
    ),
    BirdBuddySensorDescription(
        key="today_sightings",
        name="Today's sightings",
        state_class=SensorStateClass.TOTAL,
        icon="mdi:counter",
        value_fn=lambda d: d.get("today_sightings"),
    ),
    BirdBuddySensorDescription(
        key="animals_today",
        name="Animals today",
        state_class=SensorStateClass.TOTAL,
        icon="mdi:paw",
        value_fn=lambda d: d.get("animals_today"),
    ),
    BirdBuddySensorDescription(
        key="cpu_temp_c",
        name="CPU temperature",
        device_class=SensorDeviceClass.TEMPERATURE,
        native_unit_of_measurement=UnitOfTemperature.CELSIUS,
        state_class=SensorStateClass.MEASUREMENT,
        value_fn=lambda d: d.get("cpu_temp_c"),
    ),
)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    coordinator = hass.data[DOMAIN][entry.entry_id]
    async_add_entities(
        BirdBuddySensor(coordinator, entry.entry_id, desc) for desc in SENSORS
    )


class BirdBuddySensor(BirdBuddyEntity, SensorEntity):
    """A single value pulled from the /api/ha payload."""

    entity_description: BirdBuddySensorDescription

    def __init__(self, coordinator, entry_id, description) -> None:
        super().__init__(coordinator, entry_id)
        self.entity_description = description
        self._attr_unique_id = f"{entry_id}_{description.key}"

    @property
    def native_value(self):
        return self.entity_description.value_fn(self.coordinator.data or {})
