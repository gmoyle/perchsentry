"""Config flow for PerchSentry."""
from __future__ import annotations

from typing import Any

import aiohttp
import async_timeout
import voluptuous as vol

from homeassistant.config_entries import ConfigFlow
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from .const import CONF_HOST, CONF_PORT, DEFAULT_HOST, DEFAULT_PORT, DOMAIN


class PerchSentryConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for PerchSentry."""

    VERSION = 1

    async def async_step_user(self, user_input: dict[str, Any] | None = None):
        """Handle the initial step — ask for host/port and verify reachability."""
        errors: dict[str, str] = {}
        if user_input is not None:
            host = user_input[CONF_HOST]
            port = user_input[CONF_PORT]
            session = async_get_clientsession(self.hass)
            try:
                async with async_timeout.timeout(10):
                    async with session.get(f"http://{host}:{port}/api/ha") as resp:
                        resp.raise_for_status()
                        await resp.json()
            except (aiohttp.ClientError, TimeoutError):
                errors["base"] = "cannot_connect"
            else:
                await self.async_set_unique_id(f"{host}:{port}")
                self._abort_if_unique_id_configured()
                return self.async_create_entry(
                    title=f"PerchSentry ({host})", data=user_input
                )

        schema = vol.Schema(
            {
                vol.Required(CONF_HOST, default=DEFAULT_HOST): str,
                vol.Required(CONF_PORT, default=DEFAULT_PORT): int,
            }
        )
        return self.async_show_form(
            step_id="user", data_schema=schema, errors=errors
        )
