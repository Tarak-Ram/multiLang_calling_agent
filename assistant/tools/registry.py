"""Default tool registry used by the worker."""
from __future__ import annotations

import httpx

from assistant.channels.base import ChannelAdapter
from assistant.tools.base import ToolRegistry
from assistant.tools.booking import (
    BookAppointment,
    CancelBooking,
    ListDates,
    ListMyBookings,
    ListServices,
    ListSlots,
    RescheduleBooking,
)
from assistant.tools.integrations import (
    CreateReturn,
    CrmUpdate,
    IntegrationClient,
    LookupOrder,
    ReportStatus,
    SendTemplate,
)


def build_registry(
    http: httpx.AsyncClient, channel: ChannelAdapter, *, allow_private_integrations: bool = False
) -> ToolRegistry:
    """Every tool the platform ships. A tenant can only USE those whose `group` is listed in
    its config `tools_enabled`; the registry enforces that on each call."""
    reg = ToolRegistry()
    client = IntegrationClient(http, allow_private=allow_private_integrations)
    for tool in (ListServices(), ListDates(), ListSlots(), ListMyBookings(), BookAppointment(),
                 CancelBooking(), RescheduleBooking(), LookupOrder(client), ReportStatus(client),
                 CreateReturn(client), CrmUpdate(client), SendTemplate(channel)):
        reg.register(tool)
    return reg
