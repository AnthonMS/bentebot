from __future__ import annotations

from collections.abc import Awaitable, Callable

from .domain import Audience, Source


class AccessPolicy:
    """Every source is checked against every recipient. Missing information denies access."""

    def __init__(self, can_read: Callable[[int, Source], Awaitable[bool]]):
        self.can_read = can_read

    async def allows(self, source: Source, audience: Audience) -> bool:
        if not audience.user_ids:
            return False
        if source.kind == "dm":
            return audience.user_ids == frozenset({source.owner_id}) and audience.destination_id is None
        if source.kind == "screen" and not audience.user_ids.issubset(source.audience_ids):
            return False
        if source.guild_id != audience.guild_id:
            return False
        for user_id in audience.user_ids:
            if not await self.can_read(user_id, source):
                return False
        return True

    async def allows_all(self, sources: tuple[Source, ...], audience: Audience) -> bool:
        for source in sources:
            if not await self.allows(source, audience):
                return False
        return True
