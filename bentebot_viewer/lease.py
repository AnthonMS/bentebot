from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Lease:
    lease_id: str
    guild_id: int
    channel_id: int
    owner_id: int
    bot_user_id: int
    viewer_user_id: int
    frame_token: str = field(repr=False)
    expires_at: float = field(compare=False)

    @classmethod
    def parse(cls, payload: dict, *, viewer_id: int, now: float) -> Lease | None:
        if not isinstance(payload, dict):
            raise ValueError("Invalid viewer lease")
        if payload.get("active") is not True:
            return None
        try:
            ids = [
                int(payload[key])
                for key in ("guild_id", "channel_id", "owner_id", "bot_user_id", "viewer_user_id")
            ]
            ttl = float(payload["lease_seconds"])
            lease_id, token = payload["lease_id"], payload["frame_token"]
            if (
                not all(0 < value < 2**64 for value in ids)
                or ids[-1] != viewer_id
                or len(set(ids[2:])) != 3
                or not 0 < ttl <= 3
                or not isinstance(lease_id, str)
                or not 16 <= len(lease_id) <= 64
                or not isinstance(token, str)
                or not 32 <= len(token) <= 128
                or not token.isascii()
            ):
                raise ValueError
            return cls(lease_id, *ids, token, now + ttl)
        except (ValueError, KeyError, TypeError, OverflowError):
            raise ValueError("Invalid viewer lease") from None


class LeaseState:
    def __init__(self, clock):
        self.clock, self.current = clock, None

    def permits(self, lease: Lease) -> bool:
        return self.current is not None and self.current == lease and self.current.expires_at > self.clock()
