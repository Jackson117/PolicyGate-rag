"""http_pool: the owned client keeps the default limits, and the pool can be read."""

import httpx

from policygate_rag.http_pool import LIMITS, make_client, pool_state


def test_limits_match_httpx_defaults_so_behaviour_does_not_change():
    default = httpx.Client()._transport._pool
    ours = make_client(180)._transport._pool
    assert ours._max_connections == default._max_connections == LIMITS.max_connections
    assert ours._max_keepalive_connections == default._max_keepalive_connections


def test_pool_state_is_readable_and_empty_at_start():
    state = pool_state(make_client(180))
    assert state == {"connections": 0, "idle": 0, "available": 0, "closed": 0,
                     "waiting_requests": 0, "assigned_requests": 0}


def test_unreadable_internals_lose_the_diagnostic_not_the_run():
    class Broken:
        _transport = None

    assert "error" in pool_state(Broken())
