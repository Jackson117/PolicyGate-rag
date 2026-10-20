"""http_pool: the owned client keeps the default limits, and the pool can be read."""


from policygate_rag.http_pool import LIMITS, make_client, pool_state


def test_limits_match_the_client_tau2_installs_so_behaviour_does_not_change():
    import litellm
    import tau2.utils.llm_utils  # noqa: F401 -- installs tau2's client

    theirs = litellm.client_session._transport._pool
    ours = make_client(180)._transport._pool
    assert ours._max_connections == theirs._max_connections == LIMITS.max_connections == 10
    assert ours._max_keepalive_connections == theirs._max_keepalive_connections == 5


def test_the_runner_installs_its_client_after_tau2s_and_it_stays():
    import litellm

    from policygate_rag.runner import install_http_client

    ours = install_http_client()
    import policygate_rag.agent  # noqa: F401 -- imports tau2's llm_utils again (cached)
    assert litellm.client_session is ours
    assert litellm.request_timeout == 180


def test_pool_state_is_readable_and_empty_at_start():
    state = pool_state(make_client(180))
    assert state == {"connections": 0, "idle": 0, "available": 0, "closed": 0,
                     "waiting_requests": 0, "assigned_requests": 0}


def test_unreadable_internals_lose_the_diagnostic_not_the_run():
    class Broken:
        _transport = None

    assert "error" in pool_state(Broken())
