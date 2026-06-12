"""Tests for the wire-method maps and two-step parse functions in `mcp.types.methods`.

The maps are data: these tests pin every row (methods per version, response
row values, class provenance) against expected sets transcribed from the
per-version ClientRequest/ServerRequest/ClientNotification/ServerNotification
union memberships in the pinned schemas, and exercise every behavior branch
of the six parse functions.
"""

import importlib.util
from collections.abc import Mapping
from types import MappingProxyType, UnionType
from typing import Any, get_args

import pydantic
import pytest

import mcp.types as types
import mcp.types.v2025_11_25 as v2025
import mcp.types.v2026_07_28 as v2026
from mcp.shared.version import KNOWN_PROTOCOL_VERSIONS
from mcp.types import methods
from mcp.types._wire_base import WireModel

# Expected method sets per protocol version, transcribed from each pinned
# schema's ClientRequest/ServerRequest/ClientNotification/ServerNotification
# union memberships minus the tasks family (the four tasks/* request methods and
# notifications/tasks/status): the SDK defines the task types but never
# dispatches them; extensions register them. The version axis is derived from
# KNOWN_PROTOCOL_VERSIONS so that adding a version without map rows fails here
# instead of silently gating every method at runtime.
EXPECTED_METHODS: dict[str, dict[str, frozenset[str]]] = {
    "2024-11-05": {
        "CLIENT_REQUESTS": frozenset(
            {
                "completion/complete",
                "initialize",
                "logging/setLevel",
                "ping",
                "prompts/get",
                "prompts/list",
                "resources/list",
                "resources/read",
                "resources/subscribe",
                "resources/templates/list",
                "resources/unsubscribe",
                "tools/call",
                "tools/list",
            }
        ),
        "CLIENT_NOTIFICATIONS": frozenset(
            {
                "notifications/cancelled",
                "notifications/initialized",
                "notifications/progress",
                "notifications/roots/list_changed",
            }
        ),
        "SERVER_REQUESTS": frozenset({"ping", "roots/list", "sampling/createMessage"}),
        "SERVER_NOTIFICATIONS": frozenset(
            {
                "notifications/cancelled",
                "notifications/message",
                "notifications/progress",
                "notifications/prompts/list_changed",
                "notifications/resources/list_changed",
                "notifications/resources/updated",
                "notifications/tools/list_changed",
            }
        ),
    },
    "2025-03-26": {
        "CLIENT_REQUESTS": frozenset(
            {
                "completion/complete",
                "initialize",
                "logging/setLevel",
                "ping",
                "prompts/get",
                "prompts/list",
                "resources/list",
                "resources/read",
                "resources/subscribe",
                "resources/templates/list",
                "resources/unsubscribe",
                "tools/call",
                "tools/list",
            }
        ),
        "CLIENT_NOTIFICATIONS": frozenset(
            {
                "notifications/cancelled",
                "notifications/initialized",
                "notifications/progress",
                "notifications/roots/list_changed",
            }
        ),
        "SERVER_REQUESTS": frozenset({"ping", "roots/list", "sampling/createMessage"}),
        "SERVER_NOTIFICATIONS": frozenset(
            {
                "notifications/cancelled",
                "notifications/message",
                "notifications/progress",
                "notifications/prompts/list_changed",
                "notifications/resources/list_changed",
                "notifications/resources/updated",
                "notifications/tools/list_changed",
            }
        ),
    },
    "2025-06-18": {
        "CLIENT_REQUESTS": frozenset(
            {
                "completion/complete",
                "initialize",
                "logging/setLevel",
                "ping",
                "prompts/get",
                "prompts/list",
                "resources/list",
                "resources/read",
                "resources/subscribe",
                "resources/templates/list",
                "resources/unsubscribe",
                "tools/call",
                "tools/list",
            }
        ),
        "CLIENT_NOTIFICATIONS": frozenset(
            {
                "notifications/cancelled",
                "notifications/initialized",
                "notifications/progress",
                "notifications/roots/list_changed",
            }
        ),
        "SERVER_REQUESTS": frozenset({"elicitation/create", "ping", "roots/list", "sampling/createMessage"}),
        "SERVER_NOTIFICATIONS": frozenset(
            {
                "notifications/cancelled",
                "notifications/message",
                "notifications/progress",
                "notifications/prompts/list_changed",
                "notifications/resources/list_changed",
                "notifications/resources/updated",
                "notifications/tools/list_changed",
            }
        ),
    },
    "2025-11-25": {
        "CLIENT_REQUESTS": frozenset(
            {
                "completion/complete",
                "initialize",
                "logging/setLevel",
                "ping",
                "prompts/get",
                "prompts/list",
                "resources/list",
                "resources/read",
                "resources/subscribe",
                "resources/templates/list",
                "resources/unsubscribe",
                "tools/call",
                "tools/list",
            }
        ),
        "CLIENT_NOTIFICATIONS": frozenset(
            {
                "notifications/cancelled",
                "notifications/initialized",
                "notifications/progress",
                "notifications/roots/list_changed",
            }
        ),
        "SERVER_REQUESTS": frozenset({"elicitation/create", "ping", "roots/list", "sampling/createMessage"}),
        "SERVER_NOTIFICATIONS": frozenset(
            {
                "notifications/cancelled",
                "notifications/elicitation/complete",
                "notifications/message",
                "notifications/progress",
                "notifications/prompts/list_changed",
                "notifications/resources/list_changed",
                "notifications/resources/updated",
                "notifications/tools/list_changed",
            }
        ),
    },
    "2026-07-28": {
        "CLIENT_REQUESTS": frozenset(
            {
                "completion/complete",
                "prompts/get",
                "prompts/list",
                "resources/list",
                "resources/read",
                "resources/templates/list",
                "server/discover",
                "subscriptions/listen",
                "tools/call",
                "tools/list",
            }
        ),
        "CLIENT_NOTIFICATIONS": frozenset({"notifications/cancelled", "notifications/progress"}),
        # The standalone server-to-client request channel does not exist at
        # this version; the empty set is deliberate, not an omission.
        "SERVER_REQUESTS": frozenset(),
        "SERVER_NOTIFICATIONS": frozenset(
            {
                "notifications/cancelled",
                "notifications/elicitation/complete",
                "notifications/message",
                "notifications/progress",
                "notifications/prompts/list_changed",
                "notifications/resources/list_changed",
                "notifications/resources/updated",
                "notifications/subscriptions/acknowledged",
                "notifications/tools/list_changed",
            }
        ),
    },
}

# Response row values pinned per (method, version): identity for class-valued
# rows, exact arm tuple (classes and order) for union rows. Without the value
# pin, mistyping a response row as the bare surface EmptyResult (all fields
# optional, extras ignored) would silently disable strict response validation
# for that row while every key-level test stays green.
EXPECTED_SERVER_RESULTS: dict[tuple[str, str], type[WireModel] | tuple[type[WireModel], ...]] = {
    # 2024-11-05
    ("completion/complete", "2024-11-05"): v2025.CompleteResult,
    ("initialize", "2024-11-05"): v2025.InitializeResult,
    ("logging/setLevel", "2024-11-05"): v2025.EmptyResult,
    ("ping", "2024-11-05"): v2025.EmptyResult,
    ("prompts/get", "2024-11-05"): v2025.GetPromptResult,
    ("prompts/list", "2024-11-05"): v2025.ListPromptsResult,
    ("resources/list", "2024-11-05"): v2025.ListResourcesResult,
    ("resources/read", "2024-11-05"): v2025.ReadResourceResult,
    ("resources/subscribe", "2024-11-05"): v2025.EmptyResult,
    ("resources/templates/list", "2024-11-05"): v2025.ListResourceTemplatesResult,
    ("resources/unsubscribe", "2024-11-05"): v2025.EmptyResult,
    ("tools/call", "2024-11-05"): v2025.CallToolResult,
    ("tools/list", "2024-11-05"): v2025.ListToolsResult,
    # 2025-03-26
    ("completion/complete", "2025-03-26"): v2025.CompleteResult,
    ("initialize", "2025-03-26"): v2025.InitializeResult,
    ("logging/setLevel", "2025-03-26"): v2025.EmptyResult,
    ("ping", "2025-03-26"): v2025.EmptyResult,
    ("prompts/get", "2025-03-26"): v2025.GetPromptResult,
    ("prompts/list", "2025-03-26"): v2025.ListPromptsResult,
    ("resources/list", "2025-03-26"): v2025.ListResourcesResult,
    ("resources/read", "2025-03-26"): v2025.ReadResourceResult,
    ("resources/subscribe", "2025-03-26"): v2025.EmptyResult,
    ("resources/templates/list", "2025-03-26"): v2025.ListResourceTemplatesResult,
    ("resources/unsubscribe", "2025-03-26"): v2025.EmptyResult,
    ("tools/call", "2025-03-26"): v2025.CallToolResult,
    ("tools/list", "2025-03-26"): v2025.ListToolsResult,
    # 2025-06-18
    ("completion/complete", "2025-06-18"): v2025.CompleteResult,
    ("initialize", "2025-06-18"): v2025.InitializeResult,
    ("logging/setLevel", "2025-06-18"): v2025.EmptyResult,
    ("ping", "2025-06-18"): v2025.EmptyResult,
    ("prompts/get", "2025-06-18"): v2025.GetPromptResult,
    ("prompts/list", "2025-06-18"): v2025.ListPromptsResult,
    ("resources/list", "2025-06-18"): v2025.ListResourcesResult,
    ("resources/read", "2025-06-18"): v2025.ReadResourceResult,
    ("resources/subscribe", "2025-06-18"): v2025.EmptyResult,
    ("resources/templates/list", "2025-06-18"): v2025.ListResourceTemplatesResult,
    ("resources/unsubscribe", "2025-06-18"): v2025.EmptyResult,
    ("tools/call", "2025-06-18"): v2025.CallToolResult,
    ("tools/list", "2025-06-18"): v2025.ListToolsResult,
    # 2025-11-25
    ("completion/complete", "2025-11-25"): v2025.CompleteResult,
    ("initialize", "2025-11-25"): v2025.InitializeResult,
    ("logging/setLevel", "2025-11-25"): v2025.EmptyResult,
    ("ping", "2025-11-25"): v2025.EmptyResult,
    ("prompts/get", "2025-11-25"): v2025.GetPromptResult,
    ("prompts/list", "2025-11-25"): v2025.ListPromptsResult,
    ("resources/list", "2025-11-25"): v2025.ListResourcesResult,
    ("resources/read", "2025-11-25"): v2025.ReadResourceResult,
    ("resources/subscribe", "2025-11-25"): v2025.EmptyResult,
    ("resources/templates/list", "2025-11-25"): v2025.ListResourceTemplatesResult,
    ("resources/unsubscribe", "2025-11-25"): v2025.EmptyResult,
    ("tools/call", "2025-11-25"): v2025.CallToolResult,
    ("tools/list", "2025-11-25"): v2025.ListToolsResult,
    # 2026-07-28
    ("completion/complete", "2026-07-28"): v2026.CompleteResult,
    ("prompts/get", "2026-07-28"): (v2026.GetPromptResult, v2026.InputRequiredResult),
    ("prompts/list", "2026-07-28"): v2026.ListPromptsResult,
    ("resources/list", "2026-07-28"): v2026.ListResourcesResult,
    ("resources/read", "2026-07-28"): (v2026.ReadResourceResult, v2026.InputRequiredResult),
    ("resources/templates/list", "2026-07-28"): v2026.ListResourceTemplatesResult,
    ("server/discover", "2026-07-28"): v2026.DiscoverResult,
    ("subscriptions/listen", "2026-07-28"): v2026.EmptyResult,
    ("tools/call", "2026-07-28"): (v2026.CallToolResult, v2026.InputRequiredResult),
    ("tools/list", "2026-07-28"): v2026.ListToolsResult,
}

EXPECTED_CLIENT_RESULTS: dict[tuple[str, str], type[WireModel] | tuple[type[WireModel], ...]] = {
    # 2024-11-05
    ("ping", "2024-11-05"): v2025.EmptyResult,
    ("roots/list", "2024-11-05"): v2025.ListRootsResult,
    ("sampling/createMessage", "2024-11-05"): v2025.CreateMessageResult,
    # 2025-03-26
    ("ping", "2025-03-26"): v2025.EmptyResult,
    ("roots/list", "2025-03-26"): v2025.ListRootsResult,
    ("sampling/createMessage", "2025-03-26"): v2025.CreateMessageResult,
    # 2025-06-18
    ("elicitation/create", "2025-06-18"): v2025.ElicitResult,
    ("ping", "2025-06-18"): v2025.EmptyResult,
    ("roots/list", "2025-06-18"): v2025.ListRootsResult,
    ("sampling/createMessage", "2025-06-18"): v2025.CreateMessageResult,
    # 2025-11-25
    ("elicitation/create", "2025-11-25"): v2025.ElicitResult,
    ("ping", "2025-11-25"): v2025.EmptyResult,
    ("roots/list", "2025-11-25"): v2025.ListRootsResult,
    ("sampling/createMessage", "2025-11-25"): v2025.CreateMessageResult,
}

# The only response rows legitimately valued by the bare surface EmptyResult.
EMPTY_SERVER_RESPONSE_METHODS = frozenset(
    {"logging/setLevel", "ping", "resources/subscribe", "resources/unsubscribe", "subscriptions/listen"}
)
EMPTY_CLIENT_RESPONSE_METHODS = frozenset({"ping"})

# Which surface package each version's rows must come from. The pre-2026
# versions share the 2025-11-25 package (the SDK's only earlier surface;
# the method gate stays version-exact through key absence even though shape
# validation for those versions is 2025-11-25-shaped).
PACKAGE_BY_VERSION = {
    "2024-11-05": "mcp.types.v2025_11_25",
    "2025-03-26": "mcp.types.v2025_11_25",
    "2025-06-18": "mcp.types.v2025_11_25",
    "2025-11-25": "mcp.types.v2025_11_25",
    "2026-07-28": "mcp.types.v2026_07_28",
}

# Requests on sessions negotiated at 2026-07-28 or later must carry the three
# reserved identity entries in params._meta; the 2026 surface types require them.
META_TRIPLE: dict[str, Any] = {
    "io.modelcontextprotocol/protocolVersion": "2026-07-28",
    "io.modelcontextprotocol/clientInfo": {"name": "client", "version": "1.0"},
    "io.modelcontextprotocol/clientCapabilities": {},
}

# One minimal valid params mapping per surface request class. Fixtures are
# keyed by the surface class, not by (method, version): rows sharing a value
# share a fixture, so a future version's diff here is proportional to actual
# shape changes. Root URIs use the file scheme because the monolith's Root.uri
# is file-scheme only (see the companion divergence test below).
REQUEST_PARAMS_FIXTURES: dict[type[WireModel], dict[str, Any] | None] = {
    v2025.CallToolRequest: {"name": "echo"},
    v2025.CompleteRequest: {"ref": {"type": "ref/prompt", "name": "p"}, "argument": {"name": "a", "value": "v"}},
    v2025.CreateMessageRequest: {
        "messages": [{"role": "user", "content": {"type": "text", "text": "hi"}}],
        "maxTokens": 100,
    },
    v2025.ElicitRequest: {"message": "m", "requestedSchema": {"type": "object", "properties": {}}},
    v2025.GetPromptRequest: {"name": "greeting"},
    v2025.InitializeRequest: {
        "protocolVersion": "2025-11-25",
        "capabilities": {},
        "clientInfo": {"name": "client", "version": "1.0"},
    },
    v2025.ListPromptsRequest: None,
    v2025.ListResourcesRequest: None,
    v2025.ListResourceTemplatesRequest: None,
    v2025.ListRootsRequest: None,
    v2025.ListToolsRequest: None,
    v2025.PingRequest: None,
    v2025.ReadResourceRequest: {"uri": "https://example.com/resource"},
    v2025.SetLevelRequest: {"level": "info"},
    v2025.SubscribeRequest: {"uri": "https://example.com/resource"},
    v2025.UnsubscribeRequest: {"uri": "https://example.com/resource"},
    v2026.CallToolRequest: {"_meta": META_TRIPLE, "name": "echo"},
    v2026.CompleteRequest: {
        "_meta": META_TRIPLE,
        "ref": {"type": "ref/prompt", "name": "p"},
        "argument": {"name": "a", "value": "v"},
    },
    v2026.DiscoverRequest: {"_meta": META_TRIPLE},
    v2026.GetPromptRequest: {"_meta": META_TRIPLE, "name": "greeting"},
    v2026.ListPromptsRequest: {"_meta": META_TRIPLE},
    v2026.ListResourcesRequest: {"_meta": META_TRIPLE},
    v2026.ListResourceTemplatesRequest: {"_meta": META_TRIPLE},
    v2026.ListToolsRequest: {"_meta": META_TRIPLE},
    v2026.ReadResourceRequest: {"_meta": META_TRIPLE, "uri": "https://example.com/resource"},
    v2026.SubscriptionsListenRequest: {"_meta": META_TRIPLE, "notifications": {}},
}

NOTIFICATION_PARAMS_FIXTURES: dict[type[WireModel], dict[str, Any] | None] = {
    v2025.CancelledNotification: {"requestId": 1},
    v2025.ElicitationCompleteNotification: {"elicitationId": "e1"},
    v2025.InitializedNotification: None,
    v2025.LoggingMessageNotification: {"level": "info", "data": "x"},
    v2025.ProgressNotification: {"progressToken": 1, "progress": 0.5},
    v2025.PromptListChangedNotification: None,
    v2025.ResourceListChangedNotification: None,
    v2025.ResourceUpdatedNotification: {"uri": "https://example.com/resource"},
    v2025.RootsListChangedNotification: None,
    v2025.ToolListChangedNotification: None,
    v2026.CancelledNotification: {"requestId": 1},
    v2026.ElicitationCompleteNotification: {"elicitationId": "e1"},
    v2026.LoggingMessageNotification: {"level": "info", "data": "x"},
    v2026.ProgressNotification: {"progressToken": 1, "progress": 0.5},
    v2026.PromptListChangedNotification: None,
    v2026.ResourceListChangedNotification: None,
    v2026.ResourceUpdatedNotification: {"uri": "https://example.com/resource"},
    v2026.SubscriptionsAcknowledgedNotification: {"notifications": {}},
    v2026.ToolListChangedNotification: None,
}

# One minimal valid result body per response row value (class or union alias).
# The 2026 bodies carry resultType and, on the cacheable results, ttlMs and
# cacheScope, because the 2026 surface requires all three on the wire.
RESULT_BODY_FIXTURES: dict[type[WireModel] | UnionType, dict[str, Any]] = {
    v2025.CallToolResult: {"content": []},
    v2025.CompleteResult: {"completion": {"values": []}},
    v2025.CreateMessageResult: {"role": "assistant", "content": {"type": "text", "text": "hi"}, "model": "m"},
    v2025.ElicitResult: {"action": "accept"},
    v2025.EmptyResult: {},
    v2025.GetPromptResult: {"messages": []},
    v2025.InitializeResult: {
        "protocolVersion": "2025-11-25",
        "capabilities": {},
        "serverInfo": {"name": "server", "version": "1.0"},
    },
    v2025.ListPromptsResult: {"prompts": []},
    v2025.ListResourcesResult: {"resources": []},
    v2025.ListResourceTemplatesResult: {"resourceTemplates": []},
    v2025.ListRootsResult: {"roots": [{"uri": "file:///workspace"}]},
    v2025.ListToolsResult: {"tools": []},
    v2025.ReadResourceResult: {"contents": []},
    v2026.AnyCallToolResult: {"content": [], "resultType": "complete"},
    v2026.AnyGetPromptResult: {"messages": [], "resultType": "complete"},
    v2026.AnyReadResourceResult: {"contents": [], "resultType": "complete", "ttlMs": 0, "cacheScope": "private"},
    v2026.CompleteResult: {"completion": {"values": []}, "resultType": "complete"},
    v2026.DiscoverResult: {
        "supportedVersions": ["2026-07-28"],
        "capabilities": {},
        "serverInfo": {"name": "server", "version": "1.0"},
        "resultType": "complete",
        "ttlMs": 0,
        "cacheScope": "private",
    },
    v2026.EmptyResult: {"resultType": "complete"},
    v2026.ListPromptsResult: {"prompts": [], "resultType": "complete", "ttlMs": 0, "cacheScope": "private"},
    v2026.ListResourcesResult: {"resources": [], "resultType": "complete", "ttlMs": 0, "cacheScope": "private"},
    v2026.ListResourceTemplatesResult: {
        "resourceTemplates": [],
        "resultType": "complete",
        "ttlMs": 0,
        "cacheScope": "private",
    },
    v2026.ListToolsResult: {"tools": [], "resultType": "complete", "ttlMs": 0, "cacheScope": "private"},
}


def test_maps_define_exactly_the_expected_methods_for_every_known_version():
    """Each known protocol version's map rows match the transcribed method sets exactly.

    Spec-mandated method matrix. The version axis is derived from
    KNOWN_PROTOCOL_VERSIONS: a version added there without an EXPECTED_METHODS
    entry (or without map rows) fails here, instead of answering
    METHOD_NOT_FOUND to every request on a successfully negotiated session.
    """
    assert set(EXPECTED_METHODS) == set(KNOWN_PROTOCOL_VERSIONS)
    surface_maps: dict[str, Mapping[tuple[str, str], object]] = {
        "CLIENT_REQUESTS": methods.CLIENT_REQUESTS,
        "CLIENT_NOTIFICATIONS": methods.CLIENT_NOTIFICATIONS,
        "SERVER_REQUESTS": methods.SERVER_REQUESTS,
        "SERVER_NOTIFICATIONS": methods.SERVER_NOTIFICATIONS,
    }
    for version in KNOWN_PROTOCOL_VERSIONS:
        for map_name, surface_map in surface_maps.items():
            derived = {method for (method, row_version) in surface_map if row_version == version}
            assert derived == EXPECTED_METHODS[version][map_name], f"{map_name} at {version}"


def test_response_map_keys_mirror_the_request_map_keys():
    """Every request that can be sent has exactly one response row at exactly the versions it exists."""
    assert set(methods.SERVER_RESULTS) == set(methods.CLIENT_REQUESTS)
    assert set(methods.CLIENT_RESULTS) == set(methods.SERVER_REQUESTS)


def test_response_row_values_match_the_pinned_classes_and_unions():
    """Response row values are pinned per (method, version): identity for classes, exact arm order for unions.

    SDK-defined data pin: response rows carry no method literal, so a mistyped
    row would otherwise silently weaken response validation. Also asserts that
    only the known empty-response methods are valued by the bare EmptyResult.
    """
    assert set(EXPECTED_SERVER_RESULTS) == set(methods.SERVER_RESULTS)
    assert set(EXPECTED_CLIENT_RESULTS) == set(methods.CLIENT_RESULTS)
    pinned = [
        (methods.SERVER_RESULTS, EXPECTED_SERVER_RESULTS, EMPTY_SERVER_RESPONSE_METHODS),
        (methods.CLIENT_RESULTS, EXPECTED_CLIENT_RESULTS, EMPTY_CLIENT_RESPONSE_METHODS),
    ]
    for response_map, expected_rows, empty_methods in pinned:
        for (method, version), expected in expected_rows.items():
            actual = response_map[(method, version)]
            if isinstance(expected, tuple):
                assert get_args(actual) == expected, f"{method} at {version}"
            else:
                assert actual is expected, f"{method} at {version}"
            if method not in empty_methods:
                assert actual is not v2025.EmptyResult, f"{method} at {version}"
                assert actual is not v2026.EmptyResult, f"{method} at {version}"


def test_surface_keys_agree_with_their_classes_and_the_monolith_maps():
    """Every surface key's method matches its class's method literal, its monolith row, and its version's package.

    SDK-defined cross-map invariants: (a) each surface method has a monolith
    row, (b) the surface class declares the key's method literal, (c) so does
    the monolith class, (d) each row's value comes from the package that
    serves its version.
    """
    request_maps: list[Mapping[tuple[str, str], type[WireModel]]] = [
        methods.CLIENT_REQUESTS,
        methods.SERVER_REQUESTS,
    ]
    notification_maps: list[Mapping[tuple[str, str], type[WireModel]]] = [
        methods.CLIENT_NOTIFICATIONS,
        methods.SERVER_NOTIFICATIONS,
    ]
    for surface_maps, monolith_map in (
        (request_maps, methods.MONOLITH_REQUESTS),
        (notification_maps, methods.MONOLITH_NOTIFICATIONS),
    ):
        for surface_map in surface_maps:
            for (method, version), surface_type in surface_map.items():
                assert method in monolith_map, f"{method} has no monolith row"
                assert get_args(surface_type.model_fields["method"].annotation) == (method,)
                assert get_args(monolith_map[method].model_fields["method"].annotation) == (method,)
                assert surface_type.__module__ == PACKAGE_BY_VERSION[version], f"{method} at {version}"
    for response_map in (methods.SERVER_RESULTS, methods.CLIENT_RESULTS):
        for (method, version), row in response_map.items():
            assert method in methods.MONOLITH_RESULTS, f"{method} has no monolith row"
            for arm in get_args(row) or (row,):
                assert arm.__module__ == PACKAGE_BY_VERSION[version], f"{method} at {version}"


def _assign_item(mapping: Any) -> None:
    """Attempt the item assignment the map types already forbid statically; the runtime must forbid it too."""
    mapping["new-key"] = None


def test_built_in_maps_are_immutable():
    """All nine built-in maps are mapping proxies that raise TypeError on mutation; extension is dict union."""
    map_names = [
        "CLIENT_NOTIFICATIONS",
        "CLIENT_REQUESTS",
        "CLIENT_RESULTS",
        "MONOLITH_NOTIFICATIONS",
        "MONOLITH_REQUESTS",
        "MONOLITH_RESULTS",
        "SERVER_NOTIFICATIONS",
        "SERVER_REQUESTS",
        "SERVER_RESULTS",
    ]
    for map_name in map_names:
        built_in = getattr(methods, map_name)
        assert isinstance(built_in, MappingProxyType), map_name
        with pytest.raises(TypeError):
            _assign_item(built_in)


def test_minimal_request_bodies_parse_through_every_request_row():
    """A minimal valid body per surface class parses into a monolith request through every request row.

    SDK-defined property, scoped to these fixtures: what the surface step
    admits, the monolith step parses (the one audited divergence is pinned in
    the non-file Root.uri test below).
    """
    for (method, version), surface_type in methods.CLIENT_REQUESTS.items():
        parsed = methods.parse_client_request(method, version, REQUEST_PARAMS_FIXTURES[surface_type])
        assert isinstance(parsed, types.Request), f"{method} at {version}"
    for (method, version), surface_type in methods.SERVER_REQUESTS.items():
        parsed = methods.parse_server_request(method, version, REQUEST_PARAMS_FIXTURES[surface_type])
        assert isinstance(parsed, types.Request), f"{method} at {version}"


def test_minimal_notification_bodies_parse_through_every_notification_row():
    """A minimal valid body per surface class parses into a monolith notification through every notification row."""
    for (method, version), surface_type in methods.CLIENT_NOTIFICATIONS.items():
        parsed = methods.parse_client_notification(method, version, NOTIFICATION_PARAMS_FIXTURES[surface_type])
        assert isinstance(parsed, types.Notification), f"{method} at {version}"
    for (method, version), surface_type in methods.SERVER_NOTIFICATIONS.items():
        parsed = methods.parse_server_notification(method, version, NOTIFICATION_PARAMS_FIXTURES[surface_type])
        assert isinstance(parsed, types.Notification), f"{method} at {version}"


def test_minimal_result_bodies_parse_through_every_result_row():
    """A minimal valid body per response row value parses into a monolith result through every response row."""
    for (method, version), row in methods.SERVER_RESULTS.items():
        parsed = methods.parse_server_result(method, version, RESULT_BODY_FIXTURES[row])
        assert isinstance(parsed, types.Result), f"{method} at {version}"
    for (method, version), row in methods.CLIENT_RESULTS.items():
        parsed = methods.parse_client_result(method, version, RESULT_BODY_FIXTURES[row])
        assert isinstance(parsed, types.Result), f"{method} at {version}"


def test_non_file_root_uri_passes_the_surface_step_and_rejects_at_the_monolith_step():
    """A non-file root URI is the one built-in shape the monolith rejects after the surface admits it.

    Pins a known divergence: the monolith's Root.uri is file-scheme only while
    both surface packages declare a plain string (matching the schema), so the
    monolith step's ValidationError is reachable for built-in rows. Outcome is
    identical to validating roots through the monolith directly.
    """
    non_file_roots = {"roots": [{"uri": "https://example.com/x"}]}
    # The surface step alone admits the body...
    pydantic.TypeAdapter(v2025.ListRootsResult).validate_python(non_file_roots)
    # ...so the two-step parse must fail at the monolith step.
    with pytest.raises(pydantic.ValidationError):
        methods.parse_client_result("roots/list", "2025-11-25", non_file_roots)

    # Same divergence through the one 2026 path that embeds a roots response.
    retry_params = {"_meta": META_TRIPLE, "name": "echo", "inputResponses": {"r1": non_file_roots}}
    frame = {"jsonrpc": "2.0", "id": 0, "method": "tools/call", "params": retry_params}
    v2026.CallToolRequest.model_validate(frame, by_name=False)
    with pytest.raises(pydantic.ValidationError):
        methods.parse_client_request("tools/call", "2026-07-28", retry_params)

    # A file-scheme root URI parses end to end on both paths.
    file_roots = {"roots": [{"uri": "file:///workspace"}]}
    assert isinstance(methods.parse_client_result("roots/list", "2025-11-25", file_roots), types.ListRootsResult)
    retried = methods.parse_client_request(
        "tools/call", "2026-07-28", {"_meta": META_TRIPLE, "name": "echo", "inputResponses": {"r1": file_roots}}
    )
    assert isinstance(retried, types.CallToolRequest)


def test_absent_map_keys_raise_key_error_for_every_gate_shape():
    """A missing (method, version) key raises KeyError for removed, future, extension, and wrong-direction methods.

    Spec-mandated gate: absence of a key IS the version gate; the session
    layer maps it to METHOD_NOT_FOUND or drop-and-log.
    """
    gated = [
        ("resources/subscribe", "2026-07-28"),  # removed at this version
        ("server/discover", "2025-11-25"),  # does not exist yet at this version
        ("tasks/get", "2025-11-25"),  # never built-in; extensions register it
        ("sampling/createMessage", "2025-11-25"),  # wrong direction for this map
    ]
    for method, version in gated:
        with pytest.raises(KeyError):
            methods.parse_client_request(method, version, None)
    # No server-to-client request exists at 2026-07-28: every method gates.
    with pytest.raises(KeyError):
        methods.parse_server_request("ping", "2026-07-28", None)


def test_unknown_version_strings_raise_value_error_on_every_parse_function():
    """An unknown version string raises ValueError naming it, even for a method that exists at known versions.

    SDK-defined guard: a typo must crash loudly instead of silently gating
    every method to METHOD_NOT_FOUND.
    """
    body_parsers = [
        methods.parse_client_request,
        methods.parse_server_request,
        methods.parse_client_notification,
        methods.parse_server_notification,
    ]
    for body_parser in body_parsers:
        with pytest.raises(ValueError) as excinfo:
            body_parser("ping", "2099-01-01", None)
        assert "2099-01-01" in str(excinfo.value)
    result_parsers = [methods.parse_server_result, methods.parse_client_result]
    for result_parser in result_parsers:
        with pytest.raises(ValueError) as excinfo:
            result_parser("ping", "2099-01-01", {})
        assert "2099-01-01" in str(excinfo.value)


def test_2026_07_28_requests_missing_a_reserved_meta_entry_reject_as_missing():
    """A 2026-07-28 request missing any reserved _meta identity entry rejects with a missing error at that entry.

    Spec-mandated: the 2026-07-28 schema requires all three entries on every
    client request; the surface types enforce it.
    """
    for absent_key in META_TRIPLE:
        partial_meta = {key: value for key, value in META_TRIPLE.items() if key != absent_key}
        with pytest.raises(pydantic.ValidationError) as excinfo:
            methods.parse_client_request("tools/list", "2026-07-28", {"_meta": partial_meta})
        assert [error["loc"] for error in excinfo.value.errors() if error["type"] == "missing"] == [
            ("params", "_meta", absent_key)
        ]


def test_2026_07_28_results_require_result_type():
    """A 2026-07-28 result body without resultType rejects, including the empty body.

    Spec-mandated: the 2026-07-28 schema requires resultType on every result.
    """
    with pytest.raises(pydantic.ValidationError):
        methods.parse_server_result("tools/call", "2026-07-28", {"content": []})
    with pytest.raises(pydantic.ValidationError):
        methods.parse_server_result("subscriptions/listen", "2026-07-28", {})


def test_empty_result_body_parses_at_versions_that_define_it():
    """An empty result body parses as EmptyResult where the version's response row is the empty result."""
    parsed = methods.parse_server_result("ping", "2025-11-25", {})
    assert isinstance(parsed, types.EmptyResult)


def test_2026_07_28_shaped_result_extras_pass_at_earlier_versions():
    """A result carrying resultType/ttlMs/cacheScope parses at 2025-11-25, extras flowing into the monolith fields.

    Pins documented leniency: the earlier surface ignores unknown keys, so
    2026-07-28-shaped extras never reject on earlier sessions; the monolith
    preserves the values on classes that declare the fields.
    """
    parsed = methods.parse_server_result(
        "tools/list", "2025-11-25", {"tools": [], "resultType": "complete", "ttlMs": 5, "cacheScope": "public"}
    )
    assert isinstance(parsed, types.ListToolsResult)
    assert parsed.result_type == "complete"
    assert parsed.ttl_ms == 5
    assert parsed.cache_scope == "public"


def test_embedded_input_request_entries_without_method_reject_at_the_surface_step():
    """An inputRequests entry with no method rejects at the surface step, which the monolith alone would admit.

    Spec-mandated: the 2026-07-28 embedded request shapes require method. The
    monolith's embedded request classes default their method literals, so the
    rejection can only come from the surface step; this asserts both halves.
    """
    body = {"resultType": "input_required", "inputRequests": {"r1": {"params": None}}}
    monolith_row = methods.MONOLITH_RESULTS["tools/call"]
    monolith_only: types.Result = pydantic.TypeAdapter[Any](monolith_row).validate_python(body)
    assert isinstance(monolith_only, types.InputRequiredResult)
    with pytest.raises(pydantic.ValidationError):
        methods.parse_server_result("tools/call", "2026-07-28", body)


def test_none_params_omit_the_key_so_required_params_reject():
    """params=None omits the params key entirely: methods with required params reject, optional ones parse."""
    with pytest.raises(pydantic.ValidationError) as excinfo:
        methods.parse_client_request("tools/call", "2025-11-25", None)
    assert [error["loc"] for error in excinfo.value.errors() if error["type"] == "missing"] == [("params",)]
    assert isinstance(methods.parse_client_request("ping", "2025-11-25", None), types.PingRequest)


def test_snake_case_spellings_of_required_aliased_fields_reject_as_missing():
    """Wire parsing is alias-only: a required aliased field sent in its snake-case spelling rejects as missing.

    SDK-defined contract (matches the session layer's by_name=False parsing).
    Both steps enforce it; the surface step fires first through the parse
    function, and the monolith class enforces the same rule directly.
    """
    snake_params = {"messages": [{"role": "user", "content": {"type": "text", "text": "hi"}}], "max_tokens": 100}
    with pytest.raises(pydantic.ValidationError) as excinfo:
        methods.parse_server_request("sampling/createMessage", "2025-11-25", snake_params)
    assert [error["loc"] for error in excinfo.value.errors() if error["type"] == "missing"] == [("params", "maxTokens")]
    with pytest.raises(pydantic.ValidationError):
        types.CreateMessageRequest.model_validate(
            {"method": "sampling/createMessage", "params": snake_params}, by_name=False
        )


def test_extension_map_rows_parse_through_the_same_functions():
    """Dict-union extension rows parse end to end through the unchanged parse functions."""
    extended_surface = {**methods.CLIENT_REQUESTS, ("tasks/get", "2025-11-25"): v2025.GetTaskRequest}
    extended_monolith = {**methods.MONOLITH_REQUESTS, "tasks/get": types.GetTaskRequest}
    parsed = methods.parse_client_request(
        "tasks/get", "2025-11-25", {"taskId": "t1"}, surface=extended_surface, monolith=extended_monolith
    )
    assert isinstance(parsed, types.GetTaskRequest)
    assert parsed.params.task_id == "t1"


def test_inconsistent_extension_maps_raise_runtime_error_after_the_surface_hit():
    """A surface row whose method has no monolith row raises RuntimeError, never the gate's KeyError.

    SDK-defined: the session layer treats KeyError as the version gate, so
    inconsistent extension maps must fail distinguishably.
    """
    extended_surface = {**methods.CLIENT_REQUESTS, ("tasks/get", "2025-11-25"): v2025.GetTaskRequest}
    with pytest.raises(RuntimeError, match="inconsistent extension maps"):
        methods.parse_client_request("tasks/get", "2025-11-25", {"taskId": "t1"}, surface=extended_surface)


def test_input_required_unions_discriminate_identically_in_both_arm_orders():
    """The three dual-result unions resolve every body shape to the same arm regardless of arm order.

    SDK-defined: discrimination rides on the models' resultType literals and
    required fields, so these unions are order-insensitive (unlike the
    sampling union, pinned separately below).
    """
    complete_bodies: dict[str, dict[str, Any]] = {
        "tools/call": {"content": []},
        "prompts/get": {"messages": []},
        "resources/read": {"contents": []},
    }
    shared_bodies: list[dict[str, Any]] = [
        {"resultType": "input_required", "inputRequests": {"r1": {"method": "roots/list"}}},
        {"resultType": "input_required", "requestState": "blob"},
    ]
    for method, complete_body in complete_bodies.items():
        row = methods.MONOLITH_RESULTS[method]
        complete_arm, input_required_arm = get_args(row)
        assert input_required_arm is types.InputRequiredResult
        bodies: list[dict[str, Any]] = [
            complete_body,  # absent tag takes the "complete" default
            {**complete_body, "resultType": "complete"},  # explicit tag
            *shared_bodies,  # input-required via inputRequests and via requestState
            {**complete_body, "resultType": "task"},  # open tag is preserved, not rejected
            {**complete_body, "resultType": "input_required"},  # pathological: complete shape plus the tag
        ]
        for body in bodies:
            forward = pydantic.TypeAdapter[Any](complete_arm | input_required_arm).validate_python(body)
            reversed_order = pydantic.TypeAdapter[Any](input_required_arm | complete_arm).validate_python(body)
            assert type(forward) is type(reversed_order), f"{method}: {body}"
            assert forward.result_type == reversed_order.result_type
        through_row = pydantic.TypeAdapter[Any](row).validate_python(complete_body)
        assert isinstance(through_row, complete_arm)
        open_tagged = pydantic.TypeAdapter[Any](row).validate_python({**complete_body, "resultType": "task"})
        assert open_tagged.result_type == "task"


def test_sampling_union_keeps_the_complete_arm_first_because_order_is_load_bearing():
    """The sampling result union pins CreateMessageResult first: a single-block body satisfies both arms.

    SDK-defined and order-sensitive: smart-union ties resolve leftmost, so
    reversing the arms silently reparses single-block bodies as the
    with-tools class. Array and tool-use bodies discriminate by shape.
    """
    assert get_args(methods.MONOLITH_RESULTS["sampling/createMessage"]) == (
        types.CreateMessageResult,
        types.CreateMessageResultWithTools,
    )
    single_block: dict[str, Any] = {"role": "assistant", "content": {"type": "text", "text": "hi"}, "model": "m"}
    through_row = methods.parse_client_result("sampling/createMessage", "2025-11-25", single_block)
    assert type(through_row) is types.CreateMessageResult
    # The regression a reorder causes, demonstrated on the reversed union.
    reversed_union = pydantic.TypeAdapter[Any](types.CreateMessageResultWithTools | types.CreateMessageResult)
    assert type(reversed_union.validate_python(single_block)) is types.CreateMessageResultWithTools

    array_body: dict[str, Any] = {"role": "assistant", "content": [{"type": "text", "text": "hi"}], "model": "m"}
    tool_use_body: dict[str, Any] = {
        "role": "assistant",
        "content": {"type": "tool_use", "name": "t", "id": "c1", "input": {}},
        "model": "m",
    }
    for body in (array_body, tool_use_body):
        parsed = methods.parse_client_result("sampling/createMessage", "2025-11-25", body)
        assert type(parsed) is types.CreateMessageResultWithTools


def test_importing_the_module_builds_no_adapters_and_identical_rows_share_one():
    """Import constructs zero result adapters; the first parse builds them and identical rows share a cache entry.

    SDK-defined import-cost contract. Executes a fresh copy of the module (so
    the assertion is order-independent) and reads its private adapter cache,
    because lazy construction is exactly the contract under test.
    """
    spec = importlib.util.find_spec("mcp.types.methods")
    assert spec is not None and spec.loader is not None
    fresh = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fresh)
    assert fresh._adapter.cache_info().currsize == 0
    # One result parse builds exactly two adapters: the surface row and the monolith row.
    fresh.parse_server_result("ping", "2025-11-25", {})
    assert fresh._adapter.cache_info().currsize == 2
    # The same rows at another version are identical objects: no new adapters.
    fresh.parse_server_result("ping", "2024-11-05", {})
    assert fresh._adapter.cache_info().currsize == 2
