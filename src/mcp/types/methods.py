"""Straight wire-method maps and two-step parse functions for the MCP protocol.

Two families of plain maps; nothing here negotiates, dispatches, or holds
session state.

Integration status: nothing in the SDK consults these maps or parse
functions yet -- sessions still validate traffic against the version-free
monolith types alone, and rewiring them onto this module is a follow-up
change. Statements in this module about what sessions do with a lookup or a
raised exception (the session layer mapping errors, servers and clients
gating on a map) describe that intended integration contract, not behavior
the SDK has today.

SURFACE maps are keyed ``(method, version)`` and valued by schema-exact
surface types (``mcp.types.v2025_11_25`` / ``mcp.types.v2026_07_28``): one
entry per method per protocol version that defines it. Absence of a key IS
the version gate: the contract is for the session layer to map a failed
request lookup to JSON-RPC -32601 (METHOD_NOT_FOUND) and to drop-and-log a
failed notification lookup.

Schema-exact comes with a granularity caveat: the method gate is
version-exact, but shape validation is only as fine-grained as the two
surface packages. Every version through 2025-11-25 validates against the
2025-11-25-shaped ``v2025_11_25`` models (those schemas evolve strictly
additively), so a 2024-11-05 session accepts later pre-2026 shapes such as
audio content or a cancellation with no ``requestId``. Across the
2025-11-25/2026-07-28 boundary the packages disagree on requiredness and
value shapes, and validation fails loudly in both directions there.

MONOLITH maps are keyed by method alone and valued by the version-free
superset types in ``mcp.types``: what user code receives. The result side is
a two-arm union where the wire is structurally dual (``tools/call``,
``prompts/get``, ``resources/read`` may answer ``InputRequiredResult`` on
sessions negotiated at 2026-07-28 or later; ``sampling/createMessage`` may
answer with array content). The three InputRequired unions self-discriminate
through the models' ``resultType`` literals; the sampling union does not --
a single-block body satisfies both of its arms, so it relies on pinned arm
order instead (see the ``MONOLITH_RESULTS`` docstring and its pinning test).
Either way there is no dispatch code here.

Parsing is two steps (the ``parse_*`` functions below): the surface type
VALIDATES, then the monolith type deserializes the same payload and is
returned. Unknown keys never fail the surface check (the surface base
ignores extras); what reaches user code is what the monolith parse keeps --
the monolith-declared fields plus ``_meta`` extras -- and a key neither
layer declares is dropped.

Outbound serialization does not live here: emission is the model's plain
``model_dump(by_alias=True, mode="json", exclude_none=True)`` at every
version, with no version parameter, no strips, and no injections.

Extension methods are dict unions over the built-ins, handed to the parse
functions' keyword parameters::

    requests = {**CLIENT_REQUESTS, ("tasks/get", "2025-11-25"): v2025.GetTaskRequest}
    bodies = {**MONOLITH_REQUESTS, "tasks/get": types.GetTaskRequest}
    parse_client_request(method, version, params, surface=requests, monolith=bodies)

The built-in maps are immutable (``MappingProxyType``). The ``tasks/*``
family (four request methods and ``notifications/tasks/status``) is
deliberately absent from the built-ins: the SDK defines the task types but
never dispatches them; extensions register them.

``initialize`` and ``ping`` are ordinary rows, not special cases: they
appear at every version whose schema defines them (every version through
2025-11-25; neither exists at 2026-07-28, so plain key absence gates them
there like any removed method), and a lookup at a defining version
validates through its row like any other. What bypasses the maps is the
pre-negotiation window only: before a version is negotiated there is
nothing to key a lookup on, so the contract validates that exchange (the
``initialize`` handshake and any early ``ping``) directly against the
monolith types.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import cache
from types import MappingProxyType, UnionType
from typing import Any, Final, TypeVar

from pydantic import BaseModel, TypeAdapter

import mcp.types as types
import mcp.types.v2025_11_25 as v2025
import mcp.types.v2026_07_28 as v2026
from mcp.shared.version import KNOWN_PROTOCOL_VERSIONS
from mcp.types._wire_base import WireModel

__all__ = [
    "CLIENT_NOTIFICATIONS",
    "CLIENT_REQUESTS",
    "CLIENT_RESULTS",
    "MONOLITH_NOTIFICATIONS",
    "MONOLITH_REQUESTS",
    "MONOLITH_RESULTS",
    "SERVER_NOTIFICATIONS",
    "SERVER_REQUESTS",
    "SERVER_RESULTS",
    "parse_client_notification",
    "parse_client_request",
    "parse_client_result",
    "parse_server_notification",
    "parse_server_request",
    "parse_server_result",
]


# --- Surface maps: client-to-server direction (servers validate inbound) ---

CLIENT_REQUESTS: Final[Mapping[tuple[str, str], type[WireModel]]] = MappingProxyType(
    {
        # 2024-11-05
        ("completion/complete", "2024-11-05"): v2025.CompleteRequest,
        ("initialize", "2024-11-05"): v2025.InitializeRequest,
        ("logging/setLevel", "2024-11-05"): v2025.SetLevelRequest,
        ("ping", "2024-11-05"): v2025.PingRequest,
        ("prompts/get", "2024-11-05"): v2025.GetPromptRequest,
        ("prompts/list", "2024-11-05"): v2025.ListPromptsRequest,
        ("resources/list", "2024-11-05"): v2025.ListResourcesRequest,
        ("resources/read", "2024-11-05"): v2025.ReadResourceRequest,
        ("resources/subscribe", "2024-11-05"): v2025.SubscribeRequest,
        ("resources/templates/list", "2024-11-05"): v2025.ListResourceTemplatesRequest,
        ("resources/unsubscribe", "2024-11-05"): v2025.UnsubscribeRequest,
        ("tools/call", "2024-11-05"): v2025.CallToolRequest,
        ("tools/list", "2024-11-05"): v2025.ListToolsRequest,
        # 2025-03-26
        ("completion/complete", "2025-03-26"): v2025.CompleteRequest,
        ("initialize", "2025-03-26"): v2025.InitializeRequest,
        ("logging/setLevel", "2025-03-26"): v2025.SetLevelRequest,
        ("ping", "2025-03-26"): v2025.PingRequest,
        ("prompts/get", "2025-03-26"): v2025.GetPromptRequest,
        ("prompts/list", "2025-03-26"): v2025.ListPromptsRequest,
        ("resources/list", "2025-03-26"): v2025.ListResourcesRequest,
        ("resources/read", "2025-03-26"): v2025.ReadResourceRequest,
        ("resources/subscribe", "2025-03-26"): v2025.SubscribeRequest,
        ("resources/templates/list", "2025-03-26"): v2025.ListResourceTemplatesRequest,
        ("resources/unsubscribe", "2025-03-26"): v2025.UnsubscribeRequest,
        ("tools/call", "2025-03-26"): v2025.CallToolRequest,
        ("tools/list", "2025-03-26"): v2025.ListToolsRequest,
        # 2025-06-18
        ("completion/complete", "2025-06-18"): v2025.CompleteRequest,
        ("initialize", "2025-06-18"): v2025.InitializeRequest,
        ("logging/setLevel", "2025-06-18"): v2025.SetLevelRequest,
        ("ping", "2025-06-18"): v2025.PingRequest,
        ("prompts/get", "2025-06-18"): v2025.GetPromptRequest,
        ("prompts/list", "2025-06-18"): v2025.ListPromptsRequest,
        ("resources/list", "2025-06-18"): v2025.ListResourcesRequest,
        ("resources/read", "2025-06-18"): v2025.ReadResourceRequest,
        ("resources/subscribe", "2025-06-18"): v2025.SubscribeRequest,
        ("resources/templates/list", "2025-06-18"): v2025.ListResourceTemplatesRequest,
        ("resources/unsubscribe", "2025-06-18"): v2025.UnsubscribeRequest,
        ("tools/call", "2025-06-18"): v2025.CallToolRequest,
        ("tools/list", "2025-06-18"): v2025.ListToolsRequest,
        # 2025-11-25 (the four tasks/* request methods are deliberately absent)
        ("completion/complete", "2025-11-25"): v2025.CompleteRequest,
        ("initialize", "2025-11-25"): v2025.InitializeRequest,
        ("logging/setLevel", "2025-11-25"): v2025.SetLevelRequest,
        ("ping", "2025-11-25"): v2025.PingRequest,
        ("prompts/get", "2025-11-25"): v2025.GetPromptRequest,
        ("prompts/list", "2025-11-25"): v2025.ListPromptsRequest,
        ("resources/list", "2025-11-25"): v2025.ListResourcesRequest,
        ("resources/read", "2025-11-25"): v2025.ReadResourceRequest,
        ("resources/subscribe", "2025-11-25"): v2025.SubscribeRequest,
        ("resources/templates/list", "2025-11-25"): v2025.ListResourceTemplatesRequest,
        ("resources/unsubscribe", "2025-11-25"): v2025.UnsubscribeRequest,
        ("tools/call", "2025-11-25"): v2025.CallToolRequest,
        ("tools/list", "2025-11-25"): v2025.ListToolsRequest,
        # 2026-07-28 (lifecycle, logging, subscribe pair removed; discover/listen added)
        ("completion/complete", "2026-07-28"): v2026.CompleteRequest,
        ("prompts/get", "2026-07-28"): v2026.GetPromptRequest,
        ("prompts/list", "2026-07-28"): v2026.ListPromptsRequest,
        ("resources/list", "2026-07-28"): v2026.ListResourcesRequest,
        ("resources/read", "2026-07-28"): v2026.ReadResourceRequest,
        ("resources/templates/list", "2026-07-28"): v2026.ListResourceTemplatesRequest,
        ("server/discover", "2026-07-28"): v2026.DiscoverRequest,
        ("subscriptions/listen", "2026-07-28"): v2026.SubscriptionsListenRequest,
        ("tools/call", "2026-07-28"): v2026.CallToolRequest,
        ("tools/list", "2026-07-28"): v2026.ListToolsRequest,
    }
)
"""Requests clients send, per protocol version. Servers gate and validate inbound requests here."""

CLIENT_NOTIFICATIONS: Final[Mapping[tuple[str, str], type[WireModel]]] = MappingProxyType(
    {
        # 2024-11-05
        ("notifications/cancelled", "2024-11-05"): v2025.CancelledNotification,
        ("notifications/initialized", "2024-11-05"): v2025.InitializedNotification,
        ("notifications/progress", "2024-11-05"): v2025.ProgressNotification,
        ("notifications/roots/list_changed", "2024-11-05"): v2025.RootsListChangedNotification,
        # 2025-03-26
        ("notifications/cancelled", "2025-03-26"): v2025.CancelledNotification,
        ("notifications/initialized", "2025-03-26"): v2025.InitializedNotification,
        ("notifications/progress", "2025-03-26"): v2025.ProgressNotification,
        ("notifications/roots/list_changed", "2025-03-26"): v2025.RootsListChangedNotification,
        # 2025-06-18
        ("notifications/cancelled", "2025-06-18"): v2025.CancelledNotification,
        ("notifications/initialized", "2025-06-18"): v2025.InitializedNotification,
        ("notifications/progress", "2025-06-18"): v2025.ProgressNotification,
        ("notifications/roots/list_changed", "2025-06-18"): v2025.RootsListChangedNotification,
        # 2025-11-25 (notifications/tasks/status is deliberately absent)
        ("notifications/cancelled", "2025-11-25"): v2025.CancelledNotification,
        ("notifications/initialized", "2025-11-25"): v2025.InitializedNotification,
        ("notifications/progress", "2025-11-25"): v2025.ProgressNotification,
        ("notifications/roots/list_changed", "2025-11-25"): v2025.RootsListChangedNotification,
        # 2026-07-28 (initialized removed with the lifecycle; roots/list_changed removed with the roots channel)
        ("notifications/cancelled", "2026-07-28"): v2026.CancelledNotification,
        ("notifications/progress", "2026-07-28"): v2026.ProgressNotification,
    }
)
"""Notifications clients send, per protocol version. Servers gate and validate inbound notifications here."""


# --- Surface maps: server-to-client direction (clients validate inbound) ---

SERVER_REQUESTS: Final[Mapping[tuple[str, str], type[WireModel]]] = MappingProxyType(
    {
        # 2024-11-05
        ("ping", "2024-11-05"): v2025.PingRequest,
        ("roots/list", "2024-11-05"): v2025.ListRootsRequest,
        ("sampling/createMessage", "2024-11-05"): v2025.CreateMessageRequest,
        # 2025-03-26
        ("ping", "2025-03-26"): v2025.PingRequest,
        ("roots/list", "2025-03-26"): v2025.ListRootsRequest,
        ("sampling/createMessage", "2025-03-26"): v2025.CreateMessageRequest,
        # 2025-06-18 (adds elicitation/create)
        ("elicitation/create", "2025-06-18"): v2025.ElicitRequest,
        ("ping", "2025-06-18"): v2025.PingRequest,
        ("roots/list", "2025-06-18"): v2025.ListRootsRequest,
        ("sampling/createMessage", "2025-06-18"): v2025.CreateMessageRequest,
        # 2025-11-25 (the four tasks/* request methods are deliberately absent)
        ("elicitation/create", "2025-11-25"): v2025.ElicitRequest,
        ("ping", "2025-11-25"): v2025.PingRequest,
        ("roots/list", "2025-11-25"): v2025.ListRootsRequest,
        ("sampling/createMessage", "2025-11-25"): v2025.CreateMessageRequest,
        # 2026-07-28: no entries. The standalone server-to-client request
        # channel does not exist at this version (no ServerRequest union in
        # the schema); ANY server-initiated request on a 2026-07-28 session
        # is therefore gated to -32601 by plain key absence.
    }
)
"""Requests servers send, per protocol version. Clients gate and validate inbound requests here."""

SERVER_NOTIFICATIONS: Final[Mapping[tuple[str, str], type[WireModel]]] = MappingProxyType(
    {
        # 2024-11-05
        ("notifications/cancelled", "2024-11-05"): v2025.CancelledNotification,
        ("notifications/message", "2024-11-05"): v2025.LoggingMessageNotification,
        ("notifications/progress", "2024-11-05"): v2025.ProgressNotification,
        ("notifications/prompts/list_changed", "2024-11-05"): v2025.PromptListChangedNotification,
        ("notifications/resources/list_changed", "2024-11-05"): v2025.ResourceListChangedNotification,
        ("notifications/resources/updated", "2024-11-05"): v2025.ResourceUpdatedNotification,
        ("notifications/tools/list_changed", "2024-11-05"): v2025.ToolListChangedNotification,
        # 2025-03-26
        ("notifications/cancelled", "2025-03-26"): v2025.CancelledNotification,
        ("notifications/message", "2025-03-26"): v2025.LoggingMessageNotification,
        ("notifications/progress", "2025-03-26"): v2025.ProgressNotification,
        ("notifications/prompts/list_changed", "2025-03-26"): v2025.PromptListChangedNotification,
        ("notifications/resources/list_changed", "2025-03-26"): v2025.ResourceListChangedNotification,
        ("notifications/resources/updated", "2025-03-26"): v2025.ResourceUpdatedNotification,
        ("notifications/tools/list_changed", "2025-03-26"): v2025.ToolListChangedNotification,
        # 2025-06-18
        ("notifications/cancelled", "2025-06-18"): v2025.CancelledNotification,
        ("notifications/message", "2025-06-18"): v2025.LoggingMessageNotification,
        ("notifications/progress", "2025-06-18"): v2025.ProgressNotification,
        ("notifications/prompts/list_changed", "2025-06-18"): v2025.PromptListChangedNotification,
        ("notifications/resources/list_changed", "2025-06-18"): v2025.ResourceListChangedNotification,
        ("notifications/resources/updated", "2025-06-18"): v2025.ResourceUpdatedNotification,
        ("notifications/tools/list_changed", "2025-06-18"): v2025.ToolListChangedNotification,
        # 2025-11-25 (adds elicitation/complete; notifications/tasks/status deliberately absent)
        ("notifications/cancelled", "2025-11-25"): v2025.CancelledNotification,
        ("notifications/elicitation/complete", "2025-11-25"): v2025.ElicitationCompleteNotification,
        ("notifications/message", "2025-11-25"): v2025.LoggingMessageNotification,
        ("notifications/progress", "2025-11-25"): v2025.ProgressNotification,
        ("notifications/prompts/list_changed", "2025-11-25"): v2025.PromptListChangedNotification,
        ("notifications/resources/list_changed", "2025-11-25"): v2025.ResourceListChangedNotification,
        ("notifications/resources/updated", "2025-11-25"): v2025.ResourceUpdatedNotification,
        ("notifications/tools/list_changed", "2025-11-25"): v2025.ToolListChangedNotification,
        # 2026-07-28 (adds subscriptions/acknowledged)
        ("notifications/cancelled", "2026-07-28"): v2026.CancelledNotification,
        ("notifications/elicitation/complete", "2026-07-28"): v2026.ElicitationCompleteNotification,
        ("notifications/message", "2026-07-28"): v2026.LoggingMessageNotification,
        ("notifications/progress", "2026-07-28"): v2026.ProgressNotification,
        ("notifications/prompts/list_changed", "2026-07-28"): v2026.PromptListChangedNotification,
        ("notifications/resources/list_changed", "2026-07-28"): v2026.ResourceListChangedNotification,
        ("notifications/resources/updated", "2026-07-28"): v2026.ResourceUpdatedNotification,
        ("notifications/subscriptions/acknowledged", "2026-07-28"): v2026.SubscriptionsAcknowledgedNotification,
        ("notifications/tools/list_changed", "2026-07-28"): v2026.ToolListChangedNotification,
    }
)
"""Notifications servers send, per protocol version. Clients gate and validate inbound notifications here."""


# --- Surface maps: responses, keyed by the (method, version) the sender used ---

SERVER_RESULTS: Final[Mapping[tuple[str, str], type[WireModel] | UnionType]] = MappingProxyType(
    {
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
        # 2026-07-28 (dual-result rows point at the version's union aliases:
        # which result arms a session can see is row DATA, never dispatch code)
        ("completion/complete", "2026-07-28"): v2026.CompleteResult,
        ("prompts/get", "2026-07-28"): v2026.AnyGetPromptResult,
        ("prompts/list", "2026-07-28"): v2026.ListPromptsResult,
        ("resources/list", "2026-07-28"): v2026.ListResourcesResult,
        ("resources/read", "2026-07-28"): v2026.AnyReadResourceResult,
        ("resources/templates/list", "2026-07-28"): v2026.ListResourceTemplatesResult,
        ("server/discover", "2026-07-28"): v2026.DiscoverResult,
        ("subscriptions/listen", "2026-07-28"): v2026.EmptyResult,
        ("tools/call", "2026-07-28"): v2026.AnyCallToolResult,
        ("tools/list", "2026-07-28"): v2026.ListToolsResult,
    }
)
"""Results servers send, keyed by the client request's (method, version). Clients validate inbound responses here."""

CLIENT_RESULTS: Final[Mapping[tuple[str, str], type[WireModel] | UnionType]] = MappingProxyType(
    {
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
        # 2026-07-28: no entries (no server-to-client requests, so no
        # client-sent results; embedded InputResponses travel inside retried
        # client requests and are validated by the request rows above).
    }
)
"""Results clients send, keyed by the server request's (method, version). Servers validate inbound responses here."""


# --- Monolith maps: version-free, what user code receives ---

MONOLITH_REQUESTS: Final[Mapping[str, type[types.Request[Any, Any]]]] = MappingProxyType(
    {
        "completion/complete": types.CompleteRequest,
        "elicitation/create": types.ElicitRequest,
        "initialize": types.InitializeRequest,
        "logging/setLevel": types.SetLevelRequest,
        "ping": types.PingRequest,
        "prompts/get": types.GetPromptRequest,
        "prompts/list": types.ListPromptsRequest,
        "resources/list": types.ListResourcesRequest,
        "resources/read": types.ReadResourceRequest,
        "resources/subscribe": types.SubscribeRequest,
        "resources/templates/list": types.ListResourceTemplatesRequest,
        "resources/unsubscribe": types.UnsubscribeRequest,
        "roots/list": types.ListRootsRequest,
        "sampling/createMessage": types.CreateMessageRequest,
        "server/discover": types.DiscoverRequest,
        "subscriptions/listen": types.SubscriptionsListenRequest,
        "tools/call": types.CallToolRequest,
        "tools/list": types.ListToolsRequest,
    }
)
"""Monolith request model per method, both directions (ping appears once; the types are identical)."""

MONOLITH_NOTIFICATIONS: Final[Mapping[str, type[types.Notification[Any, Any]]]] = MappingProxyType(
    {
        "notifications/cancelled": types.CancelledNotification,
        "notifications/elicitation/complete": types.ElicitCompleteNotification,
        "notifications/initialized": types.InitializedNotification,
        "notifications/message": types.LoggingMessageNotification,
        "notifications/progress": types.ProgressNotification,
        "notifications/prompts/list_changed": types.PromptListChangedNotification,
        "notifications/resources/list_changed": types.ResourceListChangedNotification,
        "notifications/resources/updated": types.ResourceUpdatedNotification,
        "notifications/roots/list_changed": types.RootsListChangedNotification,
        "notifications/subscriptions/acknowledged": types.SubscriptionsAcknowledgedNotification,
        "notifications/tools/list_changed": types.ToolListChangedNotification,
    }
)
"""Monolith notification model per method, both directions."""

MONOLITH_RESULTS: Final[Mapping[str, type[types.Result] | UnionType]] = MappingProxyType(
    {
        "completion/complete": types.CompleteResult,
        "elicitation/create": types.ElicitResult,
        "initialize": types.InitializeResult,
        "logging/setLevel": types.EmptyResult,
        "ping": types.EmptyResult,
        "prompts/get": types.GetPromptResult | types.InputRequiredResult,
        "prompts/list": types.ListPromptsResult,
        "resources/list": types.ListResourcesResult,
        "resources/read": types.ReadResourceResult | types.InputRequiredResult,
        "resources/subscribe": types.EmptyResult,
        "resources/templates/list": types.ListResourceTemplatesResult,
        "resources/unsubscribe": types.EmptyResult,
        "roots/list": types.ListRootsResult,
        # Arm order LOAD-BEARING: complete arm first. A single-block body satisfies
        # both arms (CreateMessageResultWithTools.content also accepts a single
        # block) and pydantic smart-union ties resolve leftmost; reversing the
        # arms silently changes wire-visible parsing. Pinned by
        # tests/types/test_methods.py::test_sampling_union_keeps_the_complete_arm_first_because_order_is_load_bearing.
        "sampling/createMessage": types.CreateMessageResult | types.CreateMessageResultWithTools,
        "server/discover": types.DiscoverResult,
        "subscriptions/listen": types.EmptyResult,
        "tools/call": types.CallToolResult | types.InputRequiredResult,
        "tools/list": types.ListToolsResult,
    }
)
"""Monolith result model (or self-discriminating two-arm union) per request method.

Membership is version-free superset data: which arm a given session can
actually see is the response surface maps' job. The InputRequired arms
discriminate through the models' resultType literals and are order-insensitive
(verified both orders); the sampling arms discriminate through content shape
(single block vs array / tool-use blocks) and their order IS load-bearing:
the complete arm must stay first, because a single-block body satisfies both
arms and smart-union ties resolve leftmost. No dispatch code anywhere.
"""


# --- Two-step parse functions -----------------------------------------------

_REQUEST_STUB: Final[Mapping[str, Any]] = MappingProxyType({"jsonrpc": "2.0", "id": 0})
"""Envelope stub merged for surface request checks (surface classes are full frames)."""

_NOTIFICATION_STUB: Final[Mapping[str, Any]] = MappingProxyType({"jsonrpc": "2.0"})
"""Envelope stub merged for surface notification checks."""


def _check_known_version(version: str) -> None:
    """Reject version strings outside KNOWN_PROTOCOL_VERSIONS with ValueError.

    The negotiated session version is always a member (negotiation validates
    against SUPPORTED_PROTOCOL_VERSIONS, a subset). Anything else is
    programmer error: a silent miss would turn a typo into every method
    answering -32601.
    """
    if version not in KNOWN_PROTOCOL_VERSIONS:
        raise ValueError(f"version must be a known protocol version, got {version!r}")


def _body(method: str, params: Mapping[str, Any] | None) -> dict[str, Any]:
    """The JSON-RPC body for a request or notification.

    The params key is omitted entirely when params is None, so a method
    whose params model has required fields rejects absent params.
    """
    body: dict[str, Any] = {"method": method}
    if params is not None:
        body["params"] = params
    return body


@cache
def _adapter(target: type[BaseModel] | UnionType) -> TypeAdapter[Any]:
    """A cached TypeAdapter for a result row (a model class or a union).

    Built lazily on first use, so importing this module constructs no
    pydantic adapters. The cache key is the row object itself: identical
    rows shared across versions build one adapter, and extension rows are
    cached the same way (classes and union expressions are both hashable).
    """
    return TypeAdapter(target)


_MonolithT = TypeVar("_MonolithT")


def _monolith_row(monolith: Mapping[str, _MonolithT], method: str) -> _MonolithT:
    """The monolith row for a method whose surface row already matched.

    A miss here is never the version gate (the surface lookup already
    passed): it means the surface and monolith maps disagree -- inconsistent
    extension maps, programmer error. Raised as RuntimeError, not KeyError,
    so the gate handling the contract assigns to the session layer
    (``except KeyError`` -> -32601 or drop-and-log) cannot swallow it.
    """
    try:
        return monolith[method]
    except KeyError:
        raise RuntimeError(f"inconsistent extension maps: surface defines {method!r} but monolith does not") from None


def parse_client_request(
    method: str,
    version: str,
    params: Mapping[str, Any] | None,
    *,
    surface: Mapping[tuple[str, str], type[WireModel]] = CLIENT_REQUESTS,
    monolith: Mapping[str, type[types.Request[Any, Any]]] = MONOLITH_REQUESTS,
) -> types.Request[Any, Any]:
    """Parse an inbound client-to-server request body (server side).

    Two steps: the surface type for (method, version) VALIDATES the request
    (a constant envelope stub supplies jsonrpc/id), then the monolith type
    for method deserializes the same body and is returned -- user handlers
    are typed against mcp.types. The method gate is version-exact; shape
    validation is as fine-grained as the surface package serving the
    negotiated version (every version through 2025-11-25 validates
    2025-11-25-shaped -- see the module docstring). Unknown keys never fail
    the surface check (surface ignores extras); the returned monolith model
    carries its declared fields plus ``_meta`` extras, and a key neither
    layer declares is dropped.

    Pass extended maps (dict unions over the built-ins) to serve extension
    methods; the built-ins never change.

    Raises:
        ValueError: version is not a known protocol version.
        KeyError: (method, version) is not in surface -- the method does not
            exist at the negotiated version (or at all). The session-layer
            contract maps this to JSON-RPC -32601.
        pydantic.ValidationError: the body is invalid at this version, or
            passes the surface step but fails the monolith parse -- possible
            where the monolith declares a stricter constrained type than the
            surfaces (the one audited built-in instance: the monolith's
            Root.uri is file-scheme FileUrl while the surfaces accept any
            string, so a 2026-07-28 retried request whose inputResponses
            embed a roots response with a non-file URI passes the surface
            and rejects here), or with shape-mismatched extension maps. The
            session-layer contract maps this to JSON-RPC -32602.
        RuntimeError: the surface row matched but the method has no monolith
            row (inconsistent extension maps; programmer error).
    """
    _check_known_version(version)
    surface_type = surface[(method, version)]
    surface_type.model_validate({**_REQUEST_STUB, **_body(method, params)}, by_name=False)
    return _monolith_row(monolith, method).model_validate(_body(method, params), by_name=False)


def parse_server_request(
    method: str,
    version: str,
    params: Mapping[str, Any] | None,
    *,
    surface: Mapping[tuple[str, str], type[WireModel]] = SERVER_REQUESTS,
    monolith: Mapping[str, type[types.Request[Any, Any]]] = MONOLITH_REQUESTS,
) -> types.Request[Any, Any]:
    """Parse an inbound server-to-client request body (client side).

    Same contract as parse_client_request, gated by SERVER_REQUESTS. On
    2026-07-28 sessions that map has no rows, so every server-initiated
    request raises KeyError (-32601 under the session-layer contract) with
    no special-casing: the table encodes the channel removal.

    Raises:
        ValueError: version is not a known protocol version.
        KeyError: (method, version) is not in surface.
        pydantic.ValidationError: the body is invalid at this version.
        RuntimeError: the surface row matched but the method has no monolith
            row (inconsistent extension maps; programmer error).
    """
    _check_known_version(version)
    surface_type = surface[(method, version)]
    surface_type.model_validate({**_REQUEST_STUB, **_body(method, params)}, by_name=False)
    return _monolith_row(monolith, method).model_validate(_body(method, params), by_name=False)


def parse_client_notification(
    method: str,
    version: str,
    params: Mapping[str, Any] | None,
    *,
    surface: Mapping[tuple[str, str], type[WireModel]] = CLIENT_NOTIFICATIONS,
    monolith: Mapping[str, type[types.Notification[Any, Any]]] = MONOLITH_NOTIFICATIONS,
) -> types.Notification[Any, Any]:
    """Parse an inbound client-to-server notification body (server side).

    Same two steps as the request functions (the notification stub has no
    id). The session-layer contract maps KeyError to drop-and-log, never an
    error to the peer.

    Raises:
        ValueError: version is not a known protocol version.
        KeyError: (method, version) is not in surface.
        pydantic.ValidationError: the body is invalid at this version.
        RuntimeError: the surface row matched but the method has no monolith
            row (inconsistent extension maps; programmer error).
    """
    _check_known_version(version)
    surface_type = surface[(method, version)]
    surface_type.model_validate({**_NOTIFICATION_STUB, **_body(method, params)}, by_name=False)
    return _monolith_row(monolith, method).model_validate(_body(method, params), by_name=False)


def parse_server_notification(
    method: str,
    version: str,
    params: Mapping[str, Any] | None,
    *,
    surface: Mapping[tuple[str, str], type[WireModel]] = SERVER_NOTIFICATIONS,
    monolith: Mapping[str, type[types.Notification[Any, Any]]] = MONOLITH_NOTIFICATIONS,
) -> types.Notification[Any, Any]:
    """Parse an inbound server-to-client notification body (client side).

    Same contract as parse_client_notification, gated by SERVER_NOTIFICATIONS.

    Raises:
        ValueError: version is not a known protocol version.
        KeyError: (method, version) is not in surface.
        pydantic.ValidationError: the body is invalid at this version.
        RuntimeError: the surface row matched but the method has no monolith
            row (inconsistent extension maps; programmer error).
    """
    _check_known_version(version)
    surface_type = surface[(method, version)]
    surface_type.model_validate({**_NOTIFICATION_STUB, **_body(method, params)}, by_name=False)
    return _monolith_row(monolith, method).model_validate(_body(method, params), by_name=False)


def parse_server_result(
    method: str,
    version: str,
    data: Mapping[str, Any],
    *,
    surface: Mapping[tuple[str, str], type[WireModel] | UnionType] = SERVER_RESULTS,
    monolith: Mapping[str, type[types.Result] | UnionType] = MONOLITH_RESULTS,
) -> types.Result:
    """Parse the result of a client-to-server request this side sent (client side).

    data is the JSON-RPC result member. The surface row for the request's
    (method, version) VALIDATES it (2025-11-25-shaped on every version
    through 2025-11-25; a server answering with shapes its surface package
    does not define fails loudly here), then the monolith row deserializes
    it. Dual-result methods return whichever arm the body discriminates to;
    isinstance narrows at the caller.

    Raises:
        ValueError: version is not a known protocol version.
        KeyError: (method, version) is not in surface -- the caller sent a
            request that does not exist at the negotiated version.
        pydantic.ValidationError: the result body is invalid at this version.
        RuntimeError: the surface row matched but the method has no monolith
            row (inconsistent extension maps; programmer error).
    """
    _check_known_version(version)
    _adapter(surface[(method, version)]).validate_python(data, by_name=False)
    result: types.Result = _adapter(_monolith_row(monolith, method)).validate_python(data, by_name=False)
    return result


def parse_client_result(
    method: str,
    version: str,
    data: Mapping[str, Any],
    *,
    surface: Mapping[tuple[str, str], type[WireModel] | UnionType] = CLIENT_RESULTS,
    monolith: Mapping[str, type[types.Result] | UnionType] = MONOLITH_RESULTS,
) -> types.Result:
    """Parse the result of a server-to-client request this side sent (server side).

    Same contract as parse_server_result, gated by CLIENT_RESULTS. Empty on
    2026-07-28 (no server-to-client requests exist there).

    Raises:
        ValueError: version is not a known protocol version.
        KeyError: (method, version) is not in surface.
        pydantic.ValidationError: the result body is invalid at this version,
            at either step -- the monolith's Root.uri (file-scheme FileUrl)
            is stricter than the surfaces' plain-string uri, so a roots/list
            result carrying a non-file URI passes the surface step and
            rejects at the monolith step.
        RuntimeError: the surface row matched but the method has no monolith
            row (inconsistent extension maps; programmer error).
    """
    _check_known_version(version)
    _adapter(surface[(method, version)]).validate_python(data, by_name=False)
    result: types.Result = _adapter(_monolith_row(monolith, method)).validate_python(data, by_name=False)
    return result
