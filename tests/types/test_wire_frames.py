"""Frame-level pins for the SDK's outbound JSON-RPC serialization.

Each test builds a monolith payload model, dumps it with the session layer's
outbound convention (`model_dump(by_alias=True, mode="json", exclude_none=True)`),
wraps it in the matching `mcp.types.jsonrpc` envelope class, and pins the
frame exactly as the transports serialize it
(`model_dump_json(by_alias=True, exclude_unset=True)`, see the stdio
transports). The pinned strings are the SDK-defined serialization contract --
the spec constrains JSON content, not key order or the SDK's
always-serialized defaults -- so a diff here is a wire-visible change that
needs a deliberate decision, not necessarily a bug.
"""

from typing import Any

from inline_snapshot import snapshot
from pydantic import BaseModel

from mcp.types import (
    METHOD_NOT_FOUND,
    CallToolRequest,
    CallToolRequestParams,
    CallToolResult,
    EmptyResult,
    ErrorData,
    InputRequiredResult,
    JSONRPCError,
    JSONRPCNotification,
    JSONRPCRequest,
    JSONRPCResponse,
    ListRootsRequest,
    ListToolsResult,
    ProgressNotification,
    ProgressNotificationParams,
    TextContent,
    Tool,
)


def _body(model: BaseModel) -> dict[str, Any]:
    """The session layer's outbound dump: one plain dump at every protocol version."""
    return model.model_dump(by_alias=True, mode="json", exclude_none=True)


def _frame(envelope: BaseModel) -> str:
    """The transports' frame serialization (see the stdio transports)."""
    return envelope.model_dump_json(by_alias=True, exclude_unset=True)


def test_request_frame_carries_the_envelope_and_the_dumped_request_body():
    """A request frame is the JSON-RPC envelope around the monolith request dump, aliases applied."""
    request = CallToolRequest(params=CallToolRequestParams(name="echo", arguments={"text": "hi"}))
    frame = JSONRPCRequest(jsonrpc="2.0", id=1, **_body(request))
    assert _frame(frame) == snapshot(
        '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"echo","arguments":{"text":"hi"}}}'
    )


def test_notification_frame_has_no_id_and_carries_the_dumped_params():
    """A notification frame carries method and params only: the id key never appears."""
    notification = ProgressNotification(params=ProgressNotificationParams(progress_token="t1", progress=0.5))
    frame = JSONRPCNotification(jsonrpc="2.0", **_body(notification))
    assert _frame(frame) == snapshot(
        '{"jsonrpc":"2.0","method":"notifications/progress","params":{"progressToken":"t1","progress":0.5}}'
    )


def test_non_empty_result_frame_always_dumps_result_type_complete():
    """A non-empty result frame carries resultType "complete" without the handler setting it.

    SDK-defined always-serialized default for the field the 2026-07-28 schema
    requires on results; earlier peers ignore the extra key.
    """
    result = CallToolResult(content=[TextContent(text="ok")])
    frame = JSONRPCResponse(jsonrpc="2.0", id=1, result=_body(result))
    assert _frame(frame) == snapshot(
        '{"jsonrpc":"2.0","id":1,"result":{"content":[{"type":"text","text":"ok"}],"isError":false,"resultType":"complete"}}'
    )


def test_cacheable_list_result_frame_always_dumps_its_caching_directives():
    """A cacheable list result frame carries ttlMs 0 and cacheScope "private" without the handler setting them.

    SDK-defined always-serialized defaults for the fields the 2026-07-28
    schema requires on cacheable results; earlier peers ignore the extra keys.
    """
    result = ListToolsResult(tools=[Tool(name="echo", input_schema={"type": "object"})])
    frame = JSONRPCResponse(jsonrpc="2.0", id=2, result=_body(result))
    assert _frame(frame) == snapshot(
        '{"jsonrpc":"2.0","id":2,"result":{"ttlMs":0,"cacheScope":"private","tools":[{"name":"echo","inputSchema":{"type":"object"}}],"resultType":"complete"}}'
    )


def test_empty_result_frame_dumps_an_empty_result_object():
    """A default EmptyResult frame carries result {} with no resultType.

    SDK-defined carve-out: deployed peers validate empty results strictly and
    reject extra keys, so the SDK never volunteers resultType on them.
    """
    frame = JSONRPCResponse(jsonrpc="2.0", id=3, result=_body(EmptyResult()))
    assert _frame(frame) == snapshot('{"jsonrpc":"2.0","id":3,"result":{}}')


def test_input_required_result_frame_carries_the_tag_and_the_embedded_requests():
    """An input-required frame travels as a plain result whose body carries the discriminating tag.

    The embedded server-initiated request dumps inside inputRequests; no
    JSON-RPC request frame exists for it (2026-07-28 MRTR flow).
    """
    result = InputRequiredResult(input_requests={"r1": ListRootsRequest()}, request_state="s1")
    frame = JSONRPCResponse(jsonrpc="2.0", id=4, result=_body(result))
    assert _frame(frame) == snapshot(
        '{"jsonrpc":"2.0","id":4,"result":{"resultType":"input_required","inputRequests":{"r1":{"method":"roots/list"}},"requestState":"s1"}}'
    )


def test_error_frame_wraps_error_data_in_the_jsonrpc_envelope():
    """An error frame carries the ErrorData object under the error key; unset data never appears."""
    frame = JSONRPCError(jsonrpc="2.0", id=5, error=ErrorData(code=METHOD_NOT_FOUND, message="Method not found"))
    assert _frame(frame) == snapshot('{"jsonrpc":"2.0","id":5,"error":{"code":-32601,"message":"Method not found"}}')
