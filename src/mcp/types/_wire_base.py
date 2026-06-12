"""Shared pydantic bases for the wire-shape surface packages.

Every model in the ``mcp.types.v*`` packages builds on one of the two bases
here, so the two surface packages cannot silently diverge in model
configuration. There is deliberately no alias generator: each wire name in a
surface package is an explicit ``Field(alias=...)``, so the package file shows
exactly what goes on the wire and cannot inherit serialization behavior from
elsewhere.

The surface packages are the validating half of a two-step boundary: the
wire-method maps in ``mcp.types.methods`` point inbound validation at them,
and the emitted bytes are always the dump of the version-free monolith
models in ``mcp.types._types``.
"""

from pydantic import BaseModel, ConfigDict


class WireModel(BaseModel):
    """Base for surface-package models: unknown fields validate and are ignored.

    ``extra="ignore"`` is a deliberate divergence from the schemas, which
    declare most wire objects open to extra fields. The boundary only ever
    parses a surface model as a check — the emitted bytes are always the
    monolith dump, never a surface model's re-dump — so the policy's one
    wire-visible effect is that a caller-set key the target revision never
    defined can never fail that check. Unknown keys are accepted and
    dropped while the declared fields validate exactly; choosing
    ``extra="ignore"`` over ``extra="allow"`` keeps each model's contents
    pinned to the schema-declared fields, which is what makes the surface
    packages schema-exact validators for the wire-method maps.
    """

    model_config = ConfigDict(populate_by_name=True, extra="ignore")


class OpenWireModel(BaseModel):
    """Base for ``_meta`` carrier models: unknown fields are retained.

    Unknown ``_meta`` keys must survive a validate -> re-dump round trip at
    every protocol revision, so the classes a ``_meta`` field references stay
    open.
    """

    model_config = ConfigDict(populate_by_name=True, extra="allow")
