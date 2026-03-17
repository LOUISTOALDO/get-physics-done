from __future__ import annotations

import json
from pathlib import Path

import anyio


def _tool_description(mcp_server: object, tool_name: str) -> str:
    async def _load() -> str:
        tools = await mcp_server.list_tools()
        return next(tool.description for tool in tools if tool.name == tool_name)

    return anyio.run(_load)


def _tool_input_schema(mcp_server: object, tool_name: str) -> dict[str, object]:
    async def _load() -> dict[str, object]:
        tools = await mcp_server.list_tools()
        return next(tool.inputSchema for tool in tools if tool.name == tool_name)

    return anyio.run(_load)


def _schema_ref(schema_fragment: dict[str, object]) -> str:
    if "$ref" in schema_fragment:
        return str(schema_fragment["$ref"])
    for branch in schema_fragment.get("anyOf", []):
        if isinstance(branch, dict) and "$ref" in branch:
            return str(branch["$ref"])
    raise AssertionError(f"No schema reference found in {schema_fragment!r}")


def _schema_object(schema: dict[str, object], schema_fragment: dict[str, object]) -> dict[str, object]:
    if "properties" in schema_fragment:
        return schema_fragment
    ref = _schema_ref(schema_fragment)
    target: object = schema
    for segment in ref.removeprefix("#/").split("/"):
        if not isinstance(target, dict):
            raise AssertionError(f"Schema pointer {ref} resolved to non-object {target!r}")
        target = target[segment]
    if not isinstance(target, dict):
        raise AssertionError(f"Schema pointer {ref} resolved to non-object {target!r}")
    return target


def _schema_anyof_object(schema_fragment: dict[str, object]) -> dict[str, object]:
    if schema_fragment.get("type") == "object":
        return schema_fragment
    for branch in schema_fragment.get("anyOf", []):
        if isinstance(branch, dict) and branch.get("type") == "object":
            return branch
    raise AssertionError(f"No object branch found in {schema_fragment!r}")


def test_run_contract_check_tool_description_surfaces_request_requirements() -> None:
    from gpd.mcp.servers.verification_server import mcp

    description = _tool_description(mcp, "run_contract_check")

    assert "``request.check_key`` or ``request.check_id`` is required" in description
    assert "``request.contract`` is optional" in description
    assert "``schema_version: 1``" in description
    assert "``request.binding``, ``request.metadata``, and ``request.observed`` are each" in description
    assert "``request.artifact_content``" in description
    assert "must be a string when present" in description
    assert "``required_request_fields``" in description
    assert "``optional_request_fields``" in description
    assert "``supported_binding_fields``" in description
    assert "``request_template``" in description
    assert "``references[].carry_forward_to`` only for workflow scope labels" in description
    assert "make resolution ambiguous" in description


def test_suggest_contract_checks_tool_description_surfaces_contract_requirements() -> None:
    from gpd.mcp.servers.verification_server import mcp

    description = _tool_description(mcp, "suggest_contract_checks")

    assert "``contract`` must be an object with ``schema_version: 1``" in description
    assert "``active_checks`` is optional and must be ``list[str]``" in description
    assert "``already_active``" in description
    assert "``supported_binding_fields``" in description
    assert "``references[].carry_forward_to`` only for workflow scope labels" in description
    assert "``run_contract_check(request=...)``" in description


def test_contract_tools_list_tools_expose_structured_request_schemas() -> None:
    from gpd.mcp.servers.verification_server import mcp

    run_schema = _tool_input_schema(mcp, "run_contract_check")
    run_request = _schema_object(run_schema, run_schema["properties"]["request"])

    assert {"check_key", "check_id", "contract", "binding", "metadata", "observed", "artifact_content"} <= set(
        run_request["properties"]
    )

    binding = _schema_anyof_object(run_request["properties"]["binding"])
    assert {"claim_ids", "reference_ids", "forbidden_proxy_ids"} <= set(binding["properties"])

    metadata = _schema_anyof_object(run_request["properties"]["metadata"])
    assert {"source_reference_id", "allowed_families", "forbidden_families"} <= set(metadata["properties"])

    observed = _schema_anyof_object(run_request["properties"]["observed"])
    assert {"metric_value", "threshold_value", "selected_family", "bias_checked"} <= set(observed["properties"])

    contract_schema = _schema_anyof_object(run_request["properties"]["contract"])
    assert {"schema_version", "scope", "claims", "references"} <= set(contract_schema["properties"])

    suggest_schema = _tool_input_schema(mcp, "suggest_contract_checks")
    contract_schema = _schema_anyof_object(suggest_schema["properties"]["contract"])
    assert {"schema_version", "scope", "claims", "references"} <= set(contract_schema["properties"])
    active_checks = suggest_schema["properties"]["active_checks"]
    assert active_checks["anyOf"][0]["type"] == "array"
    assert active_checks["anyOf"][0]["items"]["type"] == "string"


def test_assert_convention_validate_description_surfaces_required_headers() -> None:
    from gpd.mcp.servers.conventions_server import mcp

    description = _tool_description(mcp, "assert_convention_validate")

    assert "Every derivation artifact must include at least one ASSERT_CONVENTION line." in description
    assert "Missing assertions are treated as invalid, not advisory" in description


def test_public_descriptors_surface_contract_and_optional_dependency_visibility() -> None:
    from gpd.mcp.builtin_servers import build_public_descriptors

    descriptors = build_public_descriptors()

    verification = descriptors["gpd-verification"]
    assert "structured request objects or schema_version=1 contract payloads" in verification["description"]
    assert "required_request_fields" in verification["description"]
    assert "optional_request_fields" in verification["description"]
    assert "request_template" in verification["description"]
    assert "supported binding fields" in verification["description"]
    for field in (
        "binding.observable_id(s)",
        "binding.claim_id(s)",
        "binding.deliverable_id(s)",
        "binding.acceptance_test_id(s)",
        "binding.reference_id(s)",
        "binding.forbidden_proxy_id(s)",
    ):
        assert field in verification["description"]
    assert "live semantic integrity rules" in verification["description"]
    assert "target resolution ambiguous" in verification["description"]
    assert "`references[].carry_forward_to` entries as workflow scope labels only" in verification["description"]
    assert "never contract IDs" in verification["description"]

    conventions = descriptors["gpd-conventions"]
    assert "ASSERT_CONVENTION validation" in conventions["description"]
    assert "Every derivation artifact must carry at least one ASSERT_CONVENTION header" in conventions["description"]

    arxiv = descriptors["gpd-arxiv"]
    assert arxiv["optional"] is True
    assert arxiv["availability"] == "conditional"
    assert "optional Python module 'arxiv_mcp_server'" in arxiv["availability_condition"]
    assert "Optional/conditional arXiv paper search and retrieval" in arxiv["description"]


def test_public_verification_infra_descriptor_surfaces_semantic_contract_rules() -> None:
    descriptor = json.loads(
        (Path(__file__).resolve().parents[2] / "infra" / "gpd-verification.json").read_text(encoding="utf-8")
    )

    description = descriptor["description"]
    assert "structured request objects or schema_version=1 contract payloads" in description
    assert "required_request_fields" in description
    assert "optional_request_fields" in description
    assert "request_template" in description
    assert "supported binding fields" in description
    for field in (
        "binding.observable_id(s)",
        "binding.claim_id(s)",
        "binding.deliverable_id(s)",
        "binding.acceptance_test_id(s)",
        "binding.reference_id(s)",
        "binding.forbidden_proxy_id(s)",
    ):
        assert field in description
    assert "live semantic integrity rules" in description
    assert "target resolution ambiguous" in description
    assert "`references[].carry_forward_to` entries as workflow scope labels only" in description
    assert "never contract IDs" in description
