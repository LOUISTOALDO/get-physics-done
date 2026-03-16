from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "stage0"


def _load_project_contract_fixture() -> dict[str, object]:
    return json.loads((FIXTURES_DIR / "project_contract.json").read_text(encoding="utf-8"))


def _derived_template_contract() -> dict[str, object]:
    contract = copy.deepcopy(_load_project_contract_fixture())
    contract["observables"][0]["regime"] = "large-k"
    contract["approach_policy"] = {
        "allowed_fit_families": ["power_law"],
        "forbidden_fit_families": ["polynomial"],
        "allowed_estimator_families": ["bootstrap"],
        "forbidden_estimator_families": ["jackknife"],
    }
    contract["acceptance_tests"].extend(
        [
            {
                "id": "test-limit",
                "subject": "claim-benchmark",
                "kind": "limiting_case",
                "procedure": "Evaluate the large-k limit against the asymptotic target.",
                "pass_condition": "Recovers the contracted large-k scaling",
                "evidence_required": ["deliv-figure"],
                "automation": "automated",
            },
            {
                "id": "test-fit",
                "subject": "claim-benchmark",
                "kind": "other",
                "procedure": "Compare fit residuals across the approved ansatz families.",
                "pass_condition": "Selected fit stays inside the allowed family",
                "evidence_required": ["deliv-figure"],
                "automation": "hybrid",
            },
            {
                "id": "test-estimator",
                "subject": "claim-benchmark",
                "kind": "other",
                "procedure": "Bootstrap estimator diagnostics must resolve bias and variance.",
                "pass_condition": "Bootstrap estimator remains calibrated",
                "evidence_required": ["deliv-figure"],
                "automation": "hybrid",
            },
        ]
    )
    return contract


@pytest.mark.parametrize(
    ("schema_version", "expected_error"),
    [
        (2, "Invalid contract payload: schema_version must be 1"),
        ("1", "Invalid contract payload: schema_version must be the integer 1"),
        (True, "Invalid contract payload: schema_version must be the integer 1"),
    ],
)
def test_contract_tools_reject_invalid_schema_versions(schema_version: object, expected_error: str) -> None:
    from gpd.mcp.servers.verification_server import run_contract_check, suggest_contract_checks

    contract = _load_project_contract_fixture()
    contract["schema_version"] = schema_version
    contract["context_intake"] = "not-a-dict"

    run_result = run_contract_check(
        {
            "check_key": "contract.benchmark_reproduction",
            "contract": contract,
            "binding": {"claim_ids": ["claim-benchmark"]},
            "metadata": {"source_reference_id": "ref-benchmark"},
            "observed": {"metric_value": 0.01, "threshold_value": 0.02},
        }
    )

    suggest_result = suggest_contract_checks(contract)

    assert run_result == {"error": expected_error, "schema_version": 1}
    assert suggest_result == {"error": expected_error, "schema_version": 1}


def test_suggest_contract_checks_derives_request_templates_from_contract() -> None:
    from gpd.mcp.servers.verification_server import suggest_contract_checks

    result = suggest_contract_checks(_derived_template_contract())
    checks = {entry["check_key"]: entry for entry in result["suggested_checks"]}

    benchmark = checks["contract.benchmark_reproduction"]["request_template"]
    assert benchmark["binding"]["claim_ids"] == ["claim-benchmark"]
    assert benchmark["binding"]["acceptance_test_ids"] == ["test-benchmark"]
    assert benchmark["binding"]["reference_ids"] == ["ref-benchmark"]
    assert benchmark["metadata"]["source_reference_id"] == "ref-benchmark"

    limit = checks["contract.limit_recovery"]["request_template"]
    assert limit["binding"]["claim_ids"] == ["claim-benchmark"]
    assert limit["binding"]["acceptance_test_ids"] == ["test-limit"]
    assert limit["binding"]["observable_ids"] == ["obs-benchmark"]
    assert limit["metadata"]["regime_label"] == "large-k"
    assert limit["metadata"]["expected_behavior"] == "Recovers the contracted large-k scaling"

    direct_proxy = checks["contract.direct_proxy_consistency"]["request_template"]
    assert direct_proxy["binding"]["claim_ids"] == ["claim-benchmark"]
    assert direct_proxy["binding"]["forbidden_proxy_ids"] == ["fp-01"]

    fit = checks["contract.fit_family_mismatch"]["request_template"]
    assert fit["binding"]["claim_ids"] == ["claim-benchmark"]
    assert fit["binding"]["acceptance_test_ids"] == ["test-fit"]
    assert fit["binding"]["observable_ids"] == ["obs-benchmark"]
    assert fit["metadata"]["declared_family"] == "power_law"
    assert fit["metadata"]["allowed_families"] == ["power_law"]
    assert fit["metadata"]["forbidden_families"] == ["polynomial"]

    estimator = checks["contract.estimator_family_mismatch"]["request_template"]
    assert estimator["binding"]["claim_ids"] == ["claim-benchmark"]
    assert estimator["binding"]["acceptance_test_ids"] == ["test-estimator"]
    assert estimator["binding"]["observable_ids"] == ["obs-benchmark"]
    assert estimator["metadata"]["declared_family"] == "bootstrap"
    assert estimator["metadata"]["allowed_families"] == ["bootstrap"]
    assert estimator["metadata"]["forbidden_families"] == ["jackknife"]


def test_suggest_contract_checks_surfaces_salvage_warnings() -> None:
    from gpd.mcp.servers.verification_server import suggest_contract_checks

    contract = _load_project_contract_fixture()
    contract["context_intake"] = "not-a-dict"
    contract["references"][0]["notes"] = "legacy extra field"

    result = suggest_contract_checks(contract)

    assert "error" not in result
    assert any("salvaged before check suggestion" in warning for warning in result["contract_warnings"])
    assert any(entry["check_key"] == "contract.benchmark_reproduction" for entry in result["suggested_checks"])


@pytest.mark.parametrize("payload", ["not-a-dict", ["claim-benchmark"], 3])
def test_suggest_contract_checks_rejects_non_mapping_payloads(payload: object) -> None:
    from gpd.mcp.servers.verification_server import suggest_contract_checks

    result = suggest_contract_checks(payload)  # type: ignore[arg-type]

    assert result == {"error": "contract must be an object", "schema_version": 1}


@pytest.mark.parametrize(
    ("request_payload", "expected_error"),
    [
        (
            {
                "check_key": "contract.benchmark_reproduction",
                "binding": {"claim_ids": ["claim-benchmark", None]},
                "observed": {"metric_value": 0.01, "threshold_value": 0.02},
            },
            "binding.claim_ids[1] must be a non-empty string",
        ),
        (
            {
                "check_key": "contract.benchmark_reproduction",
                "binding": {"reference_ids": ["ref-benchmark", "   "]},
                "observed": {"metric_value": 0.01, "threshold_value": 0.02},
            },
            "binding.reference_ids[1] must be a non-empty string",
        ),
        (
            {
                "check_key": "contract.fit_family_mismatch",
                "metadata": {"allowed_families": ["power_law", 5]},
                "observed": {"selected_family": "power_law", "competing_family_checked": True},
            },
            "metadata.allowed_families[1] must be a non-empty string",
        ),
        (
            {
                "check_key": "contract.estimator_family_mismatch",
                "metadata": {"forbidden_families": ["", "jackknife"]},
                "observed": {
                    "selected_family": "bootstrap",
                    "bias_checked": True,
                    "calibration_checked": True,
                },
            },
            "metadata.forbidden_families[0] must be a non-empty string",
        ),
    ],
)
def test_run_contract_check_rejects_malformed_binding_and_metadata_list_members(
    request_payload: dict[str, object], expected_error: str
) -> None:
    from gpd.mcp.servers.verification_server import run_contract_check

    result = run_contract_check(request_payload)

    assert result == {"error": expected_error, "schema_version": 1}


def test_verification_server_success_responses_keep_stable_envelope_equality() -> None:
    from gpd.mcp.servers.verification_server import get_checklist, run_contract_check, suggest_contract_checks

    run_result = run_contract_check(
        {
            "check_key": "contract.benchmark_reproduction",
            "binding": {"claim_ids": ["claim-benchmark"], "reference_ids": ["ref-benchmark"]},
            "metadata": {"source_reference_id": "ref-benchmark"},
            "observed": {"metric_value": 0.01, "threshold_value": 0.02},
        }
    )
    run_expected = dict(run_result)
    run_expected.pop("schema_version")
    assert run_result == run_expected

    suggest_result = suggest_contract_checks(_derived_template_contract())
    suggest_expected = dict(suggest_result)
    suggest_expected.pop("schema_version")
    assert suggest_result == suggest_expected

    checklist_result = get_checklist("qft")
    checklist_expected = dict(checklist_result)
    checklist_expected.pop("schema_version")
    assert checklist_result == checklist_expected
