"""Focused regressions for CLAIMS.json schema visibility in review-reader prompts."""

from __future__ import annotations

from pathlib import Path

from gpd.adapters.install_utils import expand_at_includes
from gpd.mcp.paper.models import (
    ClaimIndex,
    ClaimRecord,
    ClaimType,
    ReviewConfidence,
    ReviewFinding,
    ReviewIssueSeverity,
    ReviewRecommendation,
    ReviewSupportStatus,
    StageReviewReport,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
AGENTS_DIR = REPO_ROOT / "src/gpd/agents"
REFERENCES_DIR = REPO_ROOT / "src/gpd/specs/references"


def _between(text: str, start: str, end: str) -> str:
    _, start_marker, tail = text.partition(start)
    assert start_marker, f"Missing marker: {start}"
    body, end_marker, _ = tail.partition(end)
    assert end_marker, f"Missing marker: {end}"
    return body


def _assert_schema_tokens_visible(text: str) -> None:
    for token in (*ClaimIndex.model_fields, *ClaimRecord.model_fields):
        assert f"`{token}`" in text or f'"{token}"' in text, f"Missing schema token: {token}"
    for claim_type in ClaimType:
        assert claim_type.value in text, f"Missing claim type: {claim_type.value}"


def _assert_stage_review_schema_tokens_visible(text: str) -> None:
    for token in (*StageReviewReport.model_fields, *ReviewFinding.model_fields):
        assert f"`{token}`" in text or f'"{token}"' in text, f"Missing schema token: {token}"
    for severity in ReviewIssueSeverity:
        assert severity.value in text, f"Missing severity: {severity.value}"
    for support_status in ReviewSupportStatus:
        assert support_status.value in text, f"Missing support status: {support_status.value}"
    for confidence in ReviewConfidence:
        assert confidence.value in text, f"Missing confidence: {confidence.value}"
    for recommendation in ReviewRecommendation:
        assert recommendation.value in text, f"Missing recommendation: {recommendation.value}"


def test_review_reader_prompt_surfaces_full_claim_index_schema() -> None:
    review_reader = (AGENTS_DIR / "gpd-review-reader.md").read_text(encoding="utf-8")
    claims_schema = _between(
        review_reader,
        "Required schema for `CLAIMS.json` (`ClaimIndex`):",
        "Required details for `STAGE-reader.json`:",
    )

    _assert_schema_tokens_visible(claims_schema)
    assert "do not omit them" in claims_schema


def test_peer_review_panel_reference_surfaces_stage1_claim_index_schema() -> None:
    panel = (REFERENCES_DIR / "publication" / "peer-review-panel.md").read_text(encoding="utf-8")
    claims_schema = _between(
        panel,
        "Stage 1 `CLAIMS.json` must follow this compact `ClaimIndex` shape:",
        "The final adjudicator JSON artifacts must follow these canonical schemas:",
    )

    _assert_schema_tokens_visible(claims_schema)
    assert "required `ClaimIndex` metadata" in claims_schema


def test_expanded_review_reader_prompt_keeps_claim_index_metadata_visible() -> None:
    expanded = expand_at_includes(
        (AGENTS_DIR / "gpd-review-reader.md").read_text(encoding="utf-8"),
        REPO_ROOT / "src/gpd/specs",
        "/runtime/",
    )

    assert "Peer Review Panel Protocol" in expanded
    assert '"manuscript_path": "paper/main.tex"' in expanded
    assert '"manuscript_sha256": "<sha256>"' in expanded
    assert '"supporting_artifacts": ["paper/figures/main-result.pdf"]' in expanded


def test_stage_review_agents_surface_compact_stage_review_schema() -> None:
    for agent_name, marker, stage_kind in (
        (
            "gpd-review-math.md",
            "Required schema for `STAGE-math.json` (`StageReviewReport`, mirroring the staged-review contract):",
            "math",
        ),
        (
            "gpd-review-physics.md",
            "Required schema for `STAGE-physics.json` (`StageReviewReport`, mirroring the staged-review contract):",
            "physics",
        ),
        (
            "gpd-review-significance.md",
            "Required schema for `STAGE-interestingness.json` (`StageReviewReport`, mirroring the staged-review contract):",
            "interestingness",
        ),
    ):
        expanded = expand_at_includes(
            (AGENTS_DIR / agent_name).read_text(encoding="utf-8"),
            REPO_ROOT / "src/gpd/specs",
            "/runtime/",
        )
        schema = _between(expanded, marker, "Required finding coverage:")
        _assert_stage_review_schema_tokens_visible(schema)
        assert f"`stage_id` and `stage_kind` must both be `{stage_kind}`" in schema
        assert "do not collapse them to prose or scalars" in schema
