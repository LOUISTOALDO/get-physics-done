"""Prompt/schema visibility regression for approved-mode project-contract grounding."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
NEW_PROJECT = REPO_ROOT / "src" / "gpd" / "specs" / "workflows" / "new-project.md"
STATE_SCHEMA = REPO_ROOT / "src" / "gpd" / "specs" / "templates" / "state-json-schema.md"


def test_new_project_prompt_surfaces_the_canonical_state_schema_for_project_contract_grounding() -> None:
    new_project_text = NEW_PROJECT.read_text(encoding="utf-8")

    assert "templates/state-json-schema.md" in new_project_text
    assert "Before you ask for approval, build the raw contract as a literal JSON object that follows `templates/state-json-schema.md` exactly:" in new_project_text
    assert "Do not approve a scoping contract that strips decisive outputs, anchors, prior outputs, or review/stop triggers down to generic placeholders." in new_project_text


def test_state_schema_surfaces_the_exact_approved_mode_grounding_rule() -> None:
    state_schema_text = STATE_SCHEMA.read_text(encoding="utf-8")

    assert (
        "approved project contract requires at least one concrete anchor/reference/prior-output/baseline or an "
        "explicit 'anchor unknown' blocker"
        in state_schema_text
    )
    assert "`must_include_prior_outputs[]` entries should be explicit project-artifact paths or filenames" in state_schema_text
    assert (
        "`user_asserted_anchors[]` and `known_good_baselines[]` should name a concrete benchmark, baseline, reference, notebook, figure, dataset, or comparable anchor phrase"
        in state_schema_text
    )
    assert "Placeholder or `TBD` text does not count as concrete grounding." in state_schema_text
    assert "placeholder-only wording does not satisfy approved-mode grounding" in state_schema_text
