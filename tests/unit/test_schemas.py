"""The schemas are the contract, so the contract is executed here.

A schema nothing loads is decoration. The failure it permits is specific: an object crosses a
boundary carrying fields its schema forbids, or missing fields its schema requires, and every
layer downstream accepts it because no layer ever checked.
"""

from __future__ import annotations

import json
import pathlib

import pytest
from pydantic import ValidationError

from protocol.models import (
    BundleManifest,
    Challenge,
    ExecutionReceipt,
    JudgeResult,
    ModelManifest,
    ResearchPortfolio,
    SeasonConfig,
)

SCHEMA_DIR = pathlib.Path("protocol/schemas")
ROOTS = {
    "bundle_manifest": BundleManifest,
    "challenge": Challenge,
    "execution_receipt": ExecutionReceipt,
    "judge_result": JudgeResult,
    "model_manifest": ModelManifest,
    "portfolio": ResearchPortfolio,
    "season_config": SeasonConfig,
}


def challenge_body(**over) -> dict:
    body = {
        "challenge_id": "sha256:" + "a" * 64,
        "domain": "ai_agent_architecture",
        "title": "Resource-bounded persistent planning",
        "problem_statement": "x" * 250,
        "research_objective": "y" * 60,
        "constraints": ["must run within a fixed memory budget"],
        "required_output": {
            "portfolio_size": 5,
            "ranked": True,
            "mechanism_required": True,
            "prior_art_comparison_required": True,
            "falsification_plan_required": True,
            "simulation_or_calculation_required": True,
        },
        "resource_limits": {
            "maximum_wall_time_seconds": 1800,
            "maximum_rcc": 400,
            "maximum_search_calls": 100,
        },
        "generator_family": "gpt",
    }
    body.update(over)
    return body


# --------------------------------------------------------------------------
# Every schema has a model, and every model refuses unknown fields
# --------------------------------------------------------------------------


def test_every_shipped_schema_has_a_generated_root_model():
    """A schema with no model is a schema nothing can validate against."""
    shipped = {p.name.removesuffix(".json") for p in SCHEMA_DIR.glob("*.json")}
    assert shipped == set(ROOTS)


@pytest.mark.parametrize("name", sorted(ROOTS))
def test_every_top_level_object_is_closed(name):
    """`additionalProperties: false` throughout.

    A schema that permitted unknown fields would validate the exact defect this layer exists
    to catch.
    """
    document = json.loads((SCHEMA_DIR / f"{name}.json").read_text())
    assert document.get("additionalProperties") is False


@pytest.mark.parametrize("name,model", sorted(ROOTS.items()))
def test_the_generated_model_forbids_extras(name, model):
    """The schema's `additionalProperties: false` must survive into the model.

    If it did not, the whole generation step would be decorative.
    """
    with pytest.raises(ValidationError) as raised:
        model.model_validate({"a_field_no_schema_declares": 1})
    assert any(error["type"] == "extra_forbidden" for error in raised.value.errors())


# --------------------------------------------------------------------------
# The shipped reference config validates
# --------------------------------------------------------------------------


def test_the_reference_season_config_validates():
    """The document an operator actually edits must pass its own schema."""
    config = json.loads(pathlib.Path("config/season.example.json").read_text())
    SeasonConfig.model_validate(config)


def test_the_reference_config_carries_no_floats():
    """It is hashed and anchored on chain; a float makes the anchor unreproducible."""
    from protocol.canonical import assert_no_floats

    assert_no_floats(json.loads(pathlib.Path("config/season.example.json").read_text()))


def test_a_misspelled_season_key_is_refused():
    """A misspelled knob configures nothing, silently, for a whole season.

    And the season is signed, so the typo is anchored on chain.
    """
    config = json.loads(pathlib.Path("config/season.example.json").read_text())
    config["scoring"]["mechanism_floor_pmm"] = 400_000  # transposed
    with pytest.raises(ValidationError) as raised:
        SeasonConfig.model_validate(config)
    assert any("mechanism_floor_pmm" in str(error["loc"]) for error in raised.value.errors())


def test_a_decimal_ratio_is_refused_by_the_type():
    """Every ratio is a ppm integer, enforced by the schema rather than by convention."""
    config = json.loads(pathlib.Path("config/season.example.json").read_text())
    config["scoring"]["mechanism_floor_ppm"] = 0.4
    with pytest.raises(ValidationError):
        SeasonConfig.model_validate(config)


# --------------------------------------------------------------------------
# The V1 domain restriction is enforced by type, not by review
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "excluded",
    ["wet_lab_chemistry", "clinical_treatment", "weapons", "legal_policy", "artistic_ideation"],
)
def test_a_v1_excluded_domain_cannot_be_expressed(excluded):
    """architecture.md 2 excludes these from V1 scoring.

    A closed enum means such a challenge cannot be constructed at all — the exclusion is a
    type error rather than a policy someone has to check.
    """
    with pytest.raises(ValidationError) as raised:
        Challenge.model_validate(challenge_body(domain=excluded))
    assert any(error["type"] == "enum" for error in raised.value.errors())


@pytest.mark.parametrize(
    "domain",
    [
        "software_architecture", "algorithms", "ai_agent_architecture", "memory_and_retrieval",
        "distributed_coordination", "digital_mechanism_design", "data_and_model_pipelines",
        "optimization_and_efficiency",
    ],
)
def test_every_v1_domain_is_accepted(domain):
    Challenge.model_validate(challenge_body(domain=domain))


# --------------------------------------------------------------------------
# What the linter's rules look like at the type level
# --------------------------------------------------------------------------


def test_a_challenge_with_no_constraints_is_refused():
    """Without constraints there is nothing for a mechanism to satisfy.

    architecture.md 7.4 step 2 makes the linter reject it; the schema makes it unconstructible.
    """
    with pytest.raises(ValidationError):
        Challenge.model_validate(challenge_body(constraints=[]))


def test_a_trivially_short_problem_statement_is_refused():
    with pytest.raises(ValidationError):
        Challenge.model_validate(challenge_body(problem_statement="too short"))


def test_a_challenge_id_must_be_a_digest():
    with pytest.raises(ValidationError):
        Challenge.model_validate(challenge_body(challenge_id="not-a-digest"))


# --------------------------------------------------------------------------
# Model manifest: one provider surface, pinned revisions
# --------------------------------------------------------------------------


def test_a_manifest_declaring_a_provider_other_than_openrouter_is_refused():
    """One surface, enforced. A route the gateway has no adapter for cannot be named."""
    with pytest.raises(ValidationError):
        ModelManifest.model_validate(
            {
                "models": [
                    {"alias": "a", "provider": "anthropic", "model_slug": "x", "role": "critique"}
                ],
                "routing_config_hash": "sha256:" + "a" * 64,
            }
        )


def test_an_abbreviated_revision_is_refused():
    """Abbreviations become ambiguous as a repository grows.

    Pinning exists precisely so the artifact cannot move.
    """
    with pytest.raises(ValidationError):
        ModelManifest.model_validate(
            {
                "models": [
                    {
                        "alias": "house",
                        "provider": "openrouter",
                        "model_slug": "org/model",
                        "role": "idea_generation",
                        "revision": "abc1234",
                    }
                ],
                "routing_config_hash": "sha256:" + "a" * 64,
            }
        )


def test_a_full_forty_character_revision_is_accepted():
    ModelManifest.model_validate(
        {
            "models": [
                {
                    "alias": "house",
                    "provider": "openrouter",
                    "model_slug": "org/model",
                    "role": "idea_generation",
                    "revision": "a" * 40,
                }
            ],
            "routing_config_hash": "sha256:" + "a" * 64,
        }
    )


# --------------------------------------------------------------------------
# The bundle manifest commits to its credential envelope without containing it
# --------------------------------------------------------------------------


def test_the_bundle_manifest_requires_a_credential_envelope_digest():
    """It commits to which envelope belongs to it, and holds none of its contents.

    Section 6.3 publishes the manifest; a credential inside it would be published too.
    """
    assert "credential_envelope_digest" in BundleManifest.model_fields
    fields = set(BundleManifest.model_fields)
    for leaking in ("credential", "api_key", "key_capsule", "openrouter_key"):
        assert leaking not in fields, f"{leaking} would be published by section 6.3"


def test_the_output_schema_is_pinned_to_the_portfolio_version():
    """A bundle declaring another output schema would produce something unscoreable."""
    with pytest.raises(ValidationError):
        BundleManifest.model_validate({"output_schema": "something_else"})


# --------------------------------------------------------------------------
# Judge results: abstain is a first-class outcome
# --------------------------------------------------------------------------


def test_abstain_is_required_rather_than_optional():
    """A judge forced to choose when it cannot discriminate contributes noise as signal."""
    assert "abstain" in JudgeResult.model_fields
    assert JudgeResult.model_fields["abstain"].is_required()


def test_a_judge_result_names_only_blinded_candidates():
    """Section 14 strips identity before this stage; the type carries no miner field."""
    JudgeResult.model_validate(
        {
            "criterion": "mechanism",
            "comparison": {"candidate_a": "anonymous-A", "candidate_b": "anonymous-B"},
            "winner": "A",
            "abstain": False,
        }
    )
    fields = set(JudgeResult.model_fields)
    for identifying in ("miner_hotkey", "miner_uid", "bundle_id"):
        assert identifying not in fields


def test_a_tie_is_expressible():
    """Collapsing a tie into a winner would invent a preference the judge did not have."""
    JudgeResult.model_validate(
        {
            "criterion": "value",
            "comparison": {"candidate_a": "A", "candidate_b": "B"},
            "winner": "tie",
            "abstain": False,
        }
    )


# --------------------------------------------------------------------------
# Receipts: the credential owner is part of the record
# --------------------------------------------------------------------------


def test_a_receipt_must_declare_who_paid_and_for_what():
    """With one provider surface these are the only things separating the two accounts."""
    for required in ("credential_owner", "purpose"):
        assert ExecutionReceipt.model_fields[required].is_required()
