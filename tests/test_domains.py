"""Contract tests for the domain extension layer.

Two kinds of test live here. The first kind checks the contract itself: version arithmetic, and
the guards in `declare()` and the policy dataclasses that are supposed to reject a malformed
domain at import time. The second kind checks this repo's own domains against `domains/
catalog.json`, so a domain cannot be added, renamed or removed without the index noticing.

The test that matters most is `test_core_imports_no_domain`. The whole architecture rests on the
dependency arrow pointing one way, and that property is invisible at runtime until the day
someone adds a convenience import to the engine and quietly teaches it one field's rules.
"""

from __future__ import annotations

import ast
import json
import unittest
from pathlib import Path

from research_pipeline import domains
from research_pipeline.domains import (
    CritiqueRule,
    DomainPolicy,
    Measure,
    StudyDesign,
    Taxonomy,
)

ROOT = Path(__file__).resolve().parent.parent
DOMAINS_DIR = ROOT / "domains"
CATALOG = json.loads((DOMAINS_DIR / "catalog.json").read_text())


def minimal_taxonomy() -> Taxonomy:
    return Taxonomy(
        (
            StudyDesign(
                "primary_human",
                "Human study",
                "new data from human participants",
                primary=True,
                rank=70,
                human=True,
            ),
            StudyDesign("unclear", "Unclear", "the text does not say"),
        )
    )


def minimal_policy(**changes) -> DomainPolicy:
    base = {
        "slug": "example",
        "label": "Example",
        "scope": "An example field.",
        "taxonomy": minimal_taxonomy(),
    }
    return DomainPolicy(**{**base, **changes})


def _resolve(node: ast.expr | None, tree: ast.Module):
    """Evaluate a literal keyword argument, following one level of module-level constant.

    A domain may pass `extends={...}` inline or hoist it to an `EXTENDS = {...}` constant. Both
    are legitimate style, so the catalogue check reads either rather than dictating one.
    """
    if node is None:
        return None
    if isinstance(node, ast.Name):
        assignment = next(
            (
                n
                for n in tree.body
                if isinstance(n, ast.Assign)
                and any(getattr(t, "id", "") == node.id for t in n.targets)
            ),
            None,
        )
        if assignment is None:
            raise AssertionError(f"cannot resolve {node.id} to a module-level constant")
        node = assignment.value
    return ast.literal_eval(node)


class VersionArithmetic(unittest.TestCase):
    def test_parses_dotted_integers(self):
        self.assertEqual(domains.parse_version("1.2.3"), (1, 2, 3))
        self.assertEqual(domains.parse_version("0.1"), (0, 1))

    def test_rejects_non_numeric(self):
        for bad in ("1.2.3a", "v1", "", "1..2"):
            with self.assertRaises(ValueError):
                domains.parse_version(bad)

    def test_compares_across_lengths(self):
        # 0.1 and 0.1.0 are the same version; padding must not make one of them larger.
        self.assertTrue(domains.satisfies("0.1", ">=0.1.0"))
        self.assertTrue(domains.satisfies("0.1.0", ">=0.1"))

    def test_range_bounds(self):
        self.assertTrue(domains.satisfies("0.1.5", ">=0.1.0,<0.2"))
        self.assertFalse(domains.satisfies("0.2.0", ">=0.1.0,<0.2"))
        self.assertFalse(domains.satisfies("0.0.9", ">=0.1.0,<0.2"))

    def test_specifier_needs_an_operator(self):
        with self.assertRaises(ValueError):
            domains.satisfies("1.0", "1.0")


class PolicyGuards(unittest.TestCase):
    def test_the_search_half_is_a_library_profile_named_by_the_policy(self):
        # Anchors, sources and search guidance live in the paper library's discipline profile;
        # a policy names it, by default under its own slug.
        policy = DomainPolicy("example", "Example", "A scope.", minimal_taxonomy())
        self.assertEqual(policy.profile, "example")
        self.assertEqual(domains.GENERIC.profile, "")
        self.assertEqual(policy.summary()["profile"], "example")
        for gone in ("anchors", "sources", "search_guidance"):
            self.assertFalse(hasattr(policy, gone), gone)

    def test_taxonomy_requires_unclear(self):
        with self.assertRaises(ValueError):
            Taxonomy((StudyDesign("primary_human", "Human", "people"),))

    def test_taxonomy_rejects_duplicate_keys(self):
        with self.assertRaises(ValueError):
            Taxonomy(
                (
                    StudyDesign("a", "A", "one"),
                    StudyDesign("a", "A again", "two"),
                    StudyDesign("unclear", "Unclear", "the text does not say"),
                )
            )

    def test_design_key_is_snake_case(self):
        with self.assertRaises(ValueError):
            StudyDesign("Primary-Human", "Human", "people")

    def test_design_needs_a_definition(self):
        # The definition is read by a small model inside the classifier prompt; an empty one
        # silently degrades classification rather than failing.
        with self.assertRaises(ValueError):
            StudyDesign("primary_human", "Human", "  ")

    def test_measure_without_units_is_rejected(self):
        with self.assertRaises(ValueError):
            Measure("empty", "Nothing", ())

    def test_policy_slug_is_kebab_case(self):
        with self.assertRaises(ValueError):
            minimal_policy(slug="Not Kebab")

    def test_policy_needs_a_scope(self):
        with self.assertRaises(ValueError):
            minimal_policy(scope="   ")

    def test_critique_rule_must_name_known_designs(self):
        with self.assertRaises(ValueError):
            minimal_policy(
                critique_rules=(CritiqueRule("bogus", frozenset({"no_such_design"}), "problem"),)
            )

    def test_min_primary_must_be_positive(self):
        with self.assertRaises(ValueError):
            minimal_policy(min_primary_for_strong=0)


class CritiqueRuleTriggering(unittest.TestCase):
    rule = CritiqueRule(
        "preclinical_only",
        frozenset({"primary_animal", "in_vitro"}),
        "animal or in-vitro evidence only.",
    )

    def test_fires_when_all_designs_fall_inside(self):
        self.assertTrue(self.rule.triggers({"primary_animal"}))
        self.assertTrue(self.rule.triggers({"primary_animal", "in_vitro"}))

    def test_silent_when_any_design_falls_outside(self):
        self.assertFalse(self.rule.triggers({"primary_animal", "primary_human"}))

    def test_silent_on_no_evidence(self):
        # No evidence is a different problem, reported elsewhere; this rule must not claim it.
        self.assertFalse(self.rule.triggers(set()))


class MeasureLookup(unittest.TestCase):
    policy = minimal_policy(
        measures=(
            Measure("time", "Intervals", ("ms", "s")),
            Measure("pressure", "Pressures", ("mmHg",)),
        )
    )

    def test_unit_lookup_is_case_insensitive(self):
        measure = self.policy.measure_for_unit("MMHG")
        assert measure is not None
        self.assertEqual(measure.key, "pressure")

    def test_unknown_unit_returns_none(self):
        self.assertIsNone(self.policy.measure_for_unit("parsec"))

    def test_unit_words_collects_every_unit(self):
        self.assertEqual(self.policy.unit_words(), {"ms", "s", "mmhg"})


class Declaration(unittest.TestCase):
    def kwargs(self, **changes):
        base = {
            "name": "rag-domain-example",
            "module": "tests.test_domains",
            "version": "0.1.0",
            "core_requires": ">=0.1.0",
            "job": "Do one thing.",
            "policy": minimal_policy(),
        }
        return {**base, **changes}

    def test_name_must_carry_the_prefix(self):
        with self.assertRaises(ValueError):
            domains.declare(**self.kwargs(name="example"))

    def test_job_is_required(self):
        with self.assertRaises(ValueError):
            domains.declare(**self.kwargs(job="  "))

    def test_version_must_be_dotted_integers(self):
        with self.assertRaises(ValueError):
            domains.declare(**self.kwargs(version="0.1.0-beta"))

    def test_incompatible_core_is_refused(self):
        with self.assertRaises(domains.IncompatibleCore):
            domains.declare(**self.kwargs(core_requires=">=99.0"))

    def test_extending_a_missing_domain_is_refused(self):
        with self.assertRaises(domains.MissingDomain):
            domains.declare(**self.kwargs(extends={"rag-domain-does-not-exist": ">=0.1.0"}))


class Resolution(unittest.TestCase):
    def test_no_domain_means_generic_not_failure(self):
        for ref in (None, "", "generic"):
            self.assertEqual(domains.active(ref).policy.slug, "generic")

    def test_unknown_domain_raises(self):
        with self.assertRaises(domains.UnknownDomain):
            domains.active("astrology")

    def test_generic_reproduces_the_pre_domain_study_types(self):
        # schema.STUDY_TYPES is derived from this; the original list must survive verbatim.
        self.assertEqual(
            sorted(domains.GENERIC.taxonomy.keys),
            sorted(
                [
                    "primary_human",
                    "primary_animal",
                    "in_vitro",
                    "computational_model",
                    "meta_analysis",
                    "review",
                    "methods",
                    "unclear",
                ]
            ),
        )

    def test_derive_changes_only_what_it_is_given(self):
        derived = domains.derive(domains.GENERIC, slug="derived", min_primary_for_strong=4)
        self.assertEqual(derived.slug, "derived")
        self.assertEqual(derived.min_primary_for_strong, 4)
        self.assertEqual(derived.taxonomy, domains.GENERIC.taxonomy)


class Catalogue(unittest.TestCase):
    """The in-repo index must match the directories and what each package declares."""

    def test_every_domain_directory_is_catalogued(self):
        on_disk = {p.name for p in DOMAINS_DIR.iterdir() if p.is_dir()}
        listed = {entry["slug"] for entry in CATALOG["domains"]}
        self.assertEqual(on_disk, listed)

    def test_catalogue_paths_and_packages_exist(self):
        for entry in CATALOG["domains"]:
            package = ROOT / entry["path"] / entry["module"] / "__init__.py"
            self.assertTrue(package.exists(), f"{entry['slug']}: missing {package}")

    def test_names_follow_the_prefix(self):
        for entry in CATALOG["domains"]:
            self.assertTrue(entry["name"].startswith(domains.NAME_PREFIX), entry["name"])
            self.assertEqual(entry["name"], domains.NAME_PREFIX + entry["slug"])

    def test_pyproject_registers_the_entry_point(self):
        for entry in CATALOG["domains"]:
            text = (ROOT / entry["path"] / "pyproject.toml").read_text()
            self.assertIn(
                f'[project.entry-points."{domains.ENTRY_POINT_GROUP}"]',
                text,
                f"{entry['slug']}: entry point group missing",
            )
            self.assertIn(f'name = "{entry["name"]}"', text)
            self.assertIn(f'{entry["slug"]} = "{entry["module"]}"', text)

    def test_declared_metadata_matches_the_catalogue(self):
        # Reads the declare() call from source rather than importing, so this test still runs
        # when the domain packages are not installed in the environment.
        for entry in CATALOG["domains"]:
            tree = ast.parse((ROOT / entry["path"] / entry["module"] / "__init__.py").read_text())
            call = next(
                (
                    node
                    for node in ast.walk(tree)
                    if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "declare"
                ),
                None,
            )
            assert call is not None, f"{entry['slug']}: no declare() call"
            kwargs = {kw.arg: kw.value for kw in call.keywords}
            self.assertEqual(ast.literal_eval(kwargs["name"]), entry["name"])
            self.assertEqual(
                ast.literal_eval(kwargs["job"]),
                entry["job"],
                f"{entry['slug']}: job drifted from the catalogue",
            )
            extends = _resolve(kwargs.get("extends"), tree) or {}
            self.assertEqual(
                sorted(extends),
                sorted(entry["extends"]),
                f"{entry['slug']}: extends drifted from the catalogue",
            )

    def test_every_domain_has_a_profile_in_the_paper_library(self):
        """A domain's search half is paper-fetch's profile of the same slug; without it a run
        would plan its queries without the field's indexed terms."""
        from paper_fetch.profiles import load_profiles

        known = load_profiles()
        for entry in CATALOG["domains"]:
            self.assertIn(entry["slug"], known, entry["slug"])


class DependencyArrow(unittest.TestCase):
    def test_core_imports_no_domain(self):
        """The engine must never import a domain package. This is the whole architecture."""
        offenders: list[str] = []
        for path in (ROOT / "research_pipeline").rglob("*.py"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                else:
                    continue
                for name in names:
                    if name.startswith("rag_"):
                        offenders.append(f"{path.relative_to(ROOT)} imports {name}")
        self.assertEqual(offenders, [])

    def test_domains_import_only_the_contract(self):
        """A domain may import the contract module and its declared siblings, nothing else."""
        by_name = {entry["name"]: entry["module"] for entry in CATALOG["domains"]}
        offenders: list[str] = []
        for entry in CATALOG["domains"]:
            allowed = {"research_pipeline.domains"}
            allowed |= {by_name[name] for name in entry["extends"]}
            path = ROOT / entry["path"] / entry["module"] / "__init__.py"
            for node in ast.walk(ast.parse(path.read_text())):
                if not isinstance(node, ast.ImportFrom) or node.level:
                    continue
                module = node.module or ""
                if module.startswith("research_pipeline") and module not in allowed:
                    offenders.append(f"{entry['slug']} imports engine internals: {module}")
                if module.startswith("rag_") and module not in allowed:
                    offenders.append(f"{entry['slug']} imports undeclared sibling: {module}")
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
