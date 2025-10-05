# -----------------------------------------------------------------------------
# Catalog loader & accessor
# Purpose: Parse a hierarchical YAML catalog (domains → topics → formulas)
# into strongly-typed objects used by the solver pipeline.
# - Depends on .types (Formula, VariableSpec, Constant) for typed payloads.
# -----------------------------------------------------------------------------

from __future__ import annotations
import yaml
from dataclasses import dataclass
from typing import Dict, List, Any
from .types import Formula, VariableSpec, Constant

# Domain-specific error to signal malformed catalog inputs, missing fields, etc.
class CatalogError(Exception): pass

@dataclass
class Catalog:
    # Map of physical/mathematical constants keyed by string id (e.g., "g", "pi")
    constants: Dict[str, Constant]
    # Flattened list of all formulas across domains/topics
    formulas: List[Formula]

    @staticmethod
    def from_yaml_dict(d: Dict[str, Any]) -> "Catalog":
        """
        Build a Catalog from a pre-parsed YAML dictionary.
        Expected YAML high-level shape:
          constants:
            g: { value: 9.81, unit: "m/s^2", tags: ["gravity"], notes: "..." }
          domains:
            physics:
              topics:
                kinematics:
                  formulas:
                    - id: ...
                      name: ...
                      eq: "..."         # original equation text (pretty)
                      rhs: "..."        # canonical right-hand-side expression
                      variables:
                        v0: { unit: "m/s", aliases: ["u"], source: "...", ... }
                      tags: ["projectile", "horizontal"]
                      rearrangements: true|false
                      priority: 0.0..1.0
        """
        consts: Dict[str, Constant] = {}
        # ---- Parse constants --------------------------------------------------
        for k, c in (d.get("constants") or {}).items():
            consts[k] = Constant(key=k, value=float(c["value"]), unit=str(c["unit"]),
                                 tags=list(c.get("tags", [])), notes=str(c.get("notes","")))
        forms: List[Formula] = []
        # ---- Drill into domains → topics → formulas ---------------------------
        domains = d.get("domains", {})
        for dom_name, dom in domains.items():
            topics = (dom or {}).get("topics", {})
            for topic_name, topic in topics.items():
                for fd in (topic or {}).get("formulas", []):
                    # Collect variable-spec metadata for the formula
                    vars_dict = {}
                    for sym, vs in (fd.get("variables") or {}).items():
                        vars_dict[sym] = VariableSpec(
                            unit=str(vs["unit"]),                 # canonical unit for symbol
                            aliases=list(vs.get("aliases", [])),  # synonyms (e.g., "u" for v0)
                            source=vs.get("source"),              # provenance/reference (optional)
                            key=vs.get("key"),                    # optional normalized feature key
                            normalize=bool(vs.get("normalize", False)),  # apply normalization?
                        )
                    # Materialize Formula with topic_path context and solver hints
                    forms.append(Formula(
                        id=fd["id"], name=fd["name"], eq=fd["eq"], rhs=fd["rhs"],
                        variables=vars_dict, tags=list(fd.get("tags", [])),
                        topic_path=[dom_name, topic_name],               # [domain, topic]
                        rearrangements=bool(fd.get("rearrangements", True)),  # allow algebraic forms
                        priority=float(fd.get("priority", 0.5))          # selection weight/heuristic
                    ))
        return Catalog(constants=consts, formulas=forms)

    @staticmethod
    def from_yaml_text(text: str) -> "Catalog":
        """
        Convenience: parse raw YAML string into a Catalog.
        Uses yaml.safe_load for security (no arbitrary object constructors).
        """
        return Catalog.from_yaml_dict(yaml.safe_load(text))

    @staticmethod
    def from_file(path: str) -> "Catalog":
        """
        Convenience: open a YAML file from disk and parse into a Catalog.
        UTF-8 is enforced to avoid cross-platform encoding issues.
        """
        with open(path, "r", encoding="utf-8") as f:
            return Catalog.from_yaml_text(f.read())

    def list_formulas(self) -> List[Dict[str, Any]]:
        """
        Flattened, UI-friendly listing of formulas across the catalog.
        Good for demos/explorers (id, name, eq, rhs, domain/topic, tags, priority).
        """
        out = []
        for f in self.formulas:
            out.append({
                "id": f.id, "name": f.name, "eq": f.eq, "rhs": f.rhs,
                "domain": f.topic_path[0], "topic": f.topic_path[1],
                "tags": f.tags, "priority": f.priority
            })
        return out
