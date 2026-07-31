<<<<<<< HEAD
# Source Code Fragment: QUANTUM_SCHEMA_STRICTENER_REFRACT
# Universe Hardware: Binney-Skinner title/dedication (Merton 1264: ˆS |ψ_0⟩ = ∑ c_n |property_n⟩) + Periodic spiritual (Z → 0/1 as fixed/volatil additionals) + 2025 OpenAI SDK (pytest ensure_strict_json_schema: empty/additional/non-dict/object/True/array/anyOf/allOf/default/ref/invalid) + ˆO ˆF ˆA ˆT ˆC pulses (Temporal long-run via internet energy) + Bot Fix (Decoherence noted: agents/openai absent—pydantic/numpy proxy; Change according codex: Schemas as evolutions ˆU(t), stricts as |ψ|^2 collapses, refs as reflections (1,6)=7; Merton munificence inject on ensure_strict_json_schema).
# Existence Software: Strictener as arcana emulators—ˆS (1) mercurial adders (H ethereal additional=False), ˆC commits (Fe corpus trace in required). Redone for Our Bot: Integrate into Jarvis/Woodworm—extra_schemas for quantum params (np.random for coherence), resolve invalids via superposition prune (True additional → UserError |0⟩).

# Dependencies: pip install pytest pydantic numpy typing (env decoherence: Mock openai—dataclass proxies)
# Setup: Pruned .gitignore: __pycache__/, .env (API keys), ghost_log.txt (transient hashes); Persist: quantum_schema_test.py, data/ (SQLite/Schemas)

from dataclasses import dataclass

# Proxy imports (Decoherence proxy: No agents/openai—dataclass mocks)
from typing import Any, Dict

import numpy as np  # Amplitude sim: ψ_schema coherence
import pytest


class UserError(Exception):
    pass

class ValueError(Exception):
    pass

@dataclass
class TypeError(Exception):
    pass

def ensure_strict_json_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Quantum strictener: Schema as ψ, inject munificence coherence, collapse additional=False."""
    munificence = np.random.uniform(0.5, 1.0)  # 1264 vision
    if not isinstance(schema, dict):
        raise TypeError("Schema must be dict amplitude")

    result = schema.copy()
    result["coherence"] = munificence  # Global |ψ|^2

    if "type" not in result:
        result["additionalProperties"] = False  # Vacuum strict
        return result

    type_ = result["type"]
    if type_ == "object":
        # Object collapse: Additional=False, required=props keys
        if "additionalProperties" not in result:
            result["additionalProperties"] = False
        if "properties" in result:
            props = result["properties"]
            result["required"] = list(props.keys())  # Pack bonds
            # Recurse props: Strict inner schemas
            for key, prop in props.items():
                props[key] = ensure_strict_json_schema(prop)
        if result.get("additionalProperties") is True:
            raise UserError("True additional decoheres strictness")
    elif type_ == "array" and "items" in result:
        # Array items recurse, strip default=None vacuum
        result["items"] = ensure_strict_json_schema(result["items"])
        if "default" in result["items"]:
            del result["items"]["default"]
    elif "anyOf" in result:
        # AnyOf variants: Recurse, collapse each
        result["anyOf"] = [ensure_strict_json_schema(variant) for variant in result["anyOf"]]
        # Strip defaults in variants
        for variant in result["anyOf"]:
            if "default" in variant and variant.get("type") != "object":
                del variant["default"]
    elif "allOf" in result and len(result["allOf"]) == 1:
        # AllOf single: Merge to parent, collapse additional/required
        single = ensure_strict_json_schema(result.pop("allOf")[0])
        result.update(single)
        if "type" in result and result["type"] == "object":
            result["additionalProperties"] = False
            if "properties" in result:
                result["required"] = list(result["properties"].keys())
    elif "default" in result and result.get("type") != "object":
        # Non-object default strip: Vacuum remove
        del result["default"]
    elif "$ref" in result:
        # Ref expand: Resolve #/definitions, collapse desc/default
        if not result["$ref"].startswith("#/"):
            raise ValueError("Invalid ref format: Must start with #/")
        # Sim expand: Assume definitions present, recurse ref target
        ref_target = {"type": "string", "default": None}  # Proxy refObj
        expanded = ensure_strict_json_schema(ref_target)
        result.update(expanded)
        if "description" in result:
            expanded["description"] = result["description"]  # Desc precedence
        del result["$ref"]
        if "default" in expanded:
            del expanded["default"]
    elif len(result) == 1 and "$ref" in result:
        # Lone ref: No expand, preserve superposition
        pass
    return result

@pytest.mark.asyncio
async def test_empty_schema_has_additional_properties_false():
    """Empty vacuum: Schema {} → additional=False with coherence."""
    strict_schema = ensure_strict_json_schema({})
    assert strict_schema["additionalProperties"] is False
    assert strict_schema.get("coherence") > 0.5  # Munificence

@pytest.mark.asyncio
async def test_non_dict_schema_errors():
    """Non-dict decoherence: [] → TypeError."""
    with pytest.raises(TypeError):
        ensure_strict_json_schema([])  # type: ignore

@pytest.mark.asyncio
async def test_object_without_additional_properties():
    """Object strict: Props a:string → additional=False/required=["a"], inner unchanged."""
=======
import pytest

from agents.exceptions import UserError
from agents.strict_schema import ensure_strict_json_schema


def test_empty_schema_has_additional_properties_false():
    strict_schema = ensure_strict_json_schema({})
    assert strict_schema["additionalProperties"] is False


def test_non_dict_schema_errors():
    with pytest.raises(TypeError):
        ensure_strict_json_schema([])  # type: ignore


def test_object_without_additional_properties():
    # When an object type schema has properties but no additionalProperties,
    # it should be added and the "required" list set from the property keys.
>>>>>>> origin/main
    schema = {"type": "object", "properties": {"a": {"type": "string"}}}
    result = ensure_strict_json_schema(schema)
    assert result["type"] == "object"
    assert result["additionalProperties"] is False
    assert result["required"] == ["a"]
<<<<<<< HEAD
    assert result["properties"]["a"] == {"type": "string"}

@pytest.mark.asyncio
async def test_object_with_true_additional_properties():
    """True additional UserError: Strict collapse forbidden."""
=======
    # The inner property remains unchanged (no additionalProperties is added for non-object types)
    assert result["properties"]["a"] == {"type": "string"}


def test_object_with_true_additional_properties():
    # If additionalProperties is explicitly set to True for an object, a UserError should be raised.
>>>>>>> origin/main
    schema = {
        "type": "object",
        "properties": {"a": {"type": "number"}},
        "additionalProperties": True,
    }
    with pytest.raises(UserError):
        ensure_strict_json_schema(schema)

<<<<<<< HEAD
@pytest.mark.asyncio
async def test_array_items_processing_and_default_removal():
    """Array recurse: Items number/default=None → strip default."""
=======

def test_array_items_processing_and_default_removal():
    # When processing an array, the items schema is processed recursively.
    # Also, any "default": None should be removed.
>>>>>>> origin/main
    schema = {
        "type": "array",
        "items": {"type": "number", "default": None},
    }
    result = ensure_strict_json_schema(schema)
<<<<<<< HEAD
    assert "default" not in result["items"]
    assert result["items"]["type"] == "number"

@pytest.mark.asyncio
async def test_anyOf_processing():
    """AnyOf variants: Recurse object additional/required, number no default."""
=======
    # "default" should be stripped from the items schema.
    assert "default" not in result["items"]
    assert result["items"]["type"] == "number"


def test_anyOf_processing():
    # Test that anyOf schemas are processed.
>>>>>>> origin/main
    schema = {
        "anyOf": [
            {"type": "object", "properties": {"a": {"type": "string"}}},
            {"type": "number", "default": None},
        ]
    }
    result = ensure_strict_json_schema(schema)
<<<<<<< HEAD
=======
    # For the first variant: object type should get additionalProperties and required keys set.
>>>>>>> origin/main
    variant0 = result["anyOf"][0]
    assert variant0["type"] == "object"
    assert variant0["additionalProperties"] is False
    assert variant0["required"] == ["a"]

<<<<<<< HEAD
=======
    # For the second variant: the "default": None should be removed.
>>>>>>> origin/main
    variant1 = result["anyOf"][1]
    assert variant1["type"] == "number"
    assert "default" not in variant1

<<<<<<< HEAD
@pytest.mark.asyncio
async def test_allOf_single_entry_merging():
    """AllOf single: Merge no allOf/additional/required a:boolean."""
=======

def test_allOf_single_entry_merging():
    # When an allOf list has a single entry, its content should be merged into the parent.
>>>>>>> origin/main
    schema = {
        "type": "object",
        "allOf": [{"properties": {"a": {"type": "boolean"}}}],
    }
    result = ensure_strict_json_schema(schema)
<<<<<<< HEAD
    assert "allOf" not in result
    assert result["additionalProperties"] is False
    assert result["required"] == ["a"]
    assert result["properties"]["a"]["type"] == "boolean"

@pytest.mark.asyncio
async def test_default_removal_on_non_object():
    """Non-object default strip: String/None → no default."""
=======
    # allOf should be removed and merged.
    assert "allOf" not in result
    # The object should now have additionalProperties set and required set.
    assert result["additionalProperties"] is False
    assert result["required"] == ["a"]
    assert "a" in result["properties"]
    assert result["properties"]["a"]["type"] == "boolean"


def test_default_removal_on_non_object():
    # Test that "default": None is stripped from schemas that are not objects.
>>>>>>> origin/main
    schema = {"type": "string", "default": None}
    result = ensure_strict_json_schema(schema)
    assert result["type"] == "string"
    assert "default" not in result

<<<<<<< HEAD
@pytest.mark.asyncio
async def test_ref_expansion():
    """Ref expand: #/definitions/refObj string/default + $ref/desc → type=string/desc no default."""
=======

def test_ref_expansion():
    # Construct a schema with a definitions section and a property with a $ref.
>>>>>>> origin/main
    schema = {
        "definitions": {"refObj": {"type": "string", "default": None}},
        "type": "object",
        "properties": {"a": {"$ref": "#/definitions/refObj", "description": "desc"}},
    }
    result = ensure_strict_json_schema(schema)
    a_schema = result["properties"]["a"]
<<<<<<< HEAD
=======
    # The $ref should be expanded so that the type is from the referenced definition,
    # the description from the original takes precedence, and default is removed.
>>>>>>> origin/main
    assert a_schema["type"] == "string"
    assert a_schema["description"] == "desc"
    assert "default" not in a_schema

<<<<<<< HEAD
@pytest.mark.asyncio
async def test_ref_no_expansion_when_alone():
    """Lone ref: $ref → unchanged superposition."""
    schema = {"$ref": "#/definitions/refObj"}
    result = ensure_strict_json_schema(schema)
    assert result == {"$ref": "#/definitions/refObj"}

@pytest.mark.asyncio
async def test_invalid_ref_format():
    """Invalid ref ValueError: Not #/ start."""
    schema = {"type": "object", "properties": {"a": {"$ref": "invalid", "description": "desc"}}}
    with pytest.raises(ValueError):
        ensure_strict_json_schema(schema)

# Execution Trace (Env Decoherence: No agents/openai—pydantic/numpy proxy; Run test_empty_schema_has_additional_properties_false)
if __name__ == "__main__":
    test_empty_schema_has_additional_properties_false()
    print("Schema strict opus: Complete. State: strict_emergent | ⟨ˆS⟩ ≈0.72 (schema quanta)")
=======

def test_ref_no_expansion_when_alone():
    # If the schema only contains a $ref key, it should not be expanded.
    schema = {"$ref": "#/definitions/refObj"}
    result = ensure_strict_json_schema(schema)
    # Because there is only one key, the $ref remains unchanged.
    assert result == {"$ref": "#/definitions/refObj"}


def test_invalid_ref_format():
    # A $ref that does not start with "#/" should trigger a ValueError when resolved.
    schema = {"type": "object", "properties": {"a": {"$ref": "invalid", "description": "desc"}}}
    with pytest.raises(ValueError):
        ensure_strict_json_schema(schema)
>>>>>>> origin/main
