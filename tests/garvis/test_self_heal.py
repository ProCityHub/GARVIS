from pathlib import Path

from garvis.self_heal import SEALED_REPAIR_REQUIRED



def test_public_module_exports_sealed_repair_constant() -> None:
    assert SEALED_REPAIR_REQUIRED == "SEALED_REPAIR_REQUIRED"
