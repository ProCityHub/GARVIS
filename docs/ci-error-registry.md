# GARVIS CI Error Registry

Authority: Adrien D. Thomas / ProCityHub

These records preserve real CI failures for repair. They are not waived, hidden, reclassified as successes, or confused with speculative scientific claims.

Open records: **43**

## Filing summary

- `immutable_value_mutation`: 1
- `import_order`: 3
- `interface_contract_mismatch`: 19
- `lambda_assignment`: 1
- `missing_type_annotation`: 1
- `non_overlapping_type_comparison`: 1
- `optional_value_not_narrowed`: 13
- `overbroad_exception_assertion`: 2
- `unknown_callable_type`: 1
- `unused_import`: 1

## Records

| ID | Tool | Classification | Location | Code |
|---|---|---|---|---|
| CI-011 | mypy | `immutable_value_mutation` | `tests/garvis/arc3/test_planner.py:93` | `misc` |
| CI-038 | ruff | `import_order` | `tests/garvis/arc3/test_goal_hypothesis.py:3:1` | `I001` |
| CI-040 | ruff | `import_order` | `tests/garvis/arc_static/test_dsl.py:3:1` | `I001` |
| CI-041 | ruff | `import_order` | `tests/garvis/arc_static/test_search.py:3:1` | `I001` |
| CI-027 | mypy | `interface_contract_mismatch` | `tests/garvis/arc3/test_action_effect_learner.py:56` | `arg-type` |
| CI-033 | mypy | `interface_contract_mismatch` | `tests/garvis/arc3/test_action_effect_learner.py:122` | `arg-type` |
| CI-034 | mypy | `interface_contract_mismatch` | `tests/garvis/arc3/test_action_effect_learner.py:133` | `arg-type` |
| CI-026 | mypy | `interface_contract_mismatch` | `tests/garvis/arc3/test_frame_differ.py:114` | `arg-type` |
| CI-014 | mypy | `interface_contract_mismatch` | `tests/garvis/arc3/test_game_memory.py:17` | `arg-type` |
| CI-015 | mypy | `interface_contract_mismatch` | `tests/garvis/arc3/test_game_memory.py:18` | `arg-type` |
| CI-016 | mypy | `interface_contract_mismatch` | `tests/garvis/arc3/test_game_memory.py:19` | `arg-type` |
| CI-017 | mypy | `interface_contract_mismatch` | `tests/garvis/arc3/test_game_memory.py:36` | `arg-type` |
| CI-018 | mypy | `interface_contract_mismatch` | `tests/garvis/arc3/test_game_memory.py:37` | `arg-type` |
| CI-019 | mypy | `interface_contract_mismatch` | `tests/garvis/arc3/test_game_memory.py:38` | `arg-type` |
| CI-020 | mypy | `interface_contract_mismatch` | `tests/garvis/arc3/test_game_memory.py:81` | `arg-type` |
| CI-021 | mypy | `interface_contract_mismatch` | `tests/garvis/arc3/test_game_memory.py:98` | `arg-type` |
| CI-022 | mypy | `interface_contract_mismatch` | `tests/garvis/arc3/test_game_memory.py:100` | `arg-type` |
| CI-023 | mypy | `interface_contract_mismatch` | `tests/garvis/arc3/test_game_memory.py:102` | `arg-type` |
| CI-024 | mypy | `interface_contract_mismatch` | `tests/garvis/arc3/test_game_memory.py:104` | `arg-type` |
| CI-012 | mypy | `interface_contract_mismatch` | `tests/garvis/arc3/test_goal_hypothesis.py:113` | `arg-type` |
| CI-013 | mypy | `interface_contract_mismatch` | `tests/garvis/arc3/test_goal_hypothesis.py:115` | `arg-type` |
| CI-010 | mypy | `interface_contract_mismatch` | `tests/garvis/arc3/test_planner.py:78` | `arg-type` |
| CI-035 | mypy | `interface_contract_mismatch` | `tests/garvis/test_anthropic_backend.py:20` | `arg-type` |
| CI-043 | ruff | `lambda_assignment` | `tests/garvis/arc_static/test_search.py:21:5` | `E731` |
| CI-036 | mypy | `missing_type_annotation` | `tests/garvis/test_anthropic_backend.py:30` | `var-annotated` |
| CI-025 | mypy | `non_overlapping_type_comparison` | `tests/garvis/arc3/test_frame_parser.py:86` | `comparison-overlap` |
| CI-028 | mypy | `optional_value_not_narrowed` | `tests/garvis/arc3/test_action_effect_learner.py:81` | `union-attr` |
| CI-029 | mypy | `optional_value_not_narrowed` | `tests/garvis/arc3/test_action_effect_learner.py:82` | `union-attr` |
| CI-030 | mypy | `optional_value_not_narrowed` | `tests/garvis/arc3/test_action_effect_learner.py:83` | `union-attr` |
| CI-031 | mypy | `optional_value_not_narrowed` | `tests/garvis/arc3/test_action_effect_learner.py:91` | `union-attr` |
| CI-032 | mypy | `optional_value_not_narrowed` | `tests/garvis/arc3/test_action_effect_learner.py:92` | `union-attr` |
| CI-003 | mypy | `optional_value_not_narrowed` | `tests/garvis/arc3/test_planner.py:13` | `union-attr` |
| CI-004 | mypy | `optional_value_not_narrowed` | `tests/garvis/arc3/test_planner.py:14` | `union-attr` |
| CI-005 | mypy | `optional_value_not_narrowed` | `tests/garvis/arc3/test_planner.py:15` | `union-attr` |
| CI-006 | mypy | `optional_value_not_narrowed` | `tests/garvis/arc3/test_planner.py:20` | `union-attr` |
| CI-007 | mypy | `optional_value_not_narrowed` | `tests/garvis/arc3/test_planner.py:20` | `union-attr` |
| CI-008 | mypy | `optional_value_not_narrowed` | `tests/garvis/arc3/test_planner.py:66` | `union-attr` |
| CI-009 | mypy | `optional_value_not_narrowed` | `tests/garvis/arc3/test_planner.py:73` | `union-attr` |
| CI-001 | mypy | `optional_value_not_narrowed` | `tests/garvis/arc_static/test_search.py:39` | `union-attr` |
| CI-037 | ruff | `overbroad_exception_assertion` | `tests/garvis/arc3/test_game_memory.py:115:10` | `B017` |
| CI-039 | ruff | `overbroad_exception_assertion` | `tests/garvis/arc3/test_planner.py:92:10` | `B017` |
| CI-002 | mypy | `unknown_callable_type` | `tests/garvis/arc_static/test_dsl.py:103` | `operator` |
| CI-042 | ruff | `unused_import` | `tests/garvis/arc_static/test_search.py:6:38` | `F401` |

Full messages and repair dispositions are stored in `config/ci_error_registry.json`.
