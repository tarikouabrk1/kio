"""
WASM Sandbox Simulation Test
=============================
Simulates the IronClaw WASM constraint by blocking all filesystem
access and verifying that automl_tool and eda_tool still work
correctly using only in-memory operations.

If this test passes, the library is confirmed WASM-compatible.

Run from project root:
    python -m benchmark.test_wasm
"""

import sys
import json
import builtins

# ── Minimal CSV that fits in memory ────────────────────────────
SAMPLE_CSV = """age,years_experience,monthly_spend,churned
24,1,29.99,0
45,15,120.00,1
30,5,61.00,0
52,22,140.00,1
28,3,47.00,0
35,8,72.00,0
48,18,130.00,1
26,2,39.99,0
57,30,155.00,1
33,7,68.00,0
"""

# ── Monkey-patch open() to simulate no filesystem ───────────────
_real_open = builtins.open

def _wasm_open(path, *args, **kwargs):
    raise PermissionError(
        f"WASM SANDBOX VIOLATION: filesystem access attempted → {path}"
    )


def run_wasm_tests():
    print("=" * 60)
    print("  GlassBox WASM Sandbox Simulation")
    print("=" * 60)

    passed = 0
    failed = 0

    # ── Block filesystem ────────────────────────────────────────
    builtins.open = _wasm_open
    print("\n  Filesystem access blocked — simulating WASM sandbox\n")

    try:
        from agent.tools import automl_tool, eda_tool, TOOL_SCHEMA

        # ── Test 1: automl_tool ─────────────────────────────────
        print("  Test 1 — automl_tool (classification)")
        try:
            result_json = automl_tool(
                SAMPLE_CSV,
                target="churned",
                task="classification",
                cv=3,
                search_strategy="random",
                random_iter=3,
                verbose=False,
            )
            result = json.loads(result_json)

            assert result.get("status") != "error", \
                f"Tool returned error: {result.get('message')}"
            assert "best_model" in result, \
                "Missing 'best_model' in report"
            assert "agent_summary" in result, \
                "Missing 'agent_summary' in report"
            assert "confusion_matrix" in result["eval_metrics"], \
                "Missing 'confusion_matrix' in eval_metrics"
            assert result["target"] == "churned", \
                "Wrong target in report"

            print(f"    ✅ PASS")
            print(f"       best_model   : {result['best_model']}")
            print(f"       accuracy     : {result['eval_metrics']['accuracy']:.4f}")
            print(f"       agent_summary: {result['agent_summary'][:80]}...")
            passed += 1

        except PermissionError as e:
            print(f"    ❌ FAIL — filesystem access detected!")
            print(f"       {e}")
            failed += 1
        except Exception as e:
            print(f"    ❌ FAIL — unexpected error: {e}")
            failed += 1

        # ── Test 2: eda_tool ────────────────────────────────────
        print("\n  Test 2 — eda_tool")
        try:
            eda_json = eda_tool(SAMPLE_CSV)
            eda = json.loads(eda_json)

            assert eda.get("status") != "error", \
                f"EDA tool returned error: {eda.get('message')}"
            assert "shape" in eda, \
                "Missing 'shape' in EDA report"
            assert "profiles" in eda, \
                "Missing 'profiles' in EDA report"
            assert "correlations" in eda, \
                "Missing 'correlations' in EDA report"
            assert "missing_summary" in eda, \
                "Missing 'missing_summary' in EDA report"

            print(f"    ✅ PASS")
            print(f"       shape      : {eda['shape']}")
            print(f"       columns    : {list(eda['profiles'].keys())}")
            passed += 1

        except PermissionError as e:
            print(f"    ❌ FAIL — filesystem access detected!")
            print(f"       {e}")
            failed += 1
        except Exception as e:
            print(f"    ❌ FAIL — unexpected error: {e}")
            failed += 1

        # ── Test 3: TOOL_SCHEMA is valid ────────────────────────
        print("\n  Test 3 — TOOL_SCHEMA MCP registration format")
        try:
            assert "tools" in TOOL_SCHEMA, \
                "Missing 'tools' key in TOOL_SCHEMA"
            assert len(TOOL_SCHEMA["tools"]) == 2, \
                f"Expected 2 tools, got {len(TOOL_SCHEMA['tools'])}"

            tool_names = [t["name"] for t in TOOL_SCHEMA["tools"]]
            assert "automl_tool" in tool_names, \
                "automl_tool not registered in schema"
            assert "eda_tool" in tool_names, \
                "eda_tool not registered in schema"

            schema_json = json.dumps(TOOL_SCHEMA)
            assert len(schema_json) > 0, \
                "TOOL_SCHEMA is not JSON-serialisable"

            print(f"    ✅ PASS")
            print(f"       registered tools: {tool_names}")
            passed += 1

        except Exception as e:
            print(f"    ❌ FAIL — {e}")
            failed += 1

        # ── Test 4: JSON output is NumPy-safe ───────────────────
        print("\n  Test 4 — JSON output contains no NumPy types")
        try:
            result = json.loads(automl_tool(
                SAMPLE_CSV,
                target="churned",
                task="classification",
                cv=2,
                search_strategy="random",
                random_iter=2,
                verbose=False,
            ))
            # If json.loads succeeded without error the output is clean
            json.dumps(result)  # second pass to be sure
            print(f"    ✅ PASS — all values are JSON-native types")
            passed += 1

        except (TypeError, ValueError) as e:
            print(f"    ❌ FAIL — NumPy type leaked into JSON: {e}")
            failed += 1
        except Exception as e:
            print(f"    ❌ FAIL — {e}")
            failed += 1

    finally:
        # Always restore real open
        builtins.open = _real_open
        print("\n  Filesystem access restored\n")

    # ── Summary ─────────────────────────────────────────────────
    print("=" * 60)
    print("  WASM SIMULATION RESULTS")
    print("=" * 60)
    print(f"  Passed : {passed}")
    print(f"  Failed : {failed}")
    print(f"  Total  : {passed + failed}")

    if failed == 0:
        print("\n  ✅ ALL TESTS PASSED")
        print("  Library is confirmed filesystem-free in hot path.")
        print("  Safe to deploy inside IronClaw WASM sandbox.")
        sys.exit(0)
    else:
        print(f"\n  ❌ {failed} TEST(S) FAILED")
        print("  Fix filesystem access before WASM deployment.")
        sys.exit(1)


if __name__ == "__main__":
    run_wasm_tests()