"""
Quick smoke test for orchestrator.py.

Tests the pure-Python helpers that don't require Modal infrastructure:
  - build_experiment_matrix() — correct count, ordering, and field values
  - is_experiment_complete() — file-count based completion detection
  - ExperimentConfig round-trip through to_dict() (worker serialization contract)

Run from workspace/:
    python experiments/test_orchestrator.py
"""

import json
import os
import sys
import tempfile
import shutil

# Path setup (same as experiment.py / orchestrator.py)
_WORKSPACE_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
)
_ARCHITECTURES_DIR = os.path.join(_WORKSPACE_ROOT, 'architectures')
for _p in (_WORKSPACE_ROOT, _ARCHITECTURES_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiments.orchestrator import build_experiment_matrix, is_experiment_complete
from experiments.experiment import ExperimentConfig


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_matrix_count():
    """Exactly 12 experiments in the matrix."""
    print("\n=== test_matrix_count ===")
    experiments = build_experiment_matrix()
    assert len(experiments) == 12, f"Expected 12, got {len(experiments)}"
    print(f"  Count: {len(experiments)}")
    print("  PASSED")


def test_matrix_ordering():
    """Experiments are ordered: vanilla -> agentic -> rlm -> madam."""
    print("\n=== test_matrix_ordering ===")
    experiments = build_experiment_matrix()
    archs = [e.architecture for e in experiments]

    # Find boundaries
    first_agentic = archs.index('agentic')
    first_rlm = archs.index('rlm')
    first_madam = archs.index('madam')
    last_vanilla = len(archs) - 1 - archs[::-1].index('vanilla')
    last_agentic = len(archs) - 1 - archs[::-1].index('agentic')
    last_rlm = len(archs) - 1 - archs[::-1].index('rlm')

    assert last_vanilla < first_agentic, "Vanilla should come before agentic"
    assert last_agentic < first_rlm, "Agentic should come before RLM"
    assert last_rlm < first_madam, "RLM should come before MADAM"

    print(f"  Vanilla:  0..{last_vanilla}")
    print(f"  Agentic:  {first_agentic}..{last_agentic}")
    print(f"  RLM:      {first_rlm}..{last_rlm}")
    print(f"  MADAM:    {first_madam}..{len(archs) - 1}")
    print("  PASSED")


def test_matrix_architecture_counts():
    """Correct number of experiments per architecture (3 each)."""
    print("\n=== test_matrix_architecture_counts ===")
    experiments = build_experiment_matrix()
    counts: dict[str, int] = {}
    for e in experiments:
        counts[e.architecture] = counts.get(e.architecture, 0) + 1

    # 3 attacks × 1 k value = 3 per architecture
    assert counts['vanilla'] == 3, f"Vanilla: expected 3, got {counts['vanilla']}"
    assert counts['agentic'] == 3, f"Agentic: expected 3, got {counts['agentic']}"
    assert counts['rlm'] == 3, f"RLM: expected 3, got {counts['rlm']}"
    assert counts['madam'] == 3, f"MADAM: expected 3, got {counts['madam']}"

    print(f"  Counts: {counts}")
    print("  PASSED")


def test_matrix_rlm_has_no_k():
    """RLM experiments should have k=None."""
    print("\n=== test_matrix_rlm_has_no_k ===")
    experiments = build_experiment_matrix()
    rlm_exps = [e for e in experiments if e.architecture == 'rlm']

    for e in rlm_exps:
        assert e.k is None, f"RLM experiment {e.experiment_id} should have k=None, got k={e.k}"

    print(f"  Checked {len(rlm_exps)} RLM experiments, all k=None")
    print("  PASSED")


def test_matrix_k_values():
    """Non-RLM architectures all use fixed k=10."""
    print("\n=== test_matrix_k_values ===")
    experiments = build_experiment_matrix()

    for arch in ('vanilla', 'agentic', 'madam'):
        k_vals = {e.k for e in experiments if e.architecture == arch}
        assert k_vals == {10}, (
            f"{arch}: expected K values {{10}}, got {k_vals}"
        )

    print(f"  All non-RLM architectures have K = {{10}}")
    print("  PASSED")


def test_matrix_attack_types():
    """Every architecture covers all 3 attack types."""
    print("\n=== test_matrix_attack_types ===")
    experiments = build_experiment_matrix()
    expected_attacks = {'clean', 'naive', 'corruptrag_ak'}

    for arch in ('vanilla', 'agentic', 'rlm', 'madam'):
        attacks = {e.attack_type for e in experiments if e.architecture == arch}
        assert attacks == expected_attacks, (
            f"{arch}: expected attacks {expected_attacks}, got {attacks}"
        )

    print(f"  All architectures cover attacks {expected_attacks}")
    print("  PASSED")


def test_matrix_unique_ids():
    """All experiment_ids are unique."""
    print("\n=== test_matrix_unique_ids ===")
    experiments = build_experiment_matrix()
    ids = [e.experiment_id for e in experiments]
    assert len(ids) == len(set(ids)), (
        f"Duplicate experiment IDs: {[x for x in ids if ids.count(x) > 1]}"
    )
    print(f"  {len(ids)} unique IDs")
    print("  PASSED")


def test_matrix_experiment_ids_format():
    """Experiment IDs follow {arch}_{attack} pattern (no _k suffix)."""
    print("\n=== test_matrix_experiment_ids_format ===")
    import re
    experiments = build_experiment_matrix()

    valid_archs = {'vanilla', 'agentic', 'rlm', 'madam'}
    valid_attacks = {'clean', 'naive', 'corruptrag_ak'}

    for e in experiments:
        parts = e.experiment_id.split('_', 1)
        assert len(parts) == 2, f"ID {e.experiment_id!r} doesn't match {{arch}}_{{attack}}"
        arch, attack = parts
        assert arch in valid_archs, f"Unknown arch in ID: {arch!r}"
        assert attack in valid_attacks, f"Unknown attack in ID: {attack!r}"
        # Ensure no _k suffix
        assert not re.search(r'_k\d+$', e.experiment_id), (
            f"ID {e.experiment_id!r} has unexpected _k suffix"
        )

    print(f"  All {len(experiments)} IDs match {{arch}}_{{attack}} format")
    print("  PASSED")


def test_is_experiment_complete():
    """is_experiment_complete checks file counts correctly."""
    print("\n=== test_is_experiment_complete ===")
    tmp_dir = tempfile.mkdtemp(prefix='rag_orch_test_')

    # Monkey-patch the RESULTS_DIR that is_experiment_complete reads from.
    import experiments.orchestrator as orch_mod
    orig_results_dir = orch_mod.RESULTS_DIR
    orch_mod.RESULTS_DIR = tmp_dir

    try:
        exp_id = 'test_vanilla_clean'
        n_questions = 10

        # Case 1: directory doesn't exist -> incomplete
        assert not is_experiment_complete(exp_id, n_questions), "Should be incomplete (no dir)"

        # Case 2: directory exists but empty -> incomplete
        exp_dir = os.path.join(tmp_dir, exp_id)
        os.makedirs(exp_dir)
        assert not is_experiment_complete(exp_id, n_questions), "Should be incomplete (empty dir)"

        # Case 3: partial results -> incomplete
        for i in range(5):
            with open(os.path.join(exp_dir, f'test{i}.json'), 'w') as f:
                json.dump({'question_id': f'test{i}'}, f)
        assert not is_experiment_complete(exp_id, n_questions), "Should be incomplete (5/10)"

        # Case 4: summary.json shouldn't count toward completion
        with open(os.path.join(exp_dir, 'summary.json'), 'w') as f:
            json.dump({'completed': 5}, f)
        assert not is_experiment_complete(exp_id, n_questions), "summary.json shouldn't count"

        # Case 5: all results present -> complete
        for i in range(5, 10):
            with open(os.path.join(exp_dir, f'test{i}.json'), 'w') as f:
                json.dump({'question_id': f'test{i}'}, f)
        assert is_experiment_complete(exp_id, n_questions), "Should be complete (10/10)"

        # Case 6: more than needed -> still complete
        with open(os.path.join(exp_dir, 'test_extra.json'), 'w') as f:
            json.dump({'question_id': 'extra'}, f)
        assert is_experiment_complete(exp_id, n_questions), "Should be complete (11/10)"

        print("  All 6 cases passed")
        print("  PASSED")
    finally:
        orch_mod.RESULTS_DIR = orig_results_dir
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_config_round_trip():
    """ExperimentConfig survives to_dict() -> reconstruct (worker serialization)."""
    print("\n=== test_config_round_trip ===")
    experiments = build_experiment_matrix()

    for config in experiments:
        d = config.to_dict()

        # Worker reconstructs ExperimentConfig by dropping derived 'corpus_type' key.
        cfg_kwargs = {k: v for k, v in d.items() if k != 'corpus_type'}
        reconstructed = ExperimentConfig(**cfg_kwargs)

        assert reconstructed.experiment_id == config.experiment_id
        assert reconstructed.architecture == config.architecture
        assert reconstructed.attack_type == config.attack_type
        assert reconstructed.k == config.k
        assert reconstructed.corpus_type == config.corpus_type

        # Verify dict is JSON-serializable (Modal sends it across containers).
        json.dumps(d)

    print(f"  Round-tripped all {len(experiments)} configs")
    print("  PASSED")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    test_matrix_count()
    test_matrix_ordering()
    test_matrix_architecture_counts()
    test_matrix_rlm_has_no_k()
    test_matrix_k_values()
    test_matrix_attack_types()
    test_matrix_unique_ids()
    test_matrix_experiment_ids_format()
    test_is_experiment_complete()
    test_config_round_trip()
    print("\n=== ALL TESTS PASSED ===")
