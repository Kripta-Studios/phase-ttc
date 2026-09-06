"""Six real ridge solves on fully synthetic rows exercise the complete nested routing backend."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from e_jepa_ttc.artifacts.risk_geometry_v10 import binding
from e_jepa_ttc.data.frozen_expert_tables_v10 import create_record, load_table
from e_jepa_ttc.evaluation.stage63_65 import benchmark_phase, strict_macro_mass
from e_jepa_ttc.models.full_cost_router_v11 import DIAGNOSTICS
from e_jepa_ttc.models.three_expert_router import PHASE17_FEATURES
from e_jepa_ttc.training import stage75_router_runtime_v11 as module


def test_complete_synthetic_router_backend_freezes_all_six_before_outer_features(
    tmp_path, monkeypatch
):
    count = 8192
    seq_ids = np.arange(count) % 9
    targets = np.array([2.0, 4.0, 8.0, -2.0])[(np.arange(count) // 9) % 4]
    metadata = pd.DataFrame(
        {
            "sample_token": [f"r{i}" for i in range(count)],
            "sequence_id": [f"s{i}" for i in seq_ids],
            "track_id": [f"t{(i // 36) % 3}" for i in range(count)],
            "target_ttc": targets,
            "outer_fold": seq_ids // 3,
        }
    )
    names = {"C_DIRECT": "S73-DIRECT-D", "C_PPS": "S73-PPS-AD"}
    rows_by_name = {}
    for family, name in names.items():
        prediction = targets + (0.05 if family == "C_DIRECT" else 0.1)
        rows = metadata.rename(columns={"target_ttc": "target_ttc_s"}).assign(
            prediction_ttc_s=prediction,
            prediction_phase=benchmark_phase(prediction),
            finite=True,
            failure=False,
        )
        for index, diagnostic in enumerate(DIAGNOSTICS):
            rows[diagnostic] = np.arange(count) % (index + 2) / (index + 2)
        rows_by_name[name] = rows
    table_objects = {}
    nested_rows = {}
    for outer in range(3):
        inner_sequences = [f"s{i}" for i in range(9) if i // 3 != outer]
        inner_map = {name: index // 2 for index, name in enumerate(inner_sequences)}
        for role in ("inner_oof", "outer_dev"):
            keep = (
                metadata.outer_fold != outer
                if role == "inner_oof"
                else metadata.outer_fold == outer
            )
            meta = metadata.loc[keep].reset_index(drop=True).copy()
            meta["role"] = role
            meta["inner_fold"] = meta.sequence_id.map(inner_map).fillna(-1).astype(int)
            target = meta.target_ttc.to_numpy(np.float64)
            expert = np.column_stack([target + 0.01, target + 0.02, target + 0.03])
            features = np.zeros((len(meta), 17), np.float64)
            for column, expert_name in enumerate(("a5", "c2f", "pair")):
                features[:, PHASE17_FEATURES.index(f"{expert_name}_benchmark_phase")] = (
                    benchmark_phase(expert[:, column])
                )
            path = tmp_path / f"old_outer{outer}_{role}.npz"
            np.savez(
                path,
                features17=features,
                expert_phase=benchmark_phase(expert),
                expert_ttc=expert,
                target_phase=benchmark_phase(target),
                global_mass=strict_macro_mass(target, meta.sequence_id.to_numpy()),
            )
            csv = path.with_suffix(".csv")
            meta.to_csv(csv, index=False)
            table_objects[(outer, role)] = load_table(
                create_record(path, csv, outer, role), role=role
            )
            if role == "inner_oof":
                for name, rows in rows_by_name.items():
                    new = rows.set_index("sample_token").loc[meta.sample_token].reset_index()
                    new["inner_fold"] = meta.inner_fold.to_numpy()
                    new["prediction_role"] = "genuinely_nested_inner_oof"
                    nested_rows[(name, outer)] = new
    source_rows = {}
    for name, rows in rows_by_name.items():
        path = tmp_path / "initial_exports" / name / "ROWS.csv"
        path.parent.mkdir(parents=True)
        rows.to_csv(path, index=False)
        source_rows[name] = binding(path)
    barrier = tmp_path / "barriers/S75_ALL_ENDPOINTS_FROZEN.json"
    barrier.parent.mkdir()
    barrier.write_text("synthetic endpoint barrier")
    runtime = SimpleNamespace(
        root=tmp_path,
        lock_sha256="a" * 64,
        lock={
            "stage_barrier_root": str(barrier.parent),
            "router_root": str(tmp_path / "routers"),
            "initial_evidence_root": str(tmp_path / "evidence"),
            "treatment_export_root": str(tmp_path / "initial_exports"),
        },
        _reserve_resources=lambda key: None,
        ledger=SimpleNamespace(record_ridge_attempt=lambda attempt, completed: None),
    )
    decision = {
        "canonical_candidates": names,
        "canonical_technically_valid": {"C_DIRECT": True, "C_PPS": True},
    }
    monkeypatch.setattr(module, "verify_stage_decision", lambda *args: (decision, (object(),)))
    monkeypatch.setattr(module, "verify_nested_barrier", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        module, "verify_initial_evidence", lambda *args: {"row_sources": source_rows}
    )
    fits = []
    actual_fit = module.fit_full_cost_endpoint

    def fit(*args, **kwargs):
        result = actual_fit(*args, **kwargs)
        fits.append(kwargs["identity"]["arm"])
        return result

    monkeypatch.setattr(module, "fit_full_cost_endpoint", fit)

    def historical(lock, outer, role):
        if role == "outer_dev":
            assert len(fits) == 6
            assert (tmp_path / "routers/ROUTER_ENDPOINTS_FROZEN.json").is_file()
        return table_objects[(outer, role)], {"technical_synthetic_fixture": True}

    monkeypatch.setattr(module, "historical_table", historical)
    monkeypatch.setattr(
        module,
        "_nested_rows",
        lambda lock, name, outer, barrier: (
            nested_rows[(name, outer)],
            {"synthetic_source": name, "outer": outer},
        ),
    )
    result = module.run_nested_routers(runtime, {"path": "synthetic decision"})
    assert len(fits) == 6 and set(result["exports"]) == set(module.ARMS)
    for source in result["exports"].values():
        import json

        record = json.loads(Path(source["path"]).read_text(encoding="utf-8"))
        rows = pd.read_csv(record["rows"]["path"])
        assert len(rows) == 8192 and rows.sample_token.tolist() == metadata.sample_token.tolist()
        assert rows.finite.all() and not rows.failure.any()
        assert record["experts"] == ["A5", "C2F", "PAIR", "DIRECT", "PPS"]
        assert all(f"{expert}_true_cost" in rows for expert in record["experts"])
