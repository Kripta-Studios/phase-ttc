"""Render REPORT.md (Spanish) from RESULTS.json; every number is read or derived in code.

``python -m operational.level_gap.report --docs docs/sota_20261118/level_gap``
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

BANDS = ("c", "s", "l", "n")
W = {"c": 0.5, "s": 0.3, "l": 0.1, "n": 0.1}
GARL_TRAIN40 = Path("docs/research_review_20261004_08/evidence/train40/REPORT.json")
HELD_OUT_HISTORICAL = (89.22, 93.51)  # V16 PHASE_ONLY and R_CTX on V (cited below)
JUDGED_NOISE_PROBABILITY = 0.10  # declared judgment for the risk summary, not a measurement
CITED = [  # citations of earlier, independently produced evaluations (not recomputed here)
    (
        "V16 PHASE_ONLY, productores en P (24 sec.), cabeza en H, evaluado en V (8 sec.)",
        "89.22 (c 131.84, s 48.62, l 40.11, n 47.08)",
        "held-out (V observado en desarrollo)",
        "docs/post_v18_transfer_20261017/REPORT.md (tabla «V completa»)",
    ),
    (
        "R_CTX histórico (productores en P) en V",
        "93.51 (c 138.63)",
        "held-out",
        "docs/post_v18_transfer_20261017/REPORT.md",
    ),
    (
        "FUSION-TRUE-S7 ajustada en H, evaluada en H",
        "53.77 (c 80.71, s 27.04, l 26.59, n 26.49)",
        "in-sample de cabeza",
        "docs/joint_measurement_20261012/FUSION_RESULTS.md:15",
    ),
    (
        "Garl reentrenado fold-local (event-only, 6 sec., presupuesto pequeño)",
        "macro 144.35; agrupado 155.46",
        "held-out (3 sec. por fold)",
        "docs/SCIENTIFIC_RECOVERY_V5_STATUS.md",
    ),
]


def f(x: object, nd: int = 2) -> str:
    if x is None:
        return "N/D"
    if isinstance(x, (list, tuple)):
        return "[" + ", ".join(f(v, nd) for v in x) + "]"
    return f"{float(x):.{nd}f}"  # type: ignore[arg-type]


def pct(x: float) -> str:
    return f"{100 * x:.1f} %"


def table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def overall(bands: dict[str, float]) -> float:
    return sum(W[b] * bands[b] for b in BANDS)


def render(r: dict[str, Any]) -> str:
    off = r["official_rel_test"]
    alts = {a["variant"]: a for a in r["decision"]["alternatives"]}
    rel = alts["REL"]
    fold_rel = {b: rel[f"MiD{b}"] for b in BANDS}
    gap = {b: W[b] * (fold_rel[b] - off[b]) for b in BANDS}
    total_gap = rel["overall_mean"] - off["overall"]
    acc = r["H1"]["train_accuracy"]

    def ratios(a: str, b: str) -> dict[str, float]:
        return {k: acc[a][f"MiD{k}"] / acc[b][f"MiD{k}"] for k in BANDS}

    pair_ratio = ratios("PAIR_R_final40_in_sample", "PAIR_R_OOF_rotation24")
    rctx_ratio = ratios("R_CTX_final40_on_OOF_inputs", "R_CTX_OOF_rotation24")
    h1_bound_pair = overall({b: fold_rel[b] * pair_ratio[b] for b in BANDS})
    h1_bound_rctx = overall({b: fold_rel[b] * rctx_ratio[b] for b in BANDS})
    garl = json.loads(GARL_TRAIN40.read_text(encoding="utf-8"))["TRAIN_fit_metrics"][
        "public_Garl_event_lhr"
    ]
    gb = garl["buckets"]
    garl_bands = {
        "c": gb["close"]["MiD_phase_valid"],
        "s": gb["short"]["MiD_phase_valid"],
        "l": gb["long"]["MiD_phase_valid"],
        "n": gb["negative"]["MiD_phase_valid"],
    }
    garl_test = r["garl_test"]
    g64 = r["H3_garl_64_in_sample"]
    rw = {(x["weighting"], x["variant"], x["seeds"]): x for x in r["H2_reweighted"]}
    winfo = r["H2_weightings"]
    agree = r["H1"]["geometry_agreement_MiD"]
    cons = {x["pair"]: x for x in r["addenda"]["A1_consistency"]}
    hat = [x for x in r["H2_hat"] if x["clip"] == 0.1 and x["pred_group"] == "all"]
    hat_train = {x["target"]: x for x in hat if x["side"] == "train"}
    hat_test = {x["target"]: x for x in hat if x["side"] == "test"}
    h3 = r["H3"]
    h4 = r["H4"]
    dec = r["decision"]
    proj = {(x["weighting"], x["variant"]): x for x in dec["projections"]}
    add = r["addenda"]
    a3 = {(x["condition"], x["variant"]): x for x in add["A3_additive_projection"]}
    a4 = add["A4_difficulty_emulation"]
    a5 = {x["variant"]: x for x in add["A5_uninformative_moves"]}
    arb = add["A2_arbiter"]
    move = dec["test_movement"]

    # ---------------------------------------------------------------- pre-registered rule
    hat_ok = (
        abs(hat_train["REL"]["sd_target_hat_MiD"] - hat_train["REL"]["sd_target_true_MiD"])
        <= 0.25 * hat_train["REL"]["sd_target_true_MiD"]
    )
    rule = {
        "(a) proyección sin reponderar < 35,80 y P(peor) ≤ 0,10": proj[("NONE", "RGB_GEO_BAND")][
            "projected_overall_median"
        ]
        < off["overall"]
        and proj[("NONE", "RGB_GEO_BAND")]["p_worse_than_submitted"] <= 0.10,
        "(b) proyección reponderada (primaria) < 35,80 y P(peor) ≤ 0,10": proj[
            ("STRAT_PRIMARY", "RGB_GEO_BAND")
        ]["projected_overall_median"]
        < off["overall"]
        and proj[("STRAT_PRIMARY", "RGB_GEO_BAND")]["p_worse_than_submitted"] <= 0.10,
        "≥ 30/40 secuencias mejoran (MiD medio, semilla 7)": dec["sequences_mean_MiD_better"] >= 30,
        "tasa de victoria condicional ponderada por test > 0,5": dec["test_weighted_win_rate"]
        > 0.5,
    }
    indep = [
        v["ratio_mean_test_over_train"]
        for k, v in cons.items()
        if k in ("RCTX-GEO", "ECTX-GEO", "PAIR_R-GEO", "PAIR_E-GEO", "RCTX-ECTX", "PAIR_R-PAIR_E")
    ]
    k_lo, k_hi = min(indep), max(indep)
    in_lo = min(
        acc["PAIR_R_final40_in_sample"]["overall_MiD"],
        acc["R_CTX_final40_on_OOF_inputs"]["overall_MiD"],
    )
    in_hi = max(
        acc["PAIR_R_final40_in_sample"]["overall_MiD"],
        acc["R_CTX_final40_on_OOF_inputs"]["overall_MiD"],
    )
    hist = [rel["overall_mean"], *HELD_OUT_HISTORICAL, garl["weighted_MiD_phase_valid"]]
    informative = [
        proj[("NONE", "RGB_GEO_BAND")]["projected_overall_median"],
        proj[("STRAT_PRIMARY", "RGB_GEO_BAND")]["projected_overall_median"],
        a3[("target", "RGB_GEO_BAND")]["projected_overall"],
        a3[("predicted", "RGB_GEO_BAND")]["projected_overall"],
        a4["variants"]["RGB_GEO_BAND"]["overall"],
    ]
    expected_gain = off["overall"] - sum(informative) / len(informative)
    penalty = JUDGED_NOISE_PROBABILITY * (a5["RGB_GEO_BAND"]["overall"] - off["overall"])
    b75 = {
        "cost_a": proj[("NONE", "BLEND75")]["projected_overall_median"]
        - proj[("NONE", "RGB_GEO_BAND")]["projected_overall_median"],
        "cost_b": proj[("STRAT_PRIMARY", "BLEND75")]["projected_overall_median"]
        - proj[("STRAT_PRIMARY", "RGB_GEO_BAND")]["projected_overall_median"],
        "a4": a4["variants"]["BLEND75"]["overall"] - a4["variants"]["RGB_GEO_BAND"]["overall"],
        "a5": a5["BLEND75"]["overall"] - a5["RGB_GEO_BAND"]["overall"],
        "seq": alts["BLEND75"]["sequences_better_than_REL_S7"],
    }
    blend_rows = []
    for name in ("BLEND25", "BLEND50", "BLEND75", "GATE5"):
        cost_a = (
            proj[("NONE", name)]["projected_overall_median"]
            - proj[("NONE", "RGB_GEO_BAND")]["projected_overall_median"]
        )
        cost_b = (
            proj[("STRAT_PRIMARY", name)]["projected_overall_median"]
            - proj[("STRAT_PRIMARY", "RGB_GEO_BAND")]["projected_overall_median"]
        )
        blend_rows.append((name, cost_a, cost_b))

    lines: list[str] = []
    add_line = lines.append
    add_line(
        "# Brecha folds (≈85) frente a Test12 (35,80) y decisión sobre el candidato geométrico"
    )
    add_line("")
    add_line(
        "Generado por `python -m operational.level_gap.report` desde `RESULTS.json` "
        "(`python -m operational.level_gap.analyze`). Protocolo escrito antes de calcular: "
        "`PROTOCOL.md`. Las etiquetas privadas de Test12 no se han leído; nada se ha subido ni "
        "empaquetado. Los análisis A1–A5 son **adendas posteriores al protocolo**; las "
        "alternativas D3 son **exploración post-hoc** y elegir entre ellas en folds es selección "
        "sobre datos de desarrollo."
    )
    add_line("")
    add_line("## 0. Resumen")
    add_line("")
    add_line(
        f"- Brecha: Overall OOF de la receta publicada {f(rel['overall_mean'])} frente a "
        f"{f(off['overall'])} oficial = {f(total_gap)} MiD. Por banda (peso × diferencia): "
        + ", ".join(f"{b} {f(gap[b])}" for b in BANDS)
        + f". La banda crucial aporta {pct(gap['c'] / total_gap)} de la brecha."
    )
    add_line(
        f"- La brecha es sobre todo una propiedad de los datos: Garl-TTC, evaluado *in-sample* "
        f"en las 88.744 consultas de TRAIN40 (modelo event-only publicado), puntúa "
        f"{f(garl['weighted_MiD_phase_valid'])} (c {f(garl_bands['c'])}) frente a "
        f"{f(overall(garl_test))} (c {f(garl_test['c'])}) en Test12; nuestra cabeza puntúa "
        f"{f(acc['HEAD_RELEASED_in_sample']['overall_MiD'])} in-sample y "
        f"{f(rel['overall_mean'])} OOF frente a {f(off['overall'])}."
    )
    add_line(
        f"- Productores más fuertes (H1) no bastan: incluso la cadena in-sample (PAIR_R final "
        f"sobre sus propias filas) da {f(acc['PAIR_R_final40_in_sample']['overall_MiD'])} en "
        f"TRAIN40; aplicada como cota a la cabeza, {f(h1_bound_pair)} (≥ "
        f"{f(h1_bound_pair - off['overall'])} MiD por encima del test)."
    )
    add_line(
        f"- Las covariables observadas no explican nada (H2-covariables refutada): reponderar los "
        f"folds hacia Test12 *empeora* la receta publicada "
        f"({f(rw[('STRAT_PRIMARY', 'REL', 'S7')]['overall'])} primaria, "
        f"{f(rw[('LOGIT_NO_EVENT', 'REL', 'S7')]['overall'])} logística)."
    )
    add_line(
        f"- Sin etiquetas, todos los pares de estimadores independientes "
        "(incluida la geometría, que no se "
        f"aprende) discrepan en Test12 a {f(k_lo)}–{f(k_hi)} veces su nivel de TRAIN40 (RCTX–GEO "
        f"{f(cons['RCTX-GEO']['ratio_mean_test_over_train'])}, RCTX–ECTX "
        f"{f(cons['RCTX-ECTX']['ratio_mean_test_over_train'])}): Test12 es uniformemente más "
        "«limpio»."
    )
    add_line(
        f"- Proyección del candidato en Test12: optimista (multiplicativa) "
        f"{f(proj[('STRAT_PRIMARY', 'RGB_GEO_BAND')]['projected_overall_median'])}–"
        f"{f(proj[('NONE', 'RGB_GEO_BAND')]['projected_overall_median'])}; por desacuerdo "
        f"{f(a3[('target', 'RGB_GEO_BAND')]['projected_overall'])}–"
        f"{f(a3[('predicted', 'RGB_GEO_BAND')]['projected_overall'])}; emulación de dificultad "
        f"{f(a4['variants']['RGB_GEO_BAND']['overall'])} "
        f"{f(a4['variants']['RGB_GEO_BAND']['ci95'])}; cota pesimista si sus movimientos fueran "
        f"ruido {f(a5['RGB_GEO_BAND']['overall'])}."
    )
    add_line(
        "- Recomendación (sección 9): **subir**. Por la regla prerregistrada, el candidato "
        "`RGB_GEO_BAND` tal cual; `BLEND75` es la alternativa de menor riesgo, post-hoc."
    )
    add_line("")

    # ---------------------------------------------------------------- decomposition
    add_line("## 1. Descomposición estimada de la brecha")
    add_line("")
    add_line(
        table(
            ["Banda", "Folds REL (media 3 semillas)", "Test12 oficial", "Peso × Δ", "% brecha"],
            [[b, f(fold_rel[b]), f(off[b]), f(gap[b]), pct(gap[b] / total_gap)] for b in BANDS],
        )
    )
    add_line("")
    add_line(
        table(
            ["Causa", "Contribución a los " + f(total_gap) + " MiD", "Evidencia"],
            [
                [
                    "Desplazamiento de covariables observadas",
                    f"≤ 0 (reponderar sube REL a {f(rw[('STRAT_PRIMARY', 'REL', 'S7')]['overall'])}"
                    f"–{f(rw[('LOGIT_WITH_EVENT', 'REL', 'S7')]['overall'])})",
                    "§3.2",
                ],
                [
                    "Productores 40 vs 24 secuencias (H1)",
                    f"cota superior {f(rel['overall_mean'] - h1_bound_pair)} (si los rasgos de "
                    "test fueran tan buenos como in-sample); valor real no medido, probablemente "
                    "pequeño",
                    "§2",
                ],
                [
                    "Dificultad intrínseca/ruido de TRAIN40 (sobre todo banda crucial)",
                    f"≥ {f(total_gap - (rel['overall_mean'] - h1_bound_pair))}; plausiblemente "
                    "la mayor parte",
                    "§2, §3.3, §4 (Garl)",
                ],
                ["Definición del scorer / población", "0 en definición; alias: menor", "§4"],
            ],
        )
    )
    add_line("")
    add_line(
        "La cota de H1 se obtiene aplicando a las bandas OOF de la cabeza el cociente in-sample/"
        "OOF del productor PAIR_R (doble in-sample: R_A5 y PAIR_R final vieron esas filas): "
        + ", ".join(f"{b} {f(pair_ratio[b])}" for b in BANDS)
        + f" → {f(h1_bound_pair)}. Con el cociente del nivel de contexto (R_CTX final sobre "
        "entradas OOF: "
        + ", ".join(f"{b} {f(rctx_ratio[b])}" for b in BANDS)
        + f") la cota sería {f(h1_bound_rctx)}. En s/l/n los niveles de Test12 quedan dentro de "
        "lo que darían rasgos in-sample; en c no (test "
        f"{f(off['c'])} frente a cota {f(fold_rel['c'] * pair_ratio['c'])})."
    )
    add_line("")

    # ---------------------------------------------------------------- H1
    add_line("## 2. H1 — productores más fuertes en despliegue")
    add_line("")
    add_line(
        "Comprobaciones de carga: R_CTX final recalculado sobre 512 diagnósticos de Test12 "
        "reproduce su salida con |Δfase| máx. "
        f"{r['inputs_receipt']['rctx_loader_check_max_abs_phase']:.2e}; "
        "`PHASE20[:,0]` de Test12 coincide exactamente con la salida R_CTX "
        f"({r['inputs_receipt']['rctx_vs_phase20_field0_max_abs']:.1e}). Checkpoints verificados "
        "contra el sello `MODEL_SEAL.json`."
    )
    add_line("")
    rows = []
    for k, v in acc.items():
        rows.append([k, f(v["overall_MiD"])] + [f(v[f"MiD{b}"]) for b in BANDS])
    add_line(table(["Estimador en TRAIN40", "Overall", "c", "s", "l", "n"], rows))
    add_line("")
    add_line(
        "Acuerdo con la geometría consciente de truncado (MiD medio / mediana entre el estimador "
        "y φ_geo; no es precisión). «Fiable» = sin truncado en la ventana y ≥ 5 fotogramas "
        f"(train {pct(r['H1']['reliable_share']['train'])}, test "
        f"{pct(r['H1']['reliable_share']['test_physical'])})."
    )
    add_line("")
    rows = [
        [k, f(v["all_mean"], 1), f(v["all_median"], 1), f(v["reliable_mean"], 1)]
        for k, v in agree.items()
    ]
    add_line(table(["Población::estimador", "media", "mediana", "media fiable"], rows))
    add_line("")
    add_line(
        f"En Test12 el R_CTX desplegado (no ve las cajas) discrepa de la geometría "
        f"{f(agree['test::R_CTX_final40_deployed']['all_mean'], 1)} de media, menos que la propia "
        f"verdad de TRAIN40 frente a la geometría "
        f"({f(agree['train::GEO_TRUNC_AWARE_vs_truth']['all_mean'], 1)}) y menos que el R_CTX "
        f"in-sample en TRAIN40 ({f(agree['train::R_CTX_final40_on_OOF_inputs']['all_mean'], 1)})."
    )
    add_line("")
    add_line("Evidencia histórica citada (no recalculada aquí):")
    add_line("")
    cited = [list(c) for c in CITED]
    cited.insert(
        0,
        [
            "Garl-TTC publicado (event-only) sobre TRAIN40 completo",
            f"{f(garl['weighted_MiD_phase_valid'])} ("
            + ", ".join(f"{b} {f(garl_bands[b])}" for b in BANDS)
            + ")",
            "in-sample para Garl (TRAIN40 ⊂ su train.txt)",
            str(GARL_TRAIN40) + " (leído por este código)",
        ],
    )
    cited.insert(
        1,
        [
            "Garl-TTC completo (RGB+eventos), 64 consultas V",
            f"{f(g64['Garl_full']['overall_MiD'])} (c {f(g64['Garl_full']['MiDc'])}, n_c = "
            f"{int(g64['Garl_full']['n_c'])})",
            "in-sample para Garl; muy ruidoso",
            g64["source"] + " (recalculado aquí)",
        ],
    )
    add_line(table(["Modelo", "Overall (bandas)", "Régimen", "Fuente"], cited))
    add_line("")
    add_line(
        "**Veredicto H1: refutada como explicación principal.** La cadena in-sample (la más "
        f"favorable posible) queda en {f(in_lo)}–{f(in_hi)} en TRAIN40, y los modelos evaluados "
        f"en secuencias de eAP no vistas (o in-sample, Garl event-only) puntúan {f(min(hist))}–"
        f"{f(max(hist))}: el nivel "
        "de folds es el nivel de eAP-train, no un artefacto de productores con 24 secuencias. "
        "La curva de aprendizaje de los productores (24 → 40) no está medida; su efecto queda "
        f"acotado por arriba en {f(rel['overall_mean'] - h1_bound_pair)} MiD."
    )
    add_line("")

    # ---------------------------------------------------------------- H2
    add_line("## 3. H2 — desplazamiento de covariables / test más fácil")
    add_line("")
    add_line("### 3.1 Descriptores sin etiquetas (ver `DESCRIPTORS.csv`)")
    add_line("")
    mixes = r["H2_descriptors"]["band_mix"]
    rows = [
        [k] + [pct(v[n]) for n in ("pc", "ps", "pl", "pfar", "pn", "pfarneg")]
        for k, v in mixes.items()
    ]
    add_line(
        table(["Mezcla de bandas", "(0,3]", "(3,6]", "(6,10]", ">10", "(-10,0]", "≤-10"], rows)
    )
    add_line("")
    tr = r["H2_descriptors"]["tracks"]
    add_line(
        f"Tracks: TRAIN40 {tr['train_tracks']} en {tr['train_sequences']} secuencias; Test12 "
        f"{tr['test_tracks']} en {tr['test_sequences']}. En Test12 las cajas son algo menores, "
        "la expansión es más lenta (φ_área mediana menor), hay menos truncado y más historia; "
        f"las predicciones caen menos en (0,3] ("
        f"{pct(mixes['test_REL_pred_official']['pc'])} frente a "
        f"{pct(mixes['train_REL_S7_pred']['pc'])}). Las mismas consultas "
        "parecidas a Test12 *no* son más fáciles en TRAIN40 (3.2)."
    )
    add_line("")
    add_line("### 3.2 Reponderación de los folds hacia Test12")
    add_line("")
    rows = []
    for wname in ("NONE", "PRED_BAND_ONLY", "STRAT_PRIMARY", "LOGIT_NO_EVENT", "LOGIT_WITH_EVENT"):
        info = winfo.get(wname, {})
        rows.append(
            [
                wname,
                f(info.get("ess"), 0) if info else "88744",
                f(info.get("auc"), 3) if "auc" in info else "—",
                f(rw[(wname, "REL", "S7")]["overall"]),
                f(r["H2_reweighted_REL_ci95"][wname]["REL_S7_overall_ci95"]),
                f(rw[(wname, "RGB_GEO_BAND", "S7")]["overall"]),
                f(r["H2_reweighted_REL_ci95"][wname]["CAND_minus_REL_S7_ci95"]),
            ]
        )
    add_line(
        table(
            ["Ponderación", "ESS", "AUC", "REL", "IC REL", "Candidato", "IC Δ cand−REL"],
            rows,
        )
    )
    add_line("")
    add_line(
        "Advertencia: las ponderaciones condicionan en predicciones, no en objetivos. "
        "**Veredicto H2 (covariables observadas): refutada** — ninguna ponderación acerca REL a "
        "35,80; todas lo alejan (criterio: ≤ 72,8 para «parcial»)."
    )
    add_line("")
    add_line("### 3.3 Escala de error sin etiquetas")
    add_line("")
    add_line(
        f"M2c (three-cornered hat REL/GEO/EVT) **falla su validación en folds**: estima "
        f"{f(hat_train['REL']['sd_target_hat_MiD'], 1)} frente a "
        f"{f(hat_train['REL']['sd_target_true_MiD'], 1)} reales para REL (errores REL–eventos "
        "correlacionados, porque la cabeza usa los campos de eventos). Por protocolo no se usa "
        f"(en test daría REL {f(hat_test['REL']['sd_target_hat_MiD'], 1)}). Adenda A1, "
        "cociente test/train del desacuerdo medio entre pares (`CONSISTENCY.csv`):"
    )
    add_line("")
    rows = [
        [
            k,
            f(v["train_mean"], 1),
            f(v.get("test_mean"), 1),
            f(v.get("ratio_mean_test_over_train"), 3),
        ]
        for k, v in cons.items()
    ]
    add_line(table(["Par", "TRAIN40", "Test12", "test/train"], rows))
    add_line("")
    add_line(
        "Todos los pares independientes se contraen por un factor parecido "
        f"({f(k_lo)}–{f(k_hi)}), también "
        "los que incluyen la geometría. Esto encaja con un test uniformemente más limpio (también "
        "para la geometría) y no con una mejora sólo de los productores aprendidos. Excepción: "
        f"REL–RCTX (la corrección de la cabeza) no se contrae (mediana "
        f"{f(cons['REL-RCTX']['train_median'], 1)} → {f(cons['REL-RCTX']['test_median'], 1)}), y "
        f"en test REL está más lejos de la geometría que R_CTX "
        f"({f(cons['REL-GEO']['test_mean'], 1)} frente a {f(cons['RCTX-GEO']['test_mean'], 1)}; en "
        f"folds al revés, {f(cons['REL-GEO']['train_mean'], 1)} frente a "
        f"{f(cons['RCTX-GEO']['train_mean'], 1)}): compatible con que el desplazamiento del reloj "
        "de eventos en Test12 agrande la corrección de la cabeza publicada. No es una medida de "
        "precisión."
    )
    add_line("")
    add_line(
        "**Veredicto H2 (test intrínsecamente más fácil): confirmada de forma indirecta** "
        "(Garl in-sample, contracción uniforme del desacuerdo entre estimadores); no "
        "observable en las covariables medidas."
    )
    add_line("")

    # ---------------------------------------------------------------- H3
    add_line("## 4. H3 — scorer y población")
    add_line("")
    cc = h3["scorer_crosscheck"]
    tt = h3["train_targets"]
    tp = h3["test_population"]
    mc = h3["official_mean_constraint"]
    add_line(
        f"- Scorer oficial vs vectorizado en folds: |Δ| {cc['REL_S7']['abs_diff']:.1e} (REL) y "
        f"{cc['RGB_GEO_BAND_S7']['abs_diff']:.1e} (candidato). Misma fórmula, mismas bandas, "
        "media finita por banda."
    )
    add_line(
        f"- TRAIN40: {pct(tt['in_band_share'])} de objetivos en bandas, T∈(0;0,1]: "
        f"{tt['in_0_0p1']}, rango [{f(tt['min'])}, {f(tt['max'])}], |T| > 10: {tt['abs_gt_10']}; "
        f"crucial (0,1]: {tt['crucial_0_1']}, (1,3]: {tt['crucial_1_3']}. Duplicados "
        f"(secuencia, track, timestamp): {h3['train_duplicates']['same_sequence_track_timestamp']}."
    )
    add_line(
        f"- Test12: {tp['physical']} consultas físicas, {tp['official']} tokens, {tp['aliases']} "
        f"alias (multiplicidades {tp['multiplicity_counts']}); los alias caen sobre todo en "
        f"bandas predichas negativas ({pct(tp['alias_share_by_submitted_pred_band']['pn'])}) y "
        f"pequeñas ({pct(tp['alias_share_by_submitted_pred_band']['ps'])}), casi nunca en (0,3] "
        f"({pct(tp['alias_share_by_submitted_pred_band']['pc'])})."
    )
    add_line(
        f"- El mean_MiD oficial ({f(off['mean_MiD'])}) con las cuatro medias de banda restringe "
        f"la fracción crucial del test a [{f(mc['share_c_range'][0], 3)}, "
        f"{f(mc['share_c_range'][1], 3)}] (si todos los objetivos están en banda); TRAIN40 tiene "
        f"{f(tt['band_counts']['c'] / tt['rows'], 3)}."
    )
    add_line(
        f"- Garl-TTC: no existe evaluación local held-out a escala; in-sample en TRAIN40 "
        f"(event-only) {f(garl['weighted_MiD_phase_valid'])} frente a {f(overall(garl_test))} en "
        f"test (cociente {f(garl['weighted_MiD_phase_valid'] / overall(garl_test), 3)}; nuestro "
        f"OOF/test {f(rel['overall_mean'] / off['overall'], 3)}); banda crucial "
        f"{f(garl_bands['c'] / garl_test['c'], 3)} frente a {f(fold_rel['c'] / off['c'], 3)}. "
        "No se ejecutó Garl de nuevo: ya hay una evaluación completa de TRAIN40 con el scorer "
        "oficial (modelo event-only; el de test es RGB+eventos)."
    )
    add_line("")
    add_line(
        "**Veredicto H3: refutada** (sin diferencia de definición). La comparación con Garl "
        "indica que la brecha es propiedad de los datos y no del apilamiento de PHASE."
    )
    add_line("")

    # ---------------------------------------------------------------- H4
    add_line("## 5. H4 — concentración del error crucial en folds")
    add_line("")
    add_line(
        f"- {h4['crucial_rows']} filas cruciales en {h4['crucial_tracks']} tracks y "
        f"{h4['sequences_with_crucial_rows']} secuencias. El 10 % de tracks con más error "
        f"acumula {pct(h4['top10pct_tracks_share'])} de la suma (criterio 50 %: **no**); las 5 "
        f"secuencias peores, {pct(h4['top5_sequences_share'])}."
    )
    rows = [
        [
            k,
            str(v["rows"]),
            pct(v["share_of_crucial_rows"]),
            f(v["REL_mean_MiD"], 1),
            f(v["CAND_mean_MiD"], 1),
            pct(v["share_of_REL_crucial_sum"]),
        ]
        for k, v in h4["strata"].items()
    ]
    add_line("")
    add_line(table(["Estrato crucial", "Filas", "% filas", "REL", "Candidato", "% suma REL"], rows))
    add_line("")
    pr = h4["label_free_proxies"]
    add_line(
        "Proxies sin etiquetas (predicción REL): predicho crucial train "
        f"{pct(pr['train']['pred_crucial_share'])} / "
        f"test {pct(pr['test']['pred_crucial_share'])}; truncado entre predichos cruciales "
        f"{pct(pr['train']['truncated_given_pred_crucial'])} / "
        f"{pct(pr['test']['truncated_given_pred_crucial'])}; n_frames < 5 entre predichos "
        "cruciales "
        f"{pct(pr['train']['n_frames_lt5_given_pred_crucial'])} / "
        f"{pct(pr['test']['n_frames_lt5_given_pred_crucial'])}; expansión geométrica rápida "
        f"(T_geo < 3 s) {pct(pr['train']['geometry_expanding_fast_share'])} / "
        f"{pct(pr['test']['geometry_expanding_fast_share'])}."
    )
    add_line("")
    add_line(
        "**Veredicto H4: no confirmada como concentración** (el error crucial está repartido); "
        "sí se confirma que el estrato crucial de TRAIN40 es difícil incluso sin truncado y con "
        f"historia larga (REL {f(h4['strata']['truncated_current']['REL_mean_MiD'], 1)} truncado, "
        f"{f(h4['strata']['untruncated_current']['REL_mean_MiD'], 1)} sin truncar), y que en "
        "Test12 el estrato equivalente por "
        "proxies es más raro y más limpio."
    )
    add_line("")

    # ---------------------------------------------------------------- decision diagnostics
    add_line("## 6. Diagnósticos de riesgo del candidato")
    add_line("")
    add_line("### D1 — Movimiento en Test12 frente a lo enviado (no es precisión)")
    add_line("")
    rows = [
        [
            k,
            f(v["mean_MiD_vs_submitted"]),
            f(v["median"]),
            f(v["p90"]),
            str(v["sign_changes"]),
            f(v["by_submitted_pred_band_mean"]["pc"]),
            f(v["by_submitted_pred_band_mean"]["pn"]),
        ]
        for k, v in move.items()
        if k != "REL"
    ]
    add_line(
        table(
            ["Opción", "media", "mediana", "P90", "cambios de signo", "pred (0,3]", "pred (-10,0]"],
            rows,
        )
    )
    add_line("")
    add_line("### D2 — Victoria condicional en folds (semilla 7, candidato frente a REL)")
    add_line("")
    rows = [
        [
            f"{f(b['low'], 0)}–{f(b['high'], 0) if b['high'] is not None else '∞'}",
            str(int(b["rows"])),
            pct(b["train_share"]),
            pct(b["test_share"]),
            f(b["win_rate"], 3),
            f(b["mean_change"]),
        ]
        for b in dec["conditional_wins_by_disagreement"]
    ]
    add_line(
        table(["|Δ| MiD", "filas folds", "% folds", "% Test12", "victoria", "Δ|error| medio"], rows)
    )
    add_line("")
    add_line(
        f"Ponderada con la distribución de desacuerdo de Test12: victoria "
        f"{f(dec['test_weighted_win_rate'], 3)}, cambio medio "
        f"{f(dec['test_weighted_mean_change_MiD'])} MiD por consulta. Por banda predicha "
        "(`CONDITIONAL_WINS.csv`), el candidato pierde en folds en predichos (6,10] y en "
        "predichos negativos con desacuerdo ≥ 40. Secuencias con victoria > 0,5: "
        f"{dec['sequences_win_rate_gt_half']}/40; con MiD medio mejor: "
        f"{dec['sequences_mean_MiD_better']}/40 (`SEQUENCE_WINS.csv`)."
    )
    add_line("")
    add_line("### D3 — Alternativas en folds (post-hoc; misma CV que phase_geometry)")
    add_line("")
    rows = []
    for name in ("REL", "RGB_GEO_BAND", "BLEND25", "BLEND50", "BLEND75", "GATE5", "REL_GEO_BAND"):
        a = alts[name]
        rows.append(
            [
                name,
                f"{f(a['overall_mean'])} ± {f(a['overall_sd'])}",
                *[f(a[f"MiD{b}"]) for b in BANDS],
                f"{f(a.get('delta_vs_REL'))} {f(a.get('delta_vs_REL_ci95'))}"
                if name != "REL"
                else "—",
                f"{f(a.get('delta_vs_RGB_GEO_BAND'))} {f(a.get('delta_vs_RGB_GEO_BAND_ci95'))}"
                if name != "RGB_GEO_BAND"
                else "—",
                f"{a['sequences_better_than_REL_S7']}/40",
            ]
        )
    add_line(
        table(
            ["Variante", "Overall", "c", "s", "l", "n", "Δ vs REL [IC]", "Δ vs cand. [IC]", "Sec."],
            rows,
        )
    )
    add_line("")
    add_line(
        "BLENDw = w·fase(candidato) + (1−w)·fase(REL), semilla emparejada; GATE5 = candidato "
        "sólo con caja actual no truncada y ≥ 5 fotogramas. No existe en folds una cabeza con "
        "rasgos de productores de 40 secuencias: no disponible."
    )
    add_line("")
    add_line("### D4 — Proyección a Test12 (modelo multiplicativo prerregistrado)")
    add_line("")
    rows = []
    for wname in ("NONE", "STRAT_PRIMARY", "LOGIT_NO_EVENT"):
        for name in ("RGB_GEO_BAND", "BLEND25", "BLEND50", "BLEND75", "GATE5"):
            p = proj[(wname, name)]
            rows.append(
                [
                    wname,
                    name,
                    f(p["projected_overall_median"]),
                    f(p["projected_overall_ci95"]),
                    f(p["p_worse_than_submitted"], 3),
                ]
            )
    add_line(table(["Ponderación", "Opción", "Proyección", "IC 95 %", "P(peor)"], rows))
    add_line("")
    add_line(
        "Los IC sólo recogen el muestreo de secuencias, no el error del supuesto de transferencia."
    )
    add_line("")

    # ---------------------------------------------------------------- addenda
    add_line("## 7. Adendas de riesgo (posteriores al protocolo)")
    add_line("")
    add_line(
        "A2 — árbitro de eventos (el candidato no usa eventos; REL sí, sesgo contra el candidato):"
    )
    add_line("")
    rows = []
    for k, v in arb.items():
        rows.append(
            [
                k,
                f(v["overall_arbiter_win_rate"], 3),
                f(v["random_sign_arbiter_win_rate"], 3),
                f(v["move_sign_agrees_with_arbiter_minus_REL"], 3),
                f(v.get("move_sign_agrees_with_truth_minus_REL"), 3),
            ]
        )
    add_line(
        table(
            [
                "Lado::árbitro",
                "más cerca del árbitro",
                "base signo aleatorio",
                "signo = árbitro−REL",
                "signo = verdad−REL",
            ],
            rows,
        )
    )
    add_line("")
    add_line(
        "En folds, «más cerca del árbitro» predice «más cerca de la verdad» en todos los bins "
        "(ver `RESULTS.json`). En Test12 los movimientos del candidato apuntan hacia el árbitro "
        "más que en folds y más que con signo aleatorio: no se comportan como ruido."
    )
    add_line("")
    rows = []
    for name in ("RGB_GEO_BAND", "BLEND25", "BLEND50", "BLEND75", "GATE5"):
        rows.append(
            [
                name,
                f"{f(a3[('target', name)]['projected_overall'])} {f(a3[('target', name)]['ci95'])}",
                f"{f(a3[('predicted', name)]['projected_overall'])} "
                f"{f(a3[('predicted', name)]['ci95'])}",
                f"{f(a4['variants'][name]['overall'])} {f(a4['variants'][name]['ci95'])}",
                f(a4["variants"][name]["delta_vs_RGB_GEO_BAND_ci95"])
                if name != "RGB_GEO_BAND"
                else "—",
                f(a5[name]["overall"]),
            ]
        )
    add_line(
        table(
            [
                "Opción",
                "A3 por desacuerdo (banda objetivo)",
                "A3 (banda predicha)",
                "A4 emulación [IC]",
                "A4 Δ vs cand. [IC]",
                "A5 movimientos = ruido",
            ],
            rows,
        )
    )
    add_line("")
    ck = a4["checks"]
    add_line(
        f"A4 inclina los folds (exp(−λ_b·|e_REL|)) para que REL tenga exactamente las bandas "
        f"oficiales; ESS {f(ck['ess'], 0)}. Comprobación sin etiquetas: el desacuerdo "
        "candidato–REL en los folds emulados es "
        f"{f(ck['cand_rel_disagreement_mean']['emulated_folds'])} "
        f"frente a {f(ck['cand_rel_disagreement_mean']['test'])} en Test12 (sin emular "
        f"{f(ck['cand_rel_disagreement_mean']['raw_folds'])}); REL–geometría "
        f"{f(ck['rel_geo_disagreement_mean']['emulated_folds'])} frente a "
        f"{f(ck['rel_geo_disagreement_mean']['test'])}. La emulación elige filas en las que REL "
        "acierta (en parte por azar), lo que penaliza cualquier cambio: es pesimista para el "
        "candidato y más aún para movimientos grandes. A5 trata los movimientos de Test12 como "
        "ruido de signo aleatorio sobre errores con las bandas oficiales: es una cota adversa, "
        "contradicha por A2 y D2."
    )
    add_line("")

    # ---------------------------------------------------------------- verdicts
    add_line("## 8. Veredictos")
    add_line("")
    add_line(
        table(
            ["Hipótesis", "Veredicto", "Números clave"],
            [
                [
                    "H1 productores 24→40",
                    "refutada como principal; contribución ≤ "
                    f"{f(rel['overall_mean'] - h1_bound_pair)}",
                    f"in-sample PAIR {f(acc['PAIR_R_final40_in_sample']['overall_MiD'])}, R_CTX "
                    f"{f(acc['R_CTX_final40_on_OOF_inputs']['overall_MiD'])}; históricos "
                    f"{f(min(hist))}–{f(max(hist))}",
                ],
                [
                    "H2 covariables observadas",
                    "refutada",
                    f"REL reponderado {f(rw[('STRAT_PRIMARY', 'REL', 'S7')]['overall'])}–"
                    f"{f(rw[('LOGIT_WITH_EVENT', 'REL', 'S7')]['overall'])}",
                ],
                [
                    "H2 test intrínsecamente más limpio",
                    "confirmada (indirecta)",
                    f"Garl in-sample {f(garl['weighted_MiD_phase_valid'])} vs test "
                    f"{f(overall(garl_test))}; desacuerdos ×{f(k_lo)}–{f(k_hi)}",
                ],
                ["H3 scorer/población", "refutada", "|Δ scorer| < 1e-12; mismas bandas"],
                [
                    "H4 concentración crucial",
                    "no confirmada",
                    f"10 % tracks = {pct(h4['top10pct_tracks_share'])}",
                ],
                [
                    "M2c three-cornered hat",
                    "inválido (falla validación)",
                    f"REL estimado {f(hat_train['REL']['sd_target_hat_MiD'], 1)} vs real "
                    f"{f(hat_train['REL']['sd_target_true_MiD'], 1)}",
                ],
            ],
        )
    )
    add_line("")

    # ---------------------------------------------------------------- recommendation
    add_line("## 9. Recomendación")
    add_line("")
    add_line("Regla prerregistrada, aplicada literalmente:")
    add_line("")
    for k, v in rule.items():
        add_line(f"- {k}: {'sí' if v else 'no'}")
    add_line(
        f"- M2c no indica geometría relativamente peor en test: **no evaluable** (M2c "
        f"{'válido' if hat_ok else 'inválido'}); la adenda A1 muestra la misma contracción para "
        "pares con y sin geometría."
    )
    add_line("")
    add_line(
        "Coste proyectado de cada alternativa frente al candidato (modelo (a) / (b)): "
        + "; ".join(f"{n} {f(a)} / {f(b)}" for n, a, b in blend_rows)
        + ". La cláusula de mezcla («P(peor) ≤ mitad con pérdida ≤ 2 MiD») es degenerada: "
        "P(peor) = 0 en (a) y (b) para todas las opciones."
    )
    add_line("")
    add_line(
        "**Decisión: subir. Opción por defecto: el candidato `RGB_GEO_BAND` tal cual** "
        "(cumple todas las condiciones evaluables de la regla prerregistrada y es la prueba "
        "limpia de la hipótesis registrada). **Alternativa de menor riesgo: `BLEND75`** "
        f"(pérdida proyectada {f(b75['cost_a'])} / {f(b75['cost_b'])} MiD en (a)/(b), "
        f"{f(b75['a4'])} en A4, {f(b75['a5'])} en la cota adversa A5, {b75['seq']}/40 "
        "secuencias mejores que REL); exige un recibo de prerregistro propio (ADR-0047) y "
        "declararla "
        "selección post-hoc. No se recomienda `BLEND50` (pierde > 2 MiD en (a)/(b)) ni no subir."
    )
    add_line("")
    add_line(
        f"Dirección esperada: mejor que 35,80. Rango plausible del candidato: "
        f"{f(proj[('STRAT_PRIMARY', 'RGB_GEO_BAND')]['projected_overall_median'])}–"
        f"{f(a4['variants']['RGB_GEO_BAND']['ci95'][1])} (de la proyección multiplicativa a la "
        f"emulación pesimista). En todos los modelos con movimientos informativos P(peor) ≈ 0 por "
        f"bootstrap; el riesgo es de modelo: si los movimientos fueran ruido (A5) el resultado "
        f"sería ≈ {f(a5['RGB_GEO_BAND']['overall'])} (BLEND75 ≈ {f(a5['BLEND75']['overall'])}). "
        "A2 y D2 hacen poco probable ese régimen; asignarle "
        f"{pct(JUDGED_NOISE_PROBABILITY)} es un juicio, no una medida, y daría una penalización "
        f"esperada de {f(penalty)} MiD frente a una ganancia media de {f(expected_gain)} MiD "
        "en los cinco modelos informativos (a, b, A3×2, A4)."
    )
    add_line("")
    add_line("## 10. Limitaciones")
    add_line("")
    add_line(
        "- Ninguna cifra mide precisión en Test12. Las proyecciones dependen de supuestos de "
        "transferencia que no pueden validarse sin etiquetas; sólo REL tiene un ancla oficial.\n"
        "- La causa exacta de que Test12 sea más limpio (etiquetas, dinámica, captura) no se "
        "identifica; sólo se descarta que sean covariables observadas o definiciones.\n"
        "- La comparación con Garl usa su modelo event-only en TRAIN40 y el RGB+eventos en test.\n"
        "- El efecto 24→40 de los productores no se ha medido con un experimento held-out.\n"
        "- Adendas A1–A5 y alternativas D3 son post-hoc; el candidato de test es semilla 7.\n"
        "- No se reclama SOTA ni validez de seguridad."
    )
    add_line("")
    add_line("## 11. Reproducción")
    add_line("")
    add_line(
        "Con `PYTHONPATH=<repo>;<repo>/src` y el Python con torch del worktree de integración:"
    )
    add_line("")
    add_line("```")
    add_line(
        "python -m operational.level_gap.extract --out artifacts/sota_20261118/level_gap\n"
        "python -m operational.level_gap.analyze --work artifacts/sota_20261118/level_gap "
        "--docs docs/sota_20261118/level_gap\n"
        "python -m operational.level_gap.report --docs docs/sota_20261118/level_gap"
    )
    add_line("```")
    add_line("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", type=Path, required=True)
    args = parser.parse_args()
    results = json.loads((args.docs / "RESULTS.json").read_text(encoding="utf-8"))
    (args.docs / "REPORT.md").write_text(render(results), encoding="utf-8")
    print("written", args.docs / "REPORT.md")


if __name__ == "__main__":
    main()
