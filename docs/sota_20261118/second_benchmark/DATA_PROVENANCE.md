# Procedencia de datos y descargas — segundo benchmark (2026-11-18)

Todas las descargas de esta campaña están en `D:\PHASE_TTC_BENCH2_20261120\` (29,7 MB en total incluida la extracción del ZIP,
muy por debajo del límite de 60 GB). Se tratan como datos no confiables: no se ha ejecutado ningún
código descargado; los ficheros `.py`/`.ipynb` del repositorio TSTTC sólo se han leído como texto.
No se ha enviado nada del proyecto a ningún servicio; no se han creado cuentas, iniciado sesiones,
enviado formularios ni subido nada.

## 1. Descargado en esta campaña

| Fichero (bajo `D:\PHASE_TTC_BENCH2_20261120\`) | Origen | Licencia | Bytes | SHA-256 |
|---|---|---|---|---|
| `tsttc_paper\2309.01539v3.pdf` | https://arxiv.org/pdf/2309.01539v3 | licencia arXiv no exclusiva (artículo); datos CC BY-NC-SA según su apéndice A | 1.881.750 | `cb7587ab8afcea58a48a38ad227625f4a80b5ab9bc977254f03a6f1e7c4809a7` |
| `tsttc_repo_meta\TSTTC-main.zip` | https://codeload.github.com/tusen-ai/TSTTC/zip/refs/heads/main (commit `b59e7a1d07995eb8ce9d29eda239edbb769dc1ec`, último push 2024-05-16) | **sin licencia declarada** en GitHub (`license: null`): sólo lectura, no se reutiliza código | 7.501.387 | `b3c98a61382d4a6b5b3534395cc82266b78e46b006dcab755cf2550fa2547db5` |
| `tsttc_repo_meta\baidu_share_page.html` | GET de https://pan.baidu.com/s/14oHBEDytYZh6C3ah38a8zw (redirige a `share/init`, página "请输入提取码", `loginstate:0`) | — | 16.020 | `e89a0d33b6b1cde72413e71c131f09f9623ca6cfc49ee356259a3855bdc8ad48` |
| `tsttc_repo_meta\extracted\TSTTC-main\…` | extracción local del ZIP anterior (incluye binarios `.o`/`.egg` precompilados que no se han ejecutado y un checkpoint `weights/Deep_TTC.pth` de 711.747 bytes que no se ha cargado) | — | — | — |
| `FCWD_LOCAL_SHA256.tsv` | hashes calculados localmente de `E:\FCWD_dataset` (tabla 2) | — | — | — |

Intentos sin descarga: `https://open-dataset.tusen.ai/TSTTC` → DNS ENOTFOUND (2026-11-18);
`raw.githubusercontent.com/tusen-ai/TSTTC/main/README.md` → 404 (el fichero se llama `readme.md`;
se leyó desde el ZIP). Búsqueda de espejos (Hugging Face, Kaggle, Google Drive): ninguno.

**TSTTC no se ha descargado.** El único canal de distribución publicado es Baidu Netdisk (código
de extracción público `rdh3`): requiere introducir el código en un formulario y una cuenta Baidu
iniciada (y su cliente para ficheros grandes). Se detuvo esa vía. Tamaño de descarga: no publicado
en ninguna fuente accesible.

## 2. Datos locales usados (no descargados en esta campaña)

FCWD (Event-Aided TTC, ECCV 2024; proyecto https://nail-hnu.github.io/EventAidedTTC/, código
https://github.com/NAIL-HNU/event_aided_ttc @ `79ff0842955304ec4f6164ec09baddc71386d225`), adquirido
por la campaña `docs/sota_campaign_20261008` (origen en `operational/sota_eval/fcwd_assets.py`;
ese módulo no registra una licencia de los datos y esta campaña no la ha verificado: **licencia
FCWD no establecida**, uso sólo para investigación interna hasta confirmarla). Raíz local leída vía `PHASE_TTC_FCWD_ROOT`.

| Fichero | Bytes | SHA-256 | Uso en esta campaña |
|---|---|---|---|
| `sealed_labels/sequence_1/bounding_box.csv` | 6.596 | `936fcaa0115ff1e7dbe09bedf472d1eba5196e700a4e82c79c1df86e3d4375b4` | entrada (cajas) |
| `sealed_labels/sequence_2/bounding_box.csv` | 6.656 | `be8b5933038609c430b1d796038345280a589874fb6b10a06f5c7bdece16c88c` | entrada |
| `sealed_labels/sequence_3/bounding_box.csv` | 6.094 | `69291c08420cdc6629c05d57c57fa7c7c1fe87f5e00273869e6e106df42823d1` | entrada |
| `sealed_labels/sequence_1/gt_ttc.csv` | 6.913 | `20af46fab5fc4d423f6d6b894b26727b0c4fe26106dd81614fc43f8c6bc4216e` | verdad, tras sellar |
| `sealed_labels/sequence_2/gt_ttc.csv` | 7.077 | `ec1c7769def687d9c92f83685ac495dc92e36c7a489fb9d2d88049fc5090e4c8` | verdad, tras sellar |
| `sealed_labels/sequence_3/gt_ttc.csv` | 7.395 | `97761ceca733a65e3057ef2beebd28cbf7abc8a690bd20603c3a6fc177d8f862` | verdad, tras sellar |
| `inputs/sequence_1/2024-03-02-10-35-27.hdf5` | 2.383.285.966 | `24f13357929c4758df425bf02510e17cbd2fe4cec743b3759c9b6deb29c4bab3` | sólo primer/último `t` de eventos |
| `inputs/sequence_2/2024-03-02-10-37-09.hdf5` | 2.158.720.024 | `d11192014e3fda74777ee7ecf0d7cb4a78417026de66ab8479f7d8dd56214585` | ídem |
| `inputs/sequence_3/2024-03-02-10-38-25.hdf5` | 2.141.646.632 | `d77fe591a7bcd94762ab6625e71e1723e945b6747d86f6a551a8a435f79aca0c` | ídem |

Los SHA-256 de los CSV de cajas coinciden con `REFERENCE_INPUT_CONTRACT.json` y los de `gt_ttc.csv`
con `TARGET_JOIN_CONTRACT.json` de la campaña previa. Los hashes de `gt_ttc.csv` se calcularon sobre
bytes antes de sellar las predicciones (sin interpretar su contenido).

Otros insumos del repositorio (no externos): `artifacts/sota_20261118/geometry/work/train_inputs.npz`
(eAP TRAIN40, prefijo SHA-256 `1af688cf`, verificado por `train_geo.py`; completo en
`GEO_LEARNED_EAP_RECEIPT.json`) y
`docs/sota_campaign_20261008/evidence/fcwd_inference/scoring/SCORED_PREDICTIONS.csv` (anclas,
verdades y predicciones locales previas).

## 3. Publicaciones citadas (sólo lectura)

- TSTTC: Shi et al., arXiv:2309.01539 (v3).
- eAP / Garl-TTC: arXiv:2603.16303 (copia HTML ya presente en
  `docs/sota_campaign_20261008/evidence/official_contract/garl_paper_2603.16303v1.html`).
- FCWD / STRTTC: arXiv:2407.07324 y su código oficial (`metricRee.m`, `plotSimpleResult.m`,
  `main_FCWD.m` leídos en el commit indicado).
