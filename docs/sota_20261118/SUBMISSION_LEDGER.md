# Ledger de envíos al servidor GarlTTC Test

Registro acumulado de todas las subidas de la cuenta (ver
[ADR-0047](../decisions/ADR-0047-garlttc-submission-discipline.md)). Una fila por
subida, incluidas las fallidas, privadas o retiradas. Los campos no conocidos se
escriben `desconocido`; no se rellenan por inferencia. Cada fila nueva debe tener
antes un recibo `SUBMISSION_RECEIPT_<variante>.json` commiteado en esta carpeta.

| # | Variante | ID CodaBench | Fecha y hora de subida (zona) | Recibo de preregistro | SHA-256 del paquete subido | Commit de código | Receta (SHA-256) / semilla | Hipótesis | Qué se informará | Estado | Overall MiD | Notas |
|---:|---|---:|---|---|---|---|---|---|---|---|---:|---|
| 1 | PHASE_ONLY_EVENT_S7_TRAIN40_SCIENTIFIC_OOF | 1016512 | 2026-10-28 17:15 (zona desconocida; dato aportado por el propietario) | ninguno (anterior a ADR-0047) | desconocido; candidato inferido `5eb69749de7aff682eea640d843ee635379702337d929c86f50472264f1d88bc`, no confirmado por el servidor | desconocido; código commiteado después en `2ad7a057` | `OOF_RECIPE.json` (`recipe_sha256` interno `78c83c09…`) / semilla 7 | desconocido (no preregistrada) | desconocido (no preregistrado) | puntuado | 35.80145 | Ver ADR-0046. Único envío de la cuenta según declaración del propietario (2026-11-18); el candidato V16 `60b5a620…` no se subió. |
| 2 | RGB_GEO_BAND_S7_TRAIN40 | pendiente | no subido (preparado el 2026-11-19) | [SUBMISSION_RECEIPT_RGB_GEO_BAND_S7_TRAIN40.json](SUBMISSION_RECEIPT_RGB_GEO_BAND_S7_TRAIN40.json) | `8590f1151bb8b8db5baa5ae364675356a47ca93a3ee58139421f3d80d0f831df` | `3f9108a9` | `geometry/PROTOCOL.json` / semilla 7 | La geometría causal de las regiones suministradas, con pérdida por banda y sin campos de eventos, baja el Overall MiD del test por debajo de 35,80 | El resultado, sea cual sea, incluido si es peor que 35,80 | **preparado, no subido** | — | Paquete en `geometry/PHASE_GEOMETRY_RGB_GEO_BAND_S7_TRAIN40.zip`. Antes de subir: comprobar si el leaderboard conserva la mejor entrada o la última. Tras subir: completar ID, fecha y puntuación en esta fila. |

## Total de subidas de la cuenta

1 realizada (declaración del propietario del 2026-11-18) y 1 preparada sin subir (ADR-0046, pregunta 2). No
verificable de forma independiente: el leaderboard muestra una entrada por
participante.
