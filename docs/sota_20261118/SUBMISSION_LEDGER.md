# Ledger de envíos al servidor GarlTTC Test

Registro acumulado de todas las subidas de la cuenta (ver
[ADR-0047](../decisions/ADR-0047-garlttc-submission-discipline.md)). Una fila por
subida, incluidas las fallidas, privadas o retiradas. Los campos no conocidos se
escriben `desconocido`; no se rellenan por inferencia. Cada fila nueva debe tener
antes un recibo `SUBMISSION_RECEIPT_<variante>.json` commiteado en esta carpeta.

| # | Variante | ID CodaBench | Fecha y hora de subida (zona) | Recibo de preregistro | SHA-256 del paquete subido | Commit de código | Receta (SHA-256) / semilla | Hipótesis | Qué se informará | Estado | Overall MiD | Notas |
|---:|---|---:|---|---|---|---|---|---|---|---|---:|---|
| 1 | PHASE_ONLY_EVENT_S7_TRAIN40_SCIENTIFIC_OOF | 1016512 | 2026-10-28 17:15 (zona desconocida; dato aportado por el propietario) | ninguno (anterior a ADR-0047) | desconocido; candidato inferido `5eb69749de7aff682eea640d843ee635379702337d929c86f50472264f1d88bc`, no confirmado por el servidor | desconocido; código commiteado después en `2ad7a057` | `OOF_RECIPE.json` (`recipe_sha256` interno `78c83c09…`) / semilla 7 | desconocido (no preregistrada) | desconocido (no preregistrado) | puntuado | 35.80145 | Ver ADR-0046. Único envío de la cuenta según declaración del propietario (2026-11-18); el candidato V16 `60b5a620…` no se subió. |

## Total de subidas de la cuenta

1 — declaración del propietario del 2026-11-18 (ADR-0046, pregunta 2). No
verificable de forma independiente: el leaderboard muestra una entrada por
participante.
