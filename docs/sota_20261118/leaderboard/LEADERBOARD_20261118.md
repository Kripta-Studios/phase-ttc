# Leaderboard GarlTTC Test — instantánea del 2026-11-18

**Procedencia:** tabla pegada por el propietario del repositorio el 2026-11-18.
**No** ha sido descargada ni consultada por ninguna herramienta; no se ha hecho
ninguna llamada a la API de CodaBench ni ninguna subida. La zona horaria de las
fechas no está verificada. Valores con dos decimales, tal como se muestran.
Versión legible por máquina: [LEADERBOARD_20261118.json](LEADERBOARD_20261118.json).

| Rank | Participante | Fecha | ID | Event | Image | Método | Overall | MiDc | MiDs | MiDl | MiDn | FRc/FRs/FRl/FRn | Muestras |
|---:|---|---|---:|---|---|---|---:|---:|---:|---:|---:|---|---:|
| 1 | AlvaroSchwiedop | 2026-10-28 17:15 | 1016512 | true | true | PHASE-TTC (TRAIN40, 650 ms causal history) | 35.80 | 43.52 | 28.26 | 25.72 | 29.93 | 0.0 / 0.0 / 0.0 / 0.0 | 6762 |
| 2 | nail-hnu | 2026-06-28 16:02 | 817519 | true | true | Garl-TTC | 45.03 | 53.13 | 37.58 | 40.61 | 31.29 | 0.0 / 0.0 / 0.0 / 0.0 | 6762 |
| 3 | ahmed2204a | 2026-10-09 00:39 | 970318 | n/a | n/a | n/a | 45.03 | 53.13 | 37.59 | 40.61 | 31.28 | 0.0 / 0.0 / 0.0 / 0.0 | 6762 |

## Comparación con el archivo del 28 de octubre

Archivo: `paper/from_motion_to_time_20261102/data/official_benchmark.json`
(`as_of: 2026-10-28`, fuente B01 = API de leaderboard, SHA-256 original
`0c7114c5…`).

- Siguen siendo **tres** entradas, en el **mismo orden** (1016512, 817519, 970318)
  y con los mismos IDs.
- Todas las métricas y tasas de fallo coinciden a dos decimales, incluida la
  entrada 1016512 (archivo a precisión completa: overall 35,80145, MiDc 43,519,
  MiDs 28,257, MiDl 25,718, MiDn 29,931).
- Diferencias de contenido, no de valores:
  - el archivo no guarda nombres de participante ni fechas; esta instantánea los
    añade según el propietario;
  - el archivo etiqueta 970318 como «Unnamed public submission»; aquí aparece el
    participante `ahmed2204a`, con método y modalidades `n/a`;
  - el archivo sólo registra `uses_event`/`uses_image` para 1016512; aquí se
    añaden `true/true` para 817519.
- Observación: la entrada 970318 reproduce los valores de Garl-TTC salvo en
  MiDs (+0,01) y MiDn (−0,01). Su método no está declarado; no se infiere nada
  sobre él.

Esta instantánea no es una nueva evaluación ni cambia ninguna afirmación del
manuscrito. Rank 1 en este leaderboard no equivale a estado del arte; ver
[ROADMAP](../roadmap/ROADMAP.md).
