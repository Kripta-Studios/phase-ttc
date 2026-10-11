# Correcciones de la auditoría swarm — 2026-10-11

Estas correcciones están en una rama y worktree aislados. La campaña original
congela todo `src` mediante hashes y otra sesión implementa el cargador RGB.
No se han modificado sus procesos, pesos, freezes ni optimizaciones.

## Métricas y selección

EvTTC/Table VI tiene un scorer RTE separado: `100*abs(P-T)/abs(T)`, media sin
pesos eAP. Las predicciones cero o negativas se contabilizan; NaN/Inf invalidan
el agregado. El scorer comprueba identidad, duplicados y cobertura de los tres
escenarios. Una secuencia necesita asociación explícita con su escenario.

`data/protocols/evttc_author_20261011.json` registra las aclaraciones de Hongxin.
Los helpers de `data/evttc_author_protocol.py` implementan matching TTC vecino
más cercano con tolerancia 0,05 s, continuidad por centro y ROI de cuatro esquinas
mediante una transformación calibrada suministrada por el llamante. No sustituyen
la geometría histórica ni inventan una transformación o población oficial.
La lista exhaustiva de queries y el SHA del checkpoint de Table VI siguen sin
estar verificados. Test12 sigue requiriendo evaluación oficial externa.

El scorer firmado eAP conserva sus medias diagnósticas sobre valores finitos,
pero una banda ausente invalida el agregado ponderado. Contabiliza targets fuera
del dominio y filas inválidas (unión de fallo oficial y MiD no finito). La selección
exige contabilidad completa y ordena por tasa de filas inválidas, después por MiD.
`strict_metric_coverage` identifica evidencia sin filas inválidas; no impide
conservar un endpoint fixed-last con fallos declarados. Las métricas legacy sin
contadores necesitan recalcularse desde predicciones, sin reentrenar.

## Incertidumbre

`models/temporal_uncertainty.py` añade una revisión explícita de inferencia con
ambos jacobianos, transporte y covarianza. Usa por defecto una cota conservadora;
el modo independiente es una hipótesis explícita, no una covarianza medida.
Se exponen saturación y validez. El adaptador exige umbrales y clipping del
checkpoint y verifica paridad de las estimaciones puntuales.

Ejemplo para un archivo de salidas por par ya congelado:

```powershell
python -m operational.uncertainty_revision.replay --input pair_outputs.npz --output revised.npz --current-pair-weight 0.75 --ttc-clip-seconds 60 --risk-thresholds-s 0.5 1 2 4
```

Los cuatro arrays requeridos son `pair_log_ratio`, `pair_log_variance`,
`pair_known` y `pair_delta_t_s`, de forma `[batch,pairs]`; `ratio_covariance`
opcional tiene forma `[batch]`. Consultar los parámetros reales del checkpoint.
Las nuevas incertidumbres no son compatibles silenciosamente con PHASE17,
normalizadores o cabezas antiguas. Su uso exige una revisión de features y
calibración con H; cualquier reajuste debe limitarse a las cabezas necesarias.
No hay todavía una mejora medida de exactitud o calibración.

## V14

El worktree V14 también refuerza identidad token/grupo, ascendencia de cachés,
recibos de evaluación y estados de traza por intento, y añade transferencia de
endpoints congelados. Su contrato se documenta en
`operational/range_transfer/TRANSFER_MANIFEST.md` dentro de V14.
Las cachés antiguas incompatibles deben conservarse y recalcularse en una salida
versionada; no cambiar un freeze ni resetear los presupuestos de una campaña.

## Validación e integración

La carpeta superior `swarm-fixes-20261011/validation` contiene las pruebas JUnit,
el recálculo RTE de predicciones locales y la prueba CPU de carga del checkpoint
Event A5 real. Esta última usa entrada sintética y comprueba pesos/estimaciones
puntuales intactos; no mide exactitud. `validate_fixes.ps1` repite las suites
focales CPU de los tres worktrees. No se lanza un reentrenamiento largo.

Antes de integrar en una campaña futura, conservar los freezes anteriores y
registrar la nueva revisión de código y sus salidas. No copiar estos archivos
encima del árbol activo con un supervisor que verifica hashes. Las mejoras RGB
pertenecen a la otra sesión confirmada por el usuario.
