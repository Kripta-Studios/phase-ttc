# Brecha de nivel folds (≈85) frente a Test12 (35,80): protocolo

Escrito el 2026-11-18 (17:15 CET) **antes** de calcular cualquiera de las magnitudes
de abajo. Sólo se habían inspeccionado: nombres y formas de archivos, el scorer oficial,
el informe de geometría (`docs/sota_20261118/geometry/REPORT.md`) y el código de la
cadena final. Las adendas posteriores se marcan como tales en `REPORT.md`.

## Hechos de partida (no se recalculan aquí)

- Cabeza publicada (receta `REL`) en OOF agrupado de TRAIN40 (88.744 consultas, 40
  secuencias): Overall 85,17; c/s/l/n 132,41/39,74/34,65/35,79.
- Servidor Test12 (envío único 1016512): Overall 35,80; c/s/l/n 43,52/28,26/25,72/29,93.
  Referencia Garl-TTC en Test12: 53,13/37,58/40,61/31,29.
- Candidato `RGB_GEO_BAND` en OOF: 41,27; c/s/l/n 51,09/30,11/33,71/33,21.
- Brecha a explicar: 85,17 − 35,80 = 49,37 MiD de Overall (factor 0,42).

## Reglas

- Etiquetas privadas de Test12: nunca se leen ni se buscan. En Test12 sólo se usan
  entradas públicas, predicciones ya calculadas y la puntuación oficial agregada.
- Nada se sube ni se empaqueta. Sin escritura en git.
- Bootstrap: 10.000 remuestreos de secuencias completas, semilla 20261118
  (`operational.phase_geometry.metrics.bootstrap_overall`). Filas = consultas físicas.
- Para lo que importa en Test12 se usan las predicciones de semilla 7 (la del paquete
  candidato); las tablas de folds usan además la media de las tres semillas por fila.
- Todo número del informe sale de `RESULTS.json`/CSV generados por código.

## Hipótesis, medidas y criterios

### H1 — productores más fuertes en despliegue (24 vs 40 secuencias)

Medidas:
- **M1a** (cota in-sample, primer nivel): `PAIR_R` final (ajustado en las 40) aplicado a su
  propia caché de entrenamiento (`final/cache/PAIR_R.npz`, tokens del `R_A5` final, también
  ajustado en las 40) frente a la fase PAIR OOF (`PHASE17_RAW_RGB.expert_phase[:,2]`,
  productores de rotación con 24 secuencias). MiD por banda y Overall en TRAIN40.
- **M1b** (cota in-sample, nivel contexto): `R_CTX` final sobre `final/cache/R_CTX.npz`
  (entradas PHASE17 OOF) frente a `PHASE20[:,0]` OOF. Antes, comprobación de carga: el
  `R_CTX` final aplicado a las entradas de diagnóstico de Test12 debe reproducir
  `PHASE20[:,0]` de Test12 (|Δfase| máx. < 1e-4).
- **M1c** evidencia histórica de curvas de aprendizaje (puntuaciones held-out en eAP de
  modelos con distinto número de secuencias de entrenamiento), sólo citada con su ruta.
Criterio: H1 **principal** si la cota in-sample es ≤ 0,6 × OOF *y* hay evidencia de que la
puntuación held-out mejora ≥ 30 % al pasar de ~24 a 32–40 secuencias. H1 **refutada como
explicación principal** si las puntuaciones held-out históricas con distintos tamaños
quedan dentro de ±15 % del nivel de folds (curva plana): entonces productores con 40
secuencias no pueden, por sí solos, llevar un conjunto no visto de ~85 a ~36. Si no hay
datos suficientes, **indeterminada**, con la cota in-sample como límite superior del efecto.

### H2 — desplazamiento de covariables / test más fácil

Medidas (sin etiquetas de test):
- **M2a** descriptores train vs test: tamaño de caja (log_size), truncado actual y en la
  ventana, nº de fotogramas causales, longitud de track (consultas por track), pendiente de
  expansión (φ_área, φ_trunc), residuo RMS, desacuerdo RGB–evento, RGB–geometría,
  REL–geometría, distribución de la predicción y mezcla de bandas *predichas*, tanto
  ponderando cada consulta física como con la multiplicidad oficial (6.762 tokens, alias
  incluidos). Por secuencia: mezcla de bandas predichas.
- **M2b** reponderación de los folds hacia la distribución de covariables de Test12:
  (i) **primaria**: estratificación gruesa banda predicha REL (6) × terciles de log_size ×
  truncado actual × (n_frames ≥ 5); pesos = proporción test / proporción train por celda
  (celdas sin filas de train se informan y se excluyen); (ii) **secundaria**: clasificador
  logístico train-vs-test sobre covariables sin eventos (fases REL y RGB, logits de anchura,
  geometría), pesos p/(1−p) recortados en el percentil 99; (iii) variante con covariables de
  eventos. Se informa ESS. Se calcula el Overall reponderado de REL, del candidato y de las
  alternativas, con IC bootstrap de secuencias (pesos fijos).
- **M2c** escala de error sin etiquetas («three-cornered hat»): con tres estimadores
  (REL, geometría consciente de truncado, experto de eventos) y diferencias de fase
  winsorizadas a ±0,1, Var(e_A) = ½[Var(A−B) + Var(A−C) − Var(B−C)]. Se valida en folds
  frente al error real; se aplica a Test12; para REL la predicción de test se contrasta con
  el valor oficial conocido. Es la prueba de si la geometría también es más precisa en test.
Criterio: H2 **confirmada (principal)** si la reponderación primaria lleva el Overall REL
de folds a ≤ 50 (≥ 70 % de la brecha); **parcial** si lo lleva a ≤ 72,8 (≥ 25 %);
**no explicada por covariables observadas** si queda > 72,8. M2c apoya H2 si la escala
estimada de error REL en test es ≤ 0,6 × la de folds.

### H3 — diferencias de puntuación/población

Medidas: lectura del scorer oficial frente a `phase_geometry/metrics.py`; en TRAIN40:
rango de |T|, fracción fuera de bandas, T∈(0,0,1], duplicados físicos (misma secuencia,
track y timestamp), consultas por track; en Test12 (sin etiquetas): alias 957/6.762 y su
reparto por banda predicha; efecto de la multiplicidad de alias sobre el Overall simulado
en folds (no se puede simular exactamente: se informa sólo el peso relativo). Evidencia
histórica de Garl-TTC evaluado en secuencias de TRAIN (búsqueda en docs/commits). Si existe
evaluación local de Garl sobre secuencias de entrenamiento held-out con MiD oficial, se cita.
Criterio: **confirmada** si se encuentra una diferencia de definición que mueva el Overall
de folds ≥ 5 MiD; **refutada** si scorer y población coinciden en definición. El dato Garl:
si Garl (held-out o fold-local) puntúa en secuencias de TRAIN muy por encima de su Test12,
la brecha es propiedad de los datos, no del apilamiento de PHASE.

### H4 — concentración del error en la banda crucial (folds)

Medidas: aporte por secuencia y track a la suma de MiD crucial de REL (semilla 7 y media);
curva de concentración (fracción de la suma en el 10 % de tracks); T∈(0,1] frente a (1,3];
truncado; n_frames; prevalencia en test del estrato equivalente por proxies sin etiquetas.
Criterio: **confirmada** si el 10 % de tracks con más error acumula ≥ 50 % de la suma de MiD
crucial. Se informa si el estrato equivalente es raro en test.

### Decisión sobre el candidato (diagnósticos de riesgo)

- **D1** Test12: desplazamiento candidato−enviado por banda predicha (ya en geometría; se
  recalcula aquí en semilla 7).
- **D2** Folds: tasa de victoria condicional (|e_cand| < |e_REL|) por tamaño de desacuerdo
  |Δfase| (bins 0–10, 10–20, 20–40, 40–80, >80 MiD) y por banda predicha REL; estabilidad
  por secuencia (40 tasas). Se resume ponderando los bins con la distribución de
  desacuerdo de Test12.
- **D3** Alternativas (post-hoc, exploración; elegir entre ellas en folds es selección en
  datos de desarrollo): mezclas en fase w∈{0,25; 0,5; 0,75} del candidato y REL (por
  semilla emparejada), y `GATE5`: candidato sólo si caja no truncada y n_frames ≥ 5, si no
  REL. Overall, bandas, IC bootstrap pareado frente a REL y frente al candidato. No existe
  una configuración de cabeza con rasgos de productores de 40 secuencias en folds; se
  declara no disponible.
- **D4** Proyección a Test12. Modelo multiplicativo por banda:
  test_X_b = 35,80-bandas_REL_b × (MiD_X_b / MiD_REL_b) con los cocientes medidos en folds
  (a) sin reponderar, (b) con la reponderación primaria. P(peor que 35,80) = fracción de
  réplicas bootstrap de secuencias en las que el Overall proyectado > 35,80.
  Puerta de calibración: un estimador es admisible si, aplicado a REL sin cocientes
  (reponderación sola), reproduce el Overall oficial de REL en ±25 % (26,9–44,8); el modelo
  multiplicativo reproduce REL por construcción y por eso se juzga además con M2c.

### Regla de recomendación (fijada ahora)

- **Subir el candidato tal cual** si las proyecciones (a) y (b) son < 35,80 con
  P(peor) ≤ 0,10 en ambas, ≥ 30/40 secuencias mejoran en folds, la tasa de victoria
  condicional ponderada por el desacuerdo de test es > 0,5 y M2c no indica que la geometría
  sea relativamente peor en test que en folds (cociente de escala geo/REL en test ≤ 1,5 ×
  el de folds).
- **Subir una mezcla más segura** si el candidato no cumple alguna condición anterior pero
  una mezcla sí, o si su P(peor) es ≤ la mitad de la del candidato con pérdida proyectada
  ≤ 2 MiD.
- **No subir** si ninguna opción cumple, o si la evidencia en estratos parecidos a test es
  desfavorable.
