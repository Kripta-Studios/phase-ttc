# Decisión tras el diagnóstico raw

La primera intervención propuesta es **cobertura de profundidad hasta 150 m
con targets y pérdida coherentes**, frente a un control de 60 m bajo el mismo
perfil alineado. No se ha entrenado ni promovido este candidato.

## Evidencia y alcance

Las tablas de [REPORT.md](evidence/raw_diagnostics_20261101/REPORT.md) se regeneran
desde JSON verificado. El oracle de profundidad es el que más eleva la macro
entre las sustituciones ensayadas. El recall cae drásticamente más allá de 60 m;
también es casi nulo en objetos de altura proyectada inferior a 32 px.
Distancia y tamaño están correlacionados: esta observación no identifica por sí
sola cuál de los dos causa la pérdida. Ampliar rango es una hipótesis a refutar,
no una garantía de llegar a la puntuación oracle.

El oracle de clase revela un problema adicional en bicicleta y triciclo.
No se introduce simultáneamente una pérdida de clases nueva: primero se compara
la cobertura. El ranking tiene margen, pero las sustituciones ensayadas no
sustentan que reordenar el mismo pool resuelva la localización. No se ha calculado
una cota superior de AP óptima: el oracle de ranking conserva duplicados.

La asignación de máxima cardinalidad confirma que cambiar greedy por un matcher
óptimo apenas recupera recall de clase correcta. El problema no se resuelve
renombrando el matching ni cambiando la métrica publicada.

## Ensayo propuesto y condiciones de admisión

1. Mantener split, RGB 640x384, 50 queries, semilla 7 y presupuesto común de
   exposiciones. Comparar control 60 m y tratamiento 150 m. Medir tiempo completo
   y memoria antes de fijar updates; no heredar horas cloud antiguas.
2. Definir un contrato único de profundidad que alimente admisión, predictor y
   pérdida. El probe DDN demuestra por qué un cambio de YAML aislado es inválido.
3. La inicialización debe ser comparable. Cambiar bins cambia su significado
   aunque la forma del tensor coincida. Cualquier reinicialización/remapeo de
   cabeza y embeddings debe declararse y controlarse en ambos brazos.
4. Integrar validación raw y perfil/hash explícitos en un productor NUEVO.
   Rechazar opciones ignoradas. Verificar paridad del panel real con la CLI,
   checkpoint seleccionado/último y restore completo antes de entrenar largo.
5. Seleccionar con mAP alineada; mismo scoring entre brazos. Mantener además
   el candidato ranked congelado como referencia. No reajustar ranker en selección.
6. Screening solo después de forward/loss/backward, overfit corto y restore.
   Fijar el presupuesto final a partir del perfil medido antes de mirar curvas.
7. Promoción: al menos +0,005 mAP en selección completa frente al control,
   inspección de caídas >0,01 AP por clase y mejora de cobertura 60–150 m.
   Para reemplazar el candidato también debe superar su resultado reproducido;
   una mejora respecto a un control degradado no basta.

Refutación: si no mejora recall lejano ni macro con presupuesto comparable,
o daña clases próximas fuera de la regla, conservar el negativo y estudiar
resolución/profundidad geométrica. No sumar horas automáticamente.

## Métodos externos y protocolo

Se fijaron commits y hashes de README en
[PRIMARY_SOURCES.json](evidence/raw_diagnostics_20261101/PRIMARY_SOURCES.json).
[SPAN](https://github.com/WYFDUT/SPAN) queda como posible intervención geométrica
posterior; no se portaron SPAN, MonoCoP ni RARE y no se afirma admisión de sus
licencias/entornos. El diagnóstico actual justifica corregir primero cobertura
y contrato de validación del baseline existente.

El [repositorio oficial eAP](https://github.com/NAIL-HNU/eAP_dataset) enlaza la
competición 17291, coincidente con el recibo de nuestro envío. Corrección de la
nota inicial: la referencia a 16717 fue un error de lectura, no un cambio del
README; sus bytes siguen teniendo el mismo hash registrado. La recuperación
web devolvió contenido ajeno al benchmark. Posteriormente, la API respondió
correctamente al solicitar JSON explícitamente: overview, términos y fase
verificados; el leaderboard público no mostró entradas. Véase
[el recibo público](evidence/range_campaign_20261101/BENCHMARK_PUBLIC_STATUS.json).
La equivalencia privada y la admisión de datos externos siguen sin certificarse.

## Estado y continuidad

Esta iteración entrega herramientas ejecutadas, pruebas, diagnóstico completo
del pool guardado y una decisión. No cierra todos los apartados de P0–P2:
quedan reglas externas, censo TRAIN por tracks/admisión, AP por rango con una
política explícita de FP, errores de reproyección, visibilidad/iluminación
verificadas y auditoría de independencia. El trainer nuevo, ensayo y cola
reanudable quedaron pendientes al cierre del diagnóstico. La continuación
posterior está documentada en [RANGE_CAMPAIGN_20261101.md](RANGE_CAMPAIGN_20261101.md).

El mejor candidato válido sigue siendo el paquete RGB ranked del cierre;
ningún oracle debe exportarse como detector ni enviarse al benchmark.

## Reproducción

Usar el entorno existente `../phase-ttc-main-integration/.venv/Scripts/python.exe`
desde `phase-ttc-eap-diagnostics`. Los outputs de diagnóstico deben ser nuevos.

```powershell
& '../phase-ttc-main-integration/.venv/Scripts/python.exe' scripts/diagnose_detection_raw.py `
  --shards D:/PHASE_TTC_LOCAL_CONTINUATION_20261101/ranking_local_v1/shards `
  --release D:/PHASE_TTC_RELEASE_20261101/dfull-rgb-ranked-seed7 `
  --aligned D:/PHASE_TTC_BENCHMARK_ALIGNED_20261101/verified `
  --output D:/PHASE_TTC_RAW_DIAGNOSTICS_20261101/reproduction_new

& '../phase-ttc-main-integration/.venv/Scripts/python.exe' scripts/build_detection_raw_report.py `
  --evidence docs/detection_sota_20261029/evidence/raw_diagnostics_20261101 `
  --output docs/detection_sota_20261029/evidence/raw_diagnostics_20261101
```

La pasada válida está en `D:/PHASE_TTC_RAW_DIAGNOSTICS_20261101/v2`.
`v1` terminó con un fallo del comparador Python/JSON (tuplas frente a listas);
no es un resultado aceptado. Se corrigió la comparación serializada y la pasada
válida exige igualdad completa, sin tolerancia relajada. No hace falta repetirla
si permanecen iguales los hashes y la lógica.
