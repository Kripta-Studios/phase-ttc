# Registro de intentos P5

- `artifacts/sota_20261121/reliability_gate`: **sustituido e inadmisible**. La primera ejecución
  calculó pesos de bandas con las etiquetas de las 40 secuencias antes de separar cada bloque;
  por ello los recuentos del bloque exterior influyeron en la pérdida. También evaluó inicialmente
  la etiqueta de saturación virtual sobre predicciones originales. Sus métricas no se usan.
- Una segunda repetición en el mismo directorio se interrumpió deliberadamente al descubrir la
  fuga anterior; no constituye un resultado.
- `artifacts/sota_20261121/reliability_gate_v2`: cálculo corregido y completo, sustituido antes
  del cierre porque no persistía los pesos ejecutados; sus cifras no son las entregadas.
- `artifacts/sota_20261121/reliability_gate_v3`: única salida admisible. Calcula la masa dentro
  de cada fold, persiste pesos y hashes, y guarda predicciones originales y virtualmente
  recortadas producidas desde esos mismos objetos en memoria.
