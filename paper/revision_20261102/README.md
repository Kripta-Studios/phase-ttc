# PHASE-TTC: artículo breve complementario del 2 de noviembre de 2026

Artículo breve de siete páginas en inglés, con fuente LaTeX, PDF y evidencia reproducible. Es un documento complementario. La actualización principal de **From Motion to Time** es la [edición ampliada de 65 páginas](../from_motion_to_time_20261102/README.md), que conserva y amplía el artículo original. El manuscrito histórico de `paper/PHASE_TTC_Final.tex` también se conserva.

## Construcción

Requisitos: Python con matplotlib, NumPy, PyMuPDF y Pillow; pdfLaTeX y BibTeX de MiKTeX o TeX Live. Los paquetes LaTeX utilizados son estándar. No se ejecuta entrenamiento, inferencia ni acceso de red.

```powershell
python paper/revision_20261102/build_revision.py
python paper/revision_20261102/package_revision.py
```

En el ZIP autónomo, ejecutar `python build_revision.py`. El empaquetador es solo para el checkout original. El `.tex` consolidado incorpora estilo, tablas, esquema y bibliografía; necesita `figures/adaptation.pdf`, incluido en el ZIP. Se puede compilar tres veces con pdfLaTeX sin BibTeX.

Los números de las cinco tablas y del gráfico se regeneran desde JSON copiados. `source_manifest.json` autentica las cuatro tablas principales; `evidence_registry.json` y el manifiesto de detección identifican las otras fuentes. El constructor verifica hashes de las tablas principales y detecta referencias indefinidas, cajas desbordadas, glifos ausentes y fuentes sin incrustar. Las imágenes de inspección quedan en `build/`.

## Decisiones del consejo

Se incluyeron como contribuciones del sistema la interfaz causal de fase/tiempo/soporte, el adaptador compacto en TTC inverso y el contraste controlado de cadencia. Son resultados implementados y medidos; no se reclama prioridad mundial de la fusión por calidad ni de las ecuaciones físicas.

La mejora EvTTC procede del análisis local corregido de 434 consultas y se identifica como históricamente expuesto. Las cifras anteriores de 423 consultas están superadas. La discrepancia Slider-750 permanece abierta y no se anuncia una nueva puntuación oficial del adaptador. Los experimentos canónicos posteriores rechazados no se confunden con el modelo entregado.

JEPA, observabilidad, fotometría y reloj físico figuran con sus resultados negativos y límites. El trabajo eAP de detección queda en un apéndice: su AP no demuestra una mejora TTC. La latencia conserva el hardware y sus límites de medida; no acredita 50 Hz con cumplimiento de todos los plazos.

La revisión científica independiente corrigió el promedio del ensemble (en segundos), los fallbacks, la reconstrucción de entradas PAIR a cadencia canónica, los paneles distintos de geometría y las condiciones del benchmark de runtime. Las revisiones de los agentes se conservan en `review/`.

## Alcance de la revisión

Se actualizaron las referencias remotas mediante `git fetch --all --prune`, sin cambiar de rama ni modificar los experimentos. El inventario comprende 72 referencias, 241 commits únicos desde el 28 de octubre y 411 rutas Markdown modificadas directamente. El análisis de ambos padres de los merges amplía el inventario a 734 rutas y 806 blobs, incluidos documentos históricos importados; estos números no significan 806 experimentos nuevos.

Los manifiestos de cobertura distinguen lectura documental, duplicados y registros de historia generados. Los cinco índices mensuales históricos se verifican estructuralmente y se revisan en el intervalo pedido; no se presentan como una nueva lectura de toda la historia de junio a septiembre. Los resultados, métodos y decisiones relevantes se contrastan con JSON, código y recibos; no se afirma haber auditado línea por línea todos los cambios de código.

La cobertura final de `review/review_coverage.json` comprende 406 rutas con lectura semántica o verificación de sus fuentes y cinco índices históricos estructurados: ninguna de las 411 rutas recientes queda sin cobertura. El inventario conserva las versiones divergentes por rama y no trata los documentos recientes de detección como reemplazos automáticos de los de `main`.

## Estilo

Reconstrucción independiente del estilo efectivo de `arXiv-2608.27395v1.tar.gz`: dos columnas, US Letter, Latin Modern de 9,3 pt, abstract azul a todo el ancho, título de 16 pt, tablas booktabs y numeración centrada. El análisis está en `review/council_style.txt`. No se copian contenido científico, autores ni gráficos del artículo de referencia. La bibliografía se sitúa al final para evitar una página casi vacía.

El autor y su condición de estudiante en la Universidade de Santiago de Compostela proceden del documento de entrega actualizado del 1 de noviembre. No se añade afiliación de laboratorio ni se atribuye respaldo institucional.
