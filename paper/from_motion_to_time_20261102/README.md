# PHASE-TTC: From Motion to Time — edición ampliada

**Causal Event–RGB Fusion for Signed Time-to-Contact Estimation**

Álvaro Schwiedop Souto · 2 de noviembre de 2026

Esta es la actualización y ampliación del artículo original de 52 páginas. Conserva sus 19 archivos de secciones, las figuras, los datos publicados, los métodos completos, las ablations históricas, las trazas por secuencia y el índice de evidencia. El artículo breve de siete páginas preparado antes no lo sustituye.

## Archivos y construcción

`main.tex` es la fuente modular. `PHASE_TTC_From_Motion_to_Time_Expanded.tex` consolida estilo, texto, tablas y bibliografía; necesita las carpetas de figuras incluidas en el ZIP. El PDF correspondiente contiene el artículo y el suplemento completos.

```powershell
python scripts/build_expanded.py
```

El constructor verifica los hashes y las proyecciones numéricas de la evidencia histórica, regenera las tablas nuevas desde JSON y compila las dos fuentes. Requiere Python con NumPy, Matplotlib, Pillow y PyMuPDF, además de pdfLaTeX y BibTeX con XCharter, newtxmath, TikZ, tcolorbox y los paquetes habituales. No entrena modelos ni realiza inferencia ni acceso de red. Las figuras históricas se conservan sin alteraciones.

La fuente consolidada también compila con tres pasadas de pdfLaTeX sin BibTeX. `scripts/package_expanded.py` empaqueta la entrega desde el checkout original y verifica la conservación de los archivos de partida. `SOURCE_MANIFEST.json` autentica las fuentes, la evidencia y el PDF final; excluye los auxiliares de compilación, las imágenes temporales de inspección y el duplicado `main.pdf`. El `.bbl` se regenera con BibTeX al construir la fuente modular.

## Qué se amplía

- Evaluación EvTTC local corregida con 434 targets, exposición histórica explícita y discrepancia Slider-750 pendiente. Se mantienen las comparaciones históricas de cobertura y desacuerdo, identificadas como tales.
- PHASE-TTC-Adapt: geometría causal, características de calidad, routers, política exacta de fallback, promedio del ensemble y protocolo de ajuste.
- PAIR canónico: reconstrucción temporal de entradas, conversión de coordenadas, identidad numérica, 2.608 consultas, refit rechazado, sensibilidad nativa y límites de latencia.
- Piloto predictivo con profesor congelado y controles estáticos/permutados; resultados negativos de observabilidad, fotometría, superficies y dinámica.
- Detección eAP como tarea distinta, incluyendo evaluación alineada, diagnóstico geométrico ideal y resultado negativo con predicciones.
- Cronología, discusión, conclusiones, poblaciones y registro de fuentes actualizados.

Los nuevos identificadores N01–N07 complementan los históricos; no reemplazan E01–E08. `updates/` conserva los JSON, hashes y revisiones del consejo, así como la cobertura de 241 commits y 411 rutas Markdown: 406 revisadas por contenido o fuentes y cinco índices históricos verificados estructuralmente.

## Tipografía y continuidad estética

Se mantienen XCharter/newtxmath, A4, una columna, los márgenes, la portada From Motion to Time y la paleta original. La portada prescinde del logotipo y muestra el correo `alvaro@schwiedop.com` y la web [schwiedop.com/PHASE-TTC](https://schwiedop.com/PHASE-TTC). Se han estabilizado los tamaños tras cambios locales, aumentado discretamente la legibilidad de pies y claves de evidencia, suavizado el abstract con un fondo azul y ajustado el índice y los saltos de página.

La fuente arXiv suministrada sirvió como referencia de ritmo tipográfico, jerarquía y tratamiento del abstract. No se impone su formato de dos columnas ni se copian sus autores, texto científico o gráficos. Las figuras originales conservan sus colores, datos y atribuciones, descritas en `ASSETS.md`.

## Verificación

`build/validation.json` registra páginas, palabras, figuras, tablas y comprobaciones LaTeX; `updates/review/original_retention.json` compara la versión ampliada con los originales. Las páginas renderizadas se inspeccionan antes de la entrega. El ZIP incluye los recibos finales y un manifiesto SHA-256.

La ampliación conserva los límites científicos: las mejoras locales expuestas no son validación independiente; la adaptación no tiene nueva puntuación oficial acreditada; la corrección canónica posterior no sustituye el modelo entregado; el replay medido no cumple todos los plazos de 20 ms.
