# PHASE-TTC: From Motion to Time — edición ampliada

**Causal Event–RGB Fusion for Signed Time-to-Contact Estimation**

<a href="https://github.com/Kripta-Studios/phase-ttc"><img src="../../docs/media/github-logo.png" width="22" height="22" alt="GitHub"></a> [GitHub](https://github.com/Kripta-Studios/phase-ttc) · <a href="https://www.linkedin.com/in/%C3%A1lvaro-schwiedop-souto-92718b339/"><img src="../../docs/media/linkedin-logo.png" width="22" height="22" alt="LinkedIn"></a> [LinkedIn](https://www.linkedin.com/in/%C3%A1lvaro-schwiedop-souto-92718b339/)

Álvaro Schwiedop Souto · Edición ampliada del 2 de noviembre de 2026; actualización del 12 de noviembre de 2026

Esta es la actualización y ampliación del artículo original de 52 páginas. Conserva sus 19 archivos de secciones, las figuras, los datos publicados, los métodos completos, las ablations históricas, las trazas por secuencia y el índice de evidencia. El artículo breve de siete páginas preparado antes no lo sustituye.

## Comparación de inferencia del 12 de noviembre

La comparación de latencia del artículo usa exclusivamente los servicios
persistentes de PHASE-TTC y Garl-TTC: pesos residentes, historial causal y
reutilización exacta del trabajo. Ambos reciben los mismos 250 paquetes RAW de
EvTTC CPLA-low a 50 Hz, con tres pasadas por modelo. El reloj principal incluye
adquisición, cola, preparación, inferencia y entrega del TTC al host. La carga
inicial y el acceso al disco quedan fuera de ese reloj. Garl conserva el
checkpoint completo y verifica el resultado sin decodificadores auxiliares
contra la inferencia nativa original.

`data/persistent_service_20261112/` conserva los protocolos, las seis ejecuciones,
sus relojes y las comprobaciones de equivalencia. `scripts/build_persistent_comparison.py`
verifica la evidencia y regenera tablas y figuras. Los experimentos anteriores de
API por consulta permanecen como archivo histórico en `data/local_hardware_20261112/`;
sus latencias no se usan en la comparación de producción del artículo. Los
29,14 ms históricos son un p95 de PHASE, no una garantía de respuesta en 20 ms.

## Consolidación científica del 11 de noviembre

La versión actual incorpora nueve ajustes de cabezas de igual capacidad, con
productores e historial fijos, y los controles geométricos sobre soporte común.
El resultado oficial corresponde al sistema completo: los controles nuevos no
muestran una mejora consistente al añadir eventos ni una ventaja de fase sobre
TTC inverso en la cohorte de desarrollo inspeccionada. Las cabezas experimentales
no sustituyen los pesos publicados ni tienen una nueva puntuación oficial.

`data/consolidation/` contiene predicciones, protocolos, trazas y fuentes ejecutadas.
`scripts/build_consolidation.py` regenera las tablas. `disposition.json` identifica
lo corregido y los requisitos aún pendientes: nuevos datos independientes,
reconciliación del evaluador, reproducción histórica y cumplimiento de latencia.
Los controles V8 recuperados se presentan juntos; su pequeña discrepancia con el
agregado antiguo queda documentada. Las declaraciones de ausencia de nuevos
experimentos en revisiones anteriores describen aquellas ediciones históricas.

## Archivos y construcción

`main.tex` es la fuente modular. `PHASE_TTC_From_Motion_to_Time_Expanded.tex` consolida estilo, texto, tablas y bibliografía; necesita las carpetas de figuras incluidas en el ZIP. El PDF correspondiente contiene el artículo y el suplemento completos.

El [ZIP de fuentes publicado en GitHub](../PHASE_TTC_From_Motion_to_Time_Expanded_sources.zip)
se regenera al empaquetar y contiene la evidencia y el PDF verificados.

```powershell
python scripts/build_expanded.py
python scripts/verify_publication.py
python scripts/package_expanded.py
```

El constructor verifica los hashes y las proyecciones numéricas de la evidencia histórica, regenera las tablas nuevas desde JSON y compila las dos fuentes. Requiere Python con NumPy, Matplotlib, Pillow y PyMuPDF, además de pdfLaTeX y BibTeX con XCharter, newtxmath, TikZ, tcolorbox y los paquetes habituales. No entrena modelos ni realiza inferencia ni acceso de red. Las figuras históricas se conservan sin alteraciones.

La revisión del 4 de noviembre incorpora tres figuras con datos reales de EvTTC usados en la web del proyecto: escena RGB/eventos, recortes temporales del vehículo y los doce canales de entrada de eventos. Las imágenes conservan sus bytes originales, sus timestamps, la atribución y los hashes de extracción en `data/real_sensor_excerpt/`. El constructor comprueba estos archivos con `scripts/verify_sensor_excerpt.py`; `ASSETS.md` explica las transformaciones de visualización. La ampliación visual no añade experimentos ni modifica resultados.

La revisión editorial posterior aplica la guía de reescritura al abstract, las contribuciones, la descripción del modelo, los experimentos y la discusión. La evolución metodológica sustituye a la bitácora cronológica; las revisiones de código se reúnen en una tabla de reproducibilidad. La prosa usa valores redondeados y referencias de evidencia agrupadas, mientras las tablas numéricas, macros, ecuaciones y fuentes científicas mantienen sus valores. Se conserva la distinción entre resultados oficiales, adaptación local y comparaciones con exposición durante el desarrollo. Los ejemplos de la guía con cifras o interpretaciones no respaldadas se sustituyen por lo que acreditan los registros. La guía editorial local se excluye del ZIP científico; `updates/review/editorial_revision_validation.json` registra las comprobaciones de esta edición.

La fuente consolidada también compila con tres pasadas de pdfLaTeX sin BibTeX. `scripts/package_expanded.py` empaqueta la entrega desde el checkout original y verifica la conservación de los archivos de partida. `SOURCE_MANIFEST.json` autentica las fuentes, la evidencia y el PDF final; excluye los auxiliares de compilación, las imágenes temporales de inspección y el duplicado `main.pdf`. El `.bbl` se regenera con BibTeX al construir la fuente modular.

## Qué se amplía

La entrega corregida del 11 de noviembre incorpora la auditoría científica que
todavía no estaba en el PDF anterior. La transferencia vigente usa 460 consultas
elegibles de 508 candidatos: MAE fusión/RGB de 0,813/0,851 s y p95 de 2,742/3,038 s.
Los intervalos pareados de diferencia incluyen cero. Las 468 consultas y sus
conclusiones anteriores se conservan exclusivamente como evidencia histórica.
La metodología distingue generación PHASE20 por partición, refit de contexto y
las unidades de bootstrap específicas de cada estudio. El suplemento documenta
los 239 targets afectados del F1 histórico, su separación respecto al router F0
entregado y los controles y experimentos todavía pendientes.

`revision/publication_contract.json` vincula las entradas científicas y los
artefactos compilados. El empaquetador falla si cambian después de la construcción.
`updates/review/material_corrections_20261111.json` documenta la comprobación de
esta entrega. Los recibos editoriales anteriores conservan sus fechas y hashes;
no acreditan el PDF actualizado. Aquella revisión del 11 de noviembre no ejecutó nuevos entrenamientos,
inferencia de sensores ni envíos oficiales.

- Evaluación EvTTC local corregida con 434 targets, exposición histórica explícita y discrepancia Slider-750 pendiente. Se mantienen las comparaciones históricas de cobertura y desacuerdo, identificadas como tales.
- PHASE-TTC-Adapt: geometría causal, características de calidad, routers, política exacta de fallback, promedio del ensemble y protocolo de ajuste.
- PAIR canónico: reconstrucción temporal de entradas, conversión de coordenadas, identidad numérica, 2.608 consultas, refit rechazado, sensibilidad nativa y límites de latencia.
- Piloto predictivo con profesor congelado y controles estáticos/permutados; resultados negativos de observabilidad, fotometría, superficies y dinámica.
- Detección eAP como tarea distinta, incluyendo evaluación alineada, diagnóstico geométrico ideal y resultado negativo con predicciones.
- Cronología, discusión, conclusiones, poblaciones y registro de fuentes actualizados.

Los nuevos identificadores N01–N07 complementan los históricos; no reemplazan E01–E08. `updates/` conserva los JSON, hashes y comprobaciones de evidencia, así como la cobertura de 241 commits y 411 rutas Markdown: 406 revisadas por contenido o fuentes y cinco índices históricos verificados estructuralmente.

## Tipografía y continuidad estética

Se mantienen XCharter/newtxmath, A4, una columna, los márgenes, la portada From Motion to Time y la paleta original. La portada prescinde del logotipo y muestra el correo `alvaro@schwiedop.com` y la web [schwiedop.com/PHASE-TTC](https://schwiedop.com/PHASE-TTC). Se han estabilizado los tamaños tras cambios locales, aumentado discretamente la legibilidad de pies y claves de evidencia, suavizado el abstract con un fondo azul y ajustado el índice y los saltos de página.

La fuente arXiv suministrada sirvió como referencia de ritmo tipográfico, jerarquía y tratamiento del abstract. No se impone su formato de dos columnas ni se copian sus autores, texto científico o gráficos. Las figuras originales conservan sus colores, datos y atribuciones, descritas en `ASSETS.md`.

## Verificación

`build/validation.json` registra páginas, palabras, figuras, tablas y comprobaciones LaTeX; `updates/review/original_retention.json` compara la versión ampliada con los originales. Las páginas renderizadas se inspeccionan antes de la entrega. El ZIP incluye los recibos finales y un manifiesto SHA-256.

La ampliación conserva los límites científicos: las mejoras locales expuestas no son validación independiente; la adaptación no tiene nueva puntuación oficial acreditada; la corrección canónica posterior no sustituye el modelo entregado; el replay medido no cumple todos los plazos de 20 ms.

## Reestructuración narrativa y revisión gráfica del 11 de noviembre

Se aplican la guía de humanización y los principios de Kuan-Hao Huang a la
narrativa para lectores de visión y robótica: intuición, método, preguntas y
resultados. La portada incorpora un esquema conceptual con observaciones
reales; el gráfico del leaderboard pasa a su sección de resultados. Los
identificadores de evidencia se ocultan en el cuerpo mediante
`\publicationmodetrue` y vuelven a mostrarse en el suplemento. Se trasladan
allí la genealogía de checkpoints, la asociación detallada de targets y los
controles de cadencia. Los resultados negativos de aprendizaje predictivo
tienen una subsección propia, sin atribuir un fracaso universal a JEPA.

La notación de fusión describe las funciones y el ancla residual; el suplemento
conserva el almacenamiento exacto y las políticas de emisión. La escala 0,03 no
se presenta como cota de la corrección. Tampoco se afirma que la fase elimine
singularidades o que el replay cumpla el plazo de 20 ms.

`scripts/build_publication_diagrams.py` regenera los diagramas revisados:
arquitectura, rotaciones, fusión, historia, relojes y portada. Las flechas
indican señales o transferencia de modelos según su leyenda; las entradas de
entrenamiento de contexto y fusión siguen rutas distintas. Los gráficos de
latencia identifican el reloj posterior al sellado del payload. Los archivos
`publication_*` sustituyen los diagramas antiguos en el artículo, cuyos bytes
se conservan en el archivo fuente. No se modifican datos ni predicciones.

`updates/review/humanization_visual_20261111.json` registra la revisión de esta
edición y su PDF. Los recibos previos acreditan exclusivamente las entregas
identificadas por sus propios hashes.

## Segunda pasada de estilo — 11 de noviembre

La revisión posterior vuelve a leer `GUIA_HUMANIZACION_Y_REVISION_PHASE_TTC.md`
y aplica las observaciones adicionales del autor: integra el arranque de la
introducción, concentra la discrepancia del evaluador en un único párrafo del
cuerpo y elimina las negaciones de plantilla. La discusión plantea pruebas de
generalización y despliegue concretas. La latencia se interpreta desde sus dos
relojes medidos, sin atribuir un cuello de botella causal ni introducir tiempos
no respaldados. Los diagramas y los datos conservan su contenido.

El recibo `updates/review/editorial_refinement_20261111.json` identifica esta
entrega, el hash de la guía leída, las comprobaciones numéricas y la inspección
del PDF. El recibo visual anterior corresponde al PDF de la primera pasada.
