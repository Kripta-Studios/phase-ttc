# PHASE-TTC: From Motion to Time — edición ampliada

**A Causal Event–RGB System for Signed Time-to-Contact Estimation**

<a href="https://github.com/Kripta-Studios/phase-ttc"><img src="../../docs/media/github-logo.png" width="22" height="22" alt="GitHub"></a> [GitHub](https://github.com/Kripta-Studios/phase-ttc) · <a href="https://www.linkedin.com/in/%C3%A1lvaro-schwiedop-souto-92718b339/"><img src="../../docs/media/linkedin-logo.png" width="22" height="22" alt="LinkedIn"></a> [LinkedIn](https://www.linkedin.com/in/%C3%A1lvaro-schwiedop-souto-92718b339/)

Álvaro Schwiedop Souto · Edición ampliada del 2 de noviembre de 2026; actualización editorial del 21 de noviembre de 2026

La fuente modular mantiene dos cuerpos. `main.tex` construye la edición de
archivo con el suplemento completo. `main_short.tex` construye un artículo breve
desde `sections/short_article.tex` y reutiliza las tablas, figuras y bibliografía
de la edición de archivo. El recuento final de páginas debe validarse con la
plantilla de la publicación elegida; esta fuente no declara conformidad con un
límite de páginas aún no especificado.

Esta es la actualización y ampliación del artículo original de 52 páginas.
Conserva las figuras, los datos publicados, los métodos completos, las
ablations históricas, las trazas por secuencia y el índice de evidencia. El
artículo breve archivado que se preparó antes y el nuevo `main_short.tex` son
entregas distintas; ninguno sustituye la edición de archivo.

## Auditoría de alcance de afirmaciones del 18 de noviembre

Una auditoría independiente recalculó las cifras del manuscrito contra la
evidencia guardada y el código. No encontró discrepancias numéricas; sí corrigió
el alcance de varias afirmaciones. Esta entrega no añade entrenamientos,
inferencia de sensores ni envíos oficiales. Cambios:

- **Benchmark oficial.** El resumen y la sección de resultados indican que el leaderboard
  archivado tiene tres entradas, que Garl-TTC usa dos observaciones separadas
  0,1 s frente a los 650 ms de PHASE-TTC, que las regiones de objeto son las del
  benchmark y que el archivo no registra el número de envíos de la cuenta ni un
  hash del paquete puntuado. La deriva del replay nativo queda acotada en unas
  0,01 unidades MiD de media.
- **Atribución.** Los controles emparejados no atribuyen la mejora a la fusión:
  las cabezas reajustadas no se distinguen de la rama RGB (+0,004 s,
  [−0,089; +0,114]); la ventaja sobre la mezcla calibrada depende de CPNAO-high;
  la cabeza publicada coincide con la semilla más favorable. La tabla pareada
  incorpora estos tres contrastes, generados por `scripts/build_consolidation.py`.
- **Desajuste del historial de eventos.** El apéndice E.8 documenta que la rama
  de eventos se entrenó con historial cada 50 ms (siete slots, recorte de la
  consulta actual) y se evaluó en Test12 y EvTTC con historial cada 100 ms
  (cuatro slots, recorte propio de cada observación). Su efecto en la precisión
  no está medido.
- **Router.** Ocho grabaciones de carretera, tres de las cuales concentran la
  ventaja; la mezcla fija usa dos expertos y el router cuatro; en el agregado la
  mezcla fija tiene menor error puntual; diez grabaciones de ajuste son hermanas
  de cinco evaluadas.
- **Latencia.** El resumen incluye el resultado del estudio de 72 pasadas. La
  sección de runtime advierte que los tiempos de Garl dependen de nuestra
  implementación del servicio y que el colapso con cola de dos paquetes es un
  fallo de arranque cuya causa no está aislada.
- **Historia.** La tabla H8 de transferencia queda marcada como calculada con la
  asignación de etiquetas sustituida. El apéndice V8 indica que el encoder JEPA
  congelado fue peor que uno aleatorio congelado, y recoge dos negativos antes
  omitidos (cribado dense level–dynamics y SSL con CARLA).
- **Protocolo.** Se declara la reutilización de datos de desarrollo (al menos 498
  ajustes sobre la población de adaptación) y que la independencia entre
  secuencias a nivel de escena no está verificada.
- **Bibliografía.** Pasa de 13 a 32 entradas, con una subsección de trabajo
  relacionado sobre TTC por expansión, TTC con eventos y fusión eventos–imagen.

Nota histórica de aquella auditoría: el autor declaró entonces un único envío al
servidor de GarlTTC. El historial de la cuenta recuperado después documenta tres
envíos terminados, `1016512`, `1058935` y `1063261`; la declaración anterior se
conserva solo para explicar el estado del texto fechado.

## Segundo envío al servidor de test: 23,47 Overall MiD

El 18 de noviembre se envió una sola vez, con recibo de prerregistro commiteado
antes de subirlo, la variante elegida por la regla del control de geometría
(rama RGB + expansión causal de las regiones suministradas + pérdida por banda,
sin entradas de eventos). Envío `1058935`, paquete verificado contra el recibo:
**23,47 Overall MiD** (bandas c/s/l/n 24,37 / 23,95 / 22,89 / 18,13), frente a 35,80
del sistema original y 45,03 de Garl-TTC (−34 % y −48 %). Es el segundo envío de
la cuenta. Los controles TRAIN40 sitúan la mayor mejora de desarrollo en la
historia de las regiones suministradas; el resultado privado valida la receta,
no una atribución causal a esa señal.

## Tercer envío: comparador de solo cajas

El 21 de noviembre se puntuó el comparador prerregistrado que conserva los mismos
600 ms de historia de regiones suministradas y elimina también las estimaciones
RGB. El envío `1063261` obtuvo **27,97563 Overall MiD** (bandas c/s/l/n
26,38540 / 28,00007 / 32,93790 / 30,89121), con 6.762 tokens y cero fallos del
scorer. Frente a 23,47238 para RGB más geometría, la diferencia es 4,50325 y la
reducción relativa al comparador de solo cajas es 16,10 %. Es el tercer uso
secuencial del mismo servidor privado, no un test independiente. El registro
descargado quedó preservado en el commit `8fa8dda6`.

## Controles posteriores al envío del 18 de noviembre

El apéndice «Post-submission controls» añade tres controles con protocolo fijado
antes de calcular las comparaciones. Ninguno usa etiquetas privadas. En el corte
del 18 de noviembre solo el candidato RGB más geometría se había enviado; el
comparador de solo cajas se puntuó después, como tercer envío. Sus tablas se generan con `scripts/build_post_audit.py` a
partir de `data/post_audit_20261118/`.

- **Geometría de cajas.** En los folds de entrenamiento, añadir a la cabeza la
  expansión causal de las regiones suministradas baja el Overall MiD de 85,17 a
  41,96, y a 41,27 con pérdida por banda y sin eventos (variante elegida por la
  regla prefijada). Los niveles de los folds no son comparables con el 35,80 del
  test. Enviada después al servidor, esta variante puntúa 23,47 (sección
  anterior).
- **Historial de eventos alineado con el entrenamiento.** Con todo congelado, no
  mejora en EvTTC: +0,058 s de MAE [−0,004; +0,120] y +5,2 MiD [+0,9; +9,1].
- **Historia más allá de dos observaciones.** Aporta unos 3 de 85 puntos en los
  folds (+3,51 [+2,67; +4,31] con la semilla 7). No reentrena Garl-TTC.

**Identidad del paquete puntuado: verificada.** El archivo recuperado del
registro de envíos de CodaBench el 18 de noviembre tiene SHA-256
`5eb69749de7aff682eea640d843ee635379702337d929c86f50472264f1d88bc`, idéntico al
paquete TRAIN40 descrito en el artículo, y el registro de puntuación que lo
acompaña da 35,80145. El paquete anterior de otro modelo, de tamaño casi igual,
no es el puntuado.

## Servicio CPU de geometría del 21 de noviembre

El ensayo de solo cajas ejecuta ocho celdas host-paced de 60 segundos y 3.000
ofertas a 50 Hz cada una. Siete cumplen el gate estricto; una pierde cinco
ofertas. En total, 23.992 de 24.000 terminan antes de 20 ms y 23.995 producen
respuesta válida, por lo que el gate global falla. Parte de cajas TRAIN40
suministradas y no incluye detector ni tracker. Este resultado solo cubre la ruta
CPU de historia de cajas. Véase el
[informe CPU](../../docs/sota_20261121/geometry_service/REPORT.md).

## Comparación de inferencia del 12 de noviembre

La comparación usa exclusivamente los servicios persistentes de PHASE-TTC y
Garl-TTC: pesos residentes, historial causal y reutilización del trabajo bajo identidades de entrada verificadas.
El reloj principal incluye adquisición, cola, preparación, inferencia y entrega
del TTC al host. La carga inicial y el acceso al disco quedan fuera de ese reloj.

La campaña ampliada contiene seis escenas de EvTTC, 1.979 anchors distintos y
72 pasadas en procesos nuevos. Cada modelo ejecuta cuatro repeticiones por
escena con cola de dos paquetes y dos repeticiones con cola de 64. Los tramos
continuos duran entre 3,1 y 9,3 segundos; sus repeticiones no se presentan como
una grabación continua de 60 segundos. Los oráculos nativos se ejecutan después
de terminar todas las mediciones y comprueban cada predicción entregada.

`data/extended_service_20261112/` conserva los protocolos, todos los intentos,
las referencias independientes, los pilotos y los inventarios de integridad.
`scripts/build_extended_inference.py` verifica los archivos y regenera las
comparaciones usando las fuentes del verificador congeladas en el propio ZIP.
El capítulo separa cobertura, p95 de respuesta, plazo de 20 ms, tiempo de servicio,
cola, memoria, inicialización, antigüedad del RGB y comportamiento posterior al
arranque. Las distribuciones incompletas no se convierten en latencias de servicio
con cobertura completa. El análisis inicial frente a la segunda mitad se
identifica como diagnóstico retrospectivo. La entrega completa se distingue de
la admisión numérica: los fallos de tolerancia frente al oráculo se conservan por
componente, repetición e instante único, y excluyen la pasada de las cifras
agregadas que requieren paridad. No se amplían las tolerancias para admitirla.

La campaña inicial de tres pasadas sobre 250 anchors CPLA-low permanece en
`data/persistent_service_20261112/`. Los experimentos anteriores de API por
consulta se archivan en `data/local_hardware_20261112/`; sus latencias no se usan
en la comparación de producción. Los 29,14 ms históricos son un p95 de PHASE,
no una garantía de respuesta en 20 ms.

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

`main_short.tex` es la fuente breve reproducible. Se compila desde esta carpeta:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error main_short.tex
bibtex main_short
pdflatex -interaction=nonstopmode -halt-on-error main_short.tex
pdflatex -interaction=nonstopmode -halt-on-error main_short.tex
```

`scripts/make_official_submission_tables.py` regenera las tablas de envíos de
ambas fuentes desde `data/official_submissions/records.json`. Ese registro
conserva los valores a precisión completa y los SHA-256 de
`../../paper/data/official_benchmark.json` y
`../../docs/sota_20261118/leaderboard/submission_1058935/RECORD.json` y
`../../docs/sota_20261118/leaderboard/submission_1063261/RECORD.json`; cuando
existen en el checkout, el generador verifica ambos hashes. La opción `--check`
falla si cualquiera de las dos tablas está obsoleta.

El [ZIP de fuentes publicado en GitHub](../PHASE_TTC_From_Motion_to_Time_Expanded_sources.zip)
se regenera al empaquetar y contiene la evidencia y el PDF verificados.

```powershell
python scripts/build_expanded.py
python scripts/verify_publication.py
python scripts/package_expanded.py
```

El constructor verifica los hashes y las proyecciones numéricas de la evidencia histórica, regenera las tablas nuevas desde JSON y compila la edición expandida, su fuente consolidada y `main_short.tex`. El contrato de publicación sella también `PHASE_TTC_From_Motion_to_Time_Short.pdf`; el empaquetador lo copia a `output/pdf/` y lo incluye en el ZIP. Las tablas de FCWD, literatura y latencia de `sections/followup_controls.tex` se generan con `scripts/make_followup_tables.py`; sus CSV/JSON sellados y el manifiesto SHA-256 están en `data/followup/`. `python scripts/make_followup_tables.py --check` falla si una tabla o su manifiesto está obsoleto. Requiere Python con NumPy, Matplotlib, Pillow y PyMuPDF, además de pdfLaTeX y BibTeX con XCharter, newtxmath, TikZ, tcolorbox y los paquetes habituales. No entrena modelos ni realiza inferencia ni acceso de red. Las figuras históricas se conservan sin alteraciones.

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

## Declaraciones y distribución

El autor confirma que la investigación no recibió financiación y que
no existen conflictos de interés. El código y los pesos publicados usan MIT; los
modelos y datasets de terceros conservan sus propias licencias y no se
relicencian aquí. En particular, el profesor DINOv3 de los estudios V8 está
sujeto a la licencia DINOv3 de Meta.

Antes de publicar una nueva edición se deben cotejar como una sola matriz:

| Identidad | Pesos o paquete | Resultado | Protocolo y límite |
|---|---|---|---|
| PHASE-TTC | Hub `AlvaroSchwiedop/PHASE-TTC` | 35,8015; envío 1016512 | Sistema event--RGB original, hasta 650 ms; registro archivado del 28 de octubre. |
| PHASE-TTC-Adapt | Hub `AlvaroSchwiedop/PHASE-TTC-Adapt` | Evaluación local de 434 consultas | Tres routers; no es el paquete puntuado con 23,47. |
| PHASE-TTC-Geo | Paquete sellado del envío 1058935; no está en los dos repositorios Hub anteriores | 23,4724; 0 % de fallos del scorer | RGB + historia causal de cajas, sin eventos; segundo envío del 18 de noviembre. |
| Comparador de solo cajas | Paquete sellado del envío 1063261; no es un modelo Hub | 27,97563; 0 % de fallos del scorer | Historia causal de cajas, sin imagen ni eventos; tercer envío del 21 de noviembre. |

La entrega requiere PDF largo, PDF breve si la sede lo admite, fuentes, ZIP,
hashes de PDF/ZIP/pesos vistos desde el lado del lector, fichas de modelo
sincronizadas y pasos de restauración. La ficha descargada de CodaBench conserva
vacío el campo `Uses Event` de PHASE-TTC-Geo; el texto `sin eventos` procede de
la receta sellada, no de completar ese campo retrospectivamente. El ZIP de la
Space aún requiere comprobar que su copia de `app.py` coincide con el texto
publicado.

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
