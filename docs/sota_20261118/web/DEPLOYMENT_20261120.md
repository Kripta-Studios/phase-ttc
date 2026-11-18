# Despliegue web y Hugging Face: resultado de test de la variante geométrica

Fecha de ejecución: 2026-11-18, 17:21–17:40 UTC. Los nombres de release y del
directorio de trabajo llevan el prefijo `20261120`, como continuación del prefijo
`20261119` de la entrega anterior, para que el orden alfabético de `releases/` siga
siendo cronológico. Fuente: `Kripta-Studios/phase-ttc` `main` en
`72918d3c230b156a7f8a144b2b91c69f8c2670a8` ("Record the leaderboard of 18 November after
the second submission"; local = `origin/main`, árbol limpio).

| | |
|---|---|
| Release anterior | `/var/www/schwiedop/releases/20261119T170408Z-claim-scope` |
| Release nueva | `/var/www/schwiedop/releases/20261120T173106Z-test-result` |
| Directorio de trabajo | `~/schwiedop-test-result-20261120T173106Z` (en el servidor) |
| Cambio | `current` cambió de forma atómica (`ln -sfn` + `mv -Tf`) a las 17:32:16 UTC |

## Rollback

```sh
ln -sfn /var/www/schwiedop/releases/20261119T170408Z-claim-scope /var/www/schwiedop/current.new && mv -Tf /var/www/schwiedop/current.new /var/www/schwiedop/current
```

La release anterior está intacta. No se modificaron Caddy, otros sitios, servicios ni contenedores.

## Procedimiento

1. `cp -a` de la release activa a la nueva. Los tres builders sin editar
   (`python3 -B`, Python 3.12.3 del servidor) reprodujeron la release activa byte a byte
   (`diff -rq`: idénticas).
2. Entradas de los builders editadas en local con sustituciones exactas (una coincidencia
   cada una, CRLF conservado), subidas con el PDF, el `.tex` y el ZIP de `72918d3c` y
   comprobadas con `sha256sum -c` (`stage.sha256`).
3. Instaladas con `cp` (se conservan los modos de destino) y ejecutados los tres builders.
   Portada regenerada desde la página 1 del PDF nuevo (`pdftoppm -r 300` y `convert`
   a 637×901, JPEG progresivo, calidad 90, 4:2:0, 96 ppp: las mismas propiedades que la anterior).
4. Parámetro de caché `?v=20261119-claim-scope` → `?v=20261120-test-result` (portada en
   `build.py`, `paper-viewer.js` en `build-paper.py`, PDF en `paper-viewer.js`).
5. El PDF pasa de 105 a 106 páginas (la tabla por bandas del apéndice Z). Controles
   posteriores al envío: páginas 103–106; el capítulo de inferencia sigue en 91–102.
   El texto del PDF no contiene "Rank 1"; la figura 1 dice "1st of 3 entries".

## Archivos cambiados (15; ningún archivo añadido ni eliminado)

| Archivo | sha256 anterior | sha256 nuevo | modo |
|---|---|---|---|
| `build-paper.py` | `5e25054b29223a83…` | `5fec70e238377e93…` | 666 |
| `build-phase-article.py` | `ac239745aa055863…` | `ebc283f13cf7319e…` | 666 |
| `build.py` | `dd0e0112822216f1…` | `b4709872aecc0657…` | 666 |
| `public/PHASE-TTC/index.html` | `9ce2fea6045b4e9c…` | `2232f8a15797d632…` | 644 |
| `public/es/PHASE-TTC/index.html` | `73f3862cd92e6297…` | `f69a8cb0fb8382ed…` | 644 |
| `public/index.html` | `afdf81dedd7bed22…` | `86da1a423a522652…` | 644 |
| `public/es/index.html` | `09fdc8985acbf5e0…` | `364e7d483be9ee94…` | 644 |
| `public/papers/phase-ttc/index.html` | `1a85e6a119849907…` | `03fd4a4113d53d2b…` | 644 |
| `public/es/papers/phase-ttc/index.html` | `b123a13bc65648b4…` | `1b2d19e0076ea362…` | 644 |
| `public/papers/phase-ttc/expanded-edition.json` | `b2e81e1dd92f04c2…` | `3d48126e4bb33a97…` | 666 |
| `public/papers/phase-ttc/PHASE_TTC_From_Motion_to_Time_Expanded.pdf` | `1b36d8ad6c7b674b…` | `0cc7931a84e224b8…` | 666 |
| `public/papers/phase-ttc/PHASE_TTC_From_Motion_to_Time_Expanded.tex` | `473fc53304e829f8…` | `428cb794654c6afc…` | 666 |
| `public/papers/phase-ttc/PHASE_TTC_From_Motion_to_Time_Expanded_sources.zip` | `305eed10cba5a69f…` | `e9a29c80c26dab7c…` | 666 |
| `public/assets/paper-viewer.js` | `b68f7e9cea195c66…` | `65273bafe20d211f…` | 666 |
| `public/assets/phase-ttc-expanded-cover.jpg` | `dd7ce5f3b15e0f97…` | `6dc894daa8343a74…` | 666 |

Hashes completos en `verify-report.json` del directorio de trabajo.

| Archivo del artículo (= blobs de `origin/main` en `72918d3c`) | sha256 | bytes |
|---|---|---|
| PDF (106 páginas, "Updated 18 November 2026") | `0cc7931a84e224b8556494afba0c69f63f16457b3610da228c790eb70b51db7b` | 4,024,696 |
| `.tex` | `428cb794654c6afc30a24a37b054e3b9778240fb6647a9be4b642f89a51c4ea5` | 384,182 |
| ZIP de fuentes | `e9a29c80c26dab7c2139811afc051c101efdb01a359fe0e70c5d26e88b438efd` | 41,647,883 |

## Cambios de contenido (web)

Proyecto EN/ES (`build-phase-article.py`):

- Tabla del benchmark: nueva tercera fila "Geometry variant, no event inputs / Pre-registered
  second submission, 18 Nov 2026 — 23.47" (ES: "Variante geométrica, sin entradas de eventos /
  Segundo envío, prerregistrado, 18 nov 2026 — 23,47"). Las filas PHASE–TTC 35.80 y Garl-TTC 45.03 no cambian.
- "Both entries have 0% scorer-defined failures." → "All three rows have 0% scorer-defined
  failures." (ES: "Ambas entradas…" → "Las tres filas tienen un 0% de fallos según el evaluador.")
- Párrafo nuevo bajo la tabla: las dos primeras filas son el ranking archivado del 28 de octubre;
  la tercera es el segundo envío, prerregistrado y único, con un módulo distinto de los pesos
  publicados (estimaciones de la rama RGB y expansión causal de las regiones suministradas,
  pérdida ponderada por bandas, sin eventos); 23.47, 34% por debajo del sistema original y 48% por
  debajo de Garl-TTC; bandas 24.37 / 23.95 / 22.89 / 18.13 frente a 43.52 / 28.26 / 25.72 / 29.93;
  mismas regiones y 600 ms de historial de cajas; el ranking público muestra una entrada por
  participante y ahora presenta esta variante (PHASE-TTC-Geo) como primera de tres; la entrada de
  35.80 solo se conserva en el registro archivado del 28 de octubre.
- Bloque "Post-submission controls, 18 November":
  antes "Fold levels are not comparable with the test score, and this variant has not been scored on the test server."
  después "Fold levels are not comparable with the test score. Submitted once under a pre-registered protocol on
  18 November 2026, this variant scores 23.47 Overall MiD on GarlTTC Test, against 35.80 for the original system
  and 45.03 for Garl-TTC. Most of the accuracy available on this benchmark comes from the history of the supplied
  regions, and the event branch is not needed to obtain it."
  ES antes "Estos niveles no son comparables con la puntuación de test, y esta variante no se ha evaluado en el servidor de test."
  ES después "Estos niveles no son comparables con la puntuación de test. Enviada una sola vez el 18 de noviembre de 2026
  con un protocolo prerregistrado, esta variante obtiene 23,47 de Overall MiD en GarlTTC Test, frente a 35,80 del
  sistema original y 45,03 de Garl-TTC. La mayor parte de la precisión alcanzable en este benchmark procede del
  historial de las regiones proporcionadas, y la rama de eventos no es necesaria para obtenerla."
- Páginas: "pages 103–105" → "pages 103–106"; "105-page"/"105 páginas"/"PDF · 105" → 106.

Portada EN/ES (`build.py`):

- Bajo "35.80 vs. 45.03 · GarlTTC Test / Official snapshot: 28 October 2026" se añade la línea
  "Geometry variant, no event inputs (pre-registered second submission, 18 Nov 2026): 23.47"
  (ES: "Variante geométrica sin entradas de eventos (segundo envío, prerregistrado, 18 nov 2026): 23,47").
- "Methods & results" / "Métodos y resultados": se añade al final una frase con el segundo envío,
  23.47, módulo distinto, sin eventos, y la lectura permitida (la precisión procede sobre todo del
  historial de las regiones suministradas).
- "Expanded edition · 105 pages" → "106 pages" (ES: "106 páginas").

Visor (`build-paper.py`): 105 → 106 páginas (texto, `max`, contador). Subtítulo y fecha sin cambios.
`expanded-edition.json`: hashes, tamaños, 106 páginas, nota, `editorial_revision` y `source_commit`.

Se mantiene "First of three entries on the archived GarlTTC Test leaderboard (28 October 2026)"
para el 35.80. No se escribe "state of the art" ni "SOTA".

## Verificación

Antes del cambio (`verify_release.py` del directorio de trabajo, adaptado del de la entrega anterior):

- Inventario: 15 cambiados, 0 añadidos, 0 eliminados, 0 inesperados, 0 diferencias de modo o tipo.
- HTML: las 6 páginas cambiadas pasan `html.parser` sin errores, sin etiquetas sin cerrar, sin ids
  duplicados; todos los enlaces internos y fragmentos se resuelven.
- Restos: ninguno de "Rank 1", "Event–RGB Fusion", "Fusión causal", "not been scored",
  "no se ha evaluado en el servidor", "not yet archived", "no hash", "declared name",
  "105 pages/páginas/-page", "103–105", `20261119-claim-scope`, "state of the art", "SOTA".
- Todas las cadenas nuevas requeridas presentes en EN/ES. `expanded-edition.json` coincide en hash y tamaño.

Después del cambio, con curl externo:

```
200 16062  /                      (antes 15669)
200 17035  /es/                   (antes 16584)
308 → /PHASE-TTC/     200 50875   (antes 49590)
308 → /es/PHASE-TTC/  200 54292   (antes 52834)
200 4260   /papers/phase-ttc/
200 4432   /es/papers/phase-ttc/
200 2189   /sitemap.xml           (idéntico)
200 10729  alvaro.schwiedop.com/  (CV idéntico byte a byte)
```

Las 12 respuestas servidas (6 HTML, JSON, `paper-viewer.js`, portada, PDF, `.tex`, ZIP) coinciden
en sha256 con los archivos de la release. Cabeceras: `Cache-Control: no-cache`; CSP intacta.

## `~/schwiedop-site`

Es la copia de trabajo de las fuentes del sitio (su `README.md` lo documenta así). Es un repositorio
git sin commits ni remoto; las entregas del 11 de noviembre la sincronizaban copiando archivos
(`schwiedop-paper-deploy-20261111.py`, `shutil.copy2` tras copia de seguridad en `source-before/`)
y nunca hicieron commit. Estaba exactamente en el estado de `20261111T200934Z-rank1-icons`
(las entregas del 13 y 18 de noviembre no la sincronizaron).

Se sincronizó con el mismo método: copia de seguridad de los 15 archivos afectados en
`~/schwiedop-test-result-20261120T173106Z/site-before-20261118T173336Z/` y sobrescritura en el
sitio (se conservan propietario y modos). Resultado: `public/` y los cinco scripts de build idénticos
a la release nueva. Registro en `site-sync.json`. Sin commit ni push; `git status` sin cambios en el índice.

## Hugging Face

| Repositorio | Revisión anterior | Revisión nueva |
|---|---|---|
| `AlvaroSchwiedop/PHASE-TTC` (modelo) | `0b836f778eb3390a218f1a0f83ab32c7817df041` | `9c7ec9b886597cb3e744f148c51f6a16f2d29520` |
| `AlvaroSchwiedop/PHASE-TTC-Adapt` (modelo) | `e792e1639059f1db5e49c6ac6608063cd15fc5d1` | `9d30a6ccbfff1d461525e87edc74e9f4531d1e70` |
| `AlvaroSchwiedop/PHASE-TTC` (Space estático) | `faeede03f2316ef1bacc08f4e026969ccde645ca` | `2c5fc66e84f7d5421aa17b6e6543d624ffcd1bc8` |

- Tarjeta PHASE-TTC: la frase "The benchmark server stores no hash of the scored upload; that this
  release is the scored package rests on the submission's declared name and the author's statement."
  se sustituye por la verificación (`5eb69749de7aff68…`, envío 1016512, 35.80145). Párrafo nuevo
  "Second submission, 18 November 2026 (not these weights)" con 23.47, bandas, `8590f1151bb8b8db…`,
  aviso de que no son los pesos del repositorio y enlace a `docs/sota_20261118`; nota de que el ranking
  público muestra ahora PHASE-TTC-Geo y que el 35.80 solo queda en el registro archivado.
  El repositorio no contiene la variante (ni checkpoint ni paquete RGB_GEO_BAND).
- Tarjeta PHASE-TTC-Adapt: una frase tras el resultado original sobre la entrada actual del ranking
  (variante geométrica, módulo distinto, ni este adaptador ni los pesos originales).
- Eslogan de ambas tarjetas sin cambios: describe la arquitectura y no atribuye el rendimiento a la fusión.
- Space: eyebrow "Causal event + RGB fusion" → "A causal event–RGB system"; "achieved Rank 1 in the
  archived…" → "was the first of three entries on the archived… (28 October 2026)…" con la frase de
  sistema completo; párrafo nuevo con 23.47 y la nota del ranking público; "exceeds a 20 ms budget" →
  "…, and no tested configuration meets it"; README: "ranked first in the 28 October 2026 record" →
  "was the first of three entries on the archived GarlTTC Test leaderboard (28 October 2026)…" más 23.47.
  Solo se subieron `index.html` y `README.md`. Space en `RUNNING` con la revisión nueva; página y
  subdominio responden 200 y el `index.html` del repositorio coincide byte a byte con el subido.

Rollback de Hugging Face: volver a subir el `README.md`/`index.html` de la revisión anterior, p. ej.
`hf download AlvaroSchwiedop/PHASE-TTC README.md --revision 0b836f77… --local-dir prev` y
`hf upload AlvaroSchwiedop/PHASE-TTC prev/README.md README.md`. En el Space, `hf upload` falla
con un aviso de plan PRO en esta cuenta; usar `HfApi().upload_folder(..., repo_type="space")`.

## Pendiente para el propietario

- `gradio_app.zip` del Space (app local) aún dice en `app.py`: "The original model achieved Rank 1
  in the archived GarlTTC Test leaderboard on 28 October 2026". No se tocó porque es un archivo
  binario empaquetado; si se quiere corregir, hay que reconstruir el ZIP.
- En el formulario de CodaBench, el campo "Uses Event" de la entrada 1058935 está vacío; conviene
  fijarlo en `false` si se puede.
