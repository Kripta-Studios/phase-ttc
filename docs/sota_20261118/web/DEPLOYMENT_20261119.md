# Despliegue web: alcance de afirmaciones (claim scope)

Fecha de ejecución: 2026-11-18, 17:01–17:10 UTC. Los nombres de release y del
directorio de trabajo llevan el prefijo `20261119` acordado para esta entrega.
Fuente: `Kripta-Studios/phase-ttc` `main` en `96dc858bbe75f2c442b0a47e727f8d5917e17902`
(local = `origin/main` = GitHub, árbol limpio).

| | |
|---|---|
| Release anterior | `/var/www/schwiedop/releases/20261113T125514Z-extended-inference` |
| Release nueva | `/var/www/schwiedop/releases/20261119T170408Z-claim-scope` |
| Directorio de trabajo | `~/schwiedop-claim-scope-20261119T170136Z` (en el servidor) |
| Cambio | `current` cambió de forma atómica (`ln -sfn` + `mv -Tf`) |

## Rollback

```sh
ln -sfn /var/www/schwiedop/releases/20261113T125514Z-extended-inference /var/www/schwiedop/current.new && mv -Tf /var/www/schwiedop/current.new /var/www/schwiedop/current
```

La release anterior está intacta. No se modificaron Caddy, otros sitios, servicios
ni `~/schwiedop-site`.

## Procedimiento

1. `cp -a` de la release activa a la nueva. Los tres builders sin editar
   (`python3 -B build.py`, `build-phase-article.py`, `build-paper.py`, con el Python 3.12.3
   del servidor) reprodujeron la release activa byte a byte (`diff -rq`: idénticas).
2. Las entradas de los builders se editaron en local con sustituciones exactas
   (una coincidencia cada una, finales de línea CRLF conservados). Se subieron
   junto con el PDF, el `.tex` y el ZIP de fuentes de `main`, y se comprobaron con `sha256sum -c`.
3. Se instalaron en la release nueva con `cp` (se conservan los permisos de destino) y se ejecutaron los tres
   builders. La portada `phase-ttc-expanded-cover.jpg` se regeneró desde la
   página 1 del PDF nuevo (637×901, JPEG progresivo, como la anterior), porque la
   anterior mostraba el subtítulo antiguo y "Rank 1".
4. El visor "Leer en línea" es pdf.js sobre el propio PDF. `build-paper.py` actualiza
   el subtítulo, la descripción, las páginas (105) y la fecha. El parámetro de caché del PDF
   en `paper-viewer.js` pasa a `?v=20261119-claim-scope`.

## Archivos cambiados (15; ningún archivo añadido ni eliminado)

| Archivo | sha256 anterior | sha256 nuevo | modo |
|---|---|---|---|
| `build-paper.py` | `f09a20c29c5e0afa…` | `5e25054b29223a83…` | 666 |
| `build-phase-article.py` | `f0fe5541dea457fd…` | `ac239745aa055863…` | 666 |
| `build.py` | `cd992507a8136f47…` | `dd0e0112822216f1…` | 666 |
| `public/PHASE-TTC/index.html` | `4e80dd0fea060589…` | `9ce2fea6045b4e9c…` | 644 |
| `public/es/PHASE-TTC/index.html` | `a71f461db2cb8549…` | `73f3862cd92e6297…` | 644 |
| `public/index.html` | `9e9d3b4bc630bde7…` | `afdf81dedd7bed22…` | 644 |
| `public/es/index.html` | `fe17bdf83a4f2fab…` | `09fdc8985acbf5e0…` | 644 |
| `public/papers/phase-ttc/index.html` | `752bcdf409f59bc6…` | `1a85e6a119849907…` | 644 |
| `public/es/papers/phase-ttc/index.html` | `c0fa30257260fe0f…` | `b123a13bc65648b4…` | 644 |
| `public/papers/phase-ttc/expanded-edition.json` | `43545e0cc4075777…` | `b2e81e1dd92f04c2…` | 666 |
| `public/papers/phase-ttc/PHASE_TTC_From_Motion_to_Time_Expanded.pdf` | `744fb89cd80cfdce…` | `1b36d8ad6c7b674b…` | 666 |
| `public/papers/phase-ttc/PHASE_TTC_From_Motion_to_Time_Expanded.tex` | `f1486916105aedf0…` | `473fc53304e829f8…` | 666 |
| `public/papers/phase-ttc/PHASE_TTC_From_Motion_to_Time_Expanded_sources.zip` | `e414b50cce13689e…` | `305eed10cba5a69f…` | 666 |
| `public/assets/paper-viewer.js` | `c53250775a233527…` | `b68f7e9cea195c66…` | 666 |
| `public/assets/phase-ttc-expanded-cover.jpg` | `580ce2aab89a12db…` | `dd7ce5f3b15e0f97…` | 666 |

Los modos y tipos coinciden con la release anterior en todos los archivos.
`cv/`, `sitemap.xml`, `robots.txt`, CSS, JS de las páginas, figuras y vídeo no cambian.
Los hashes completos están en `verify-report.json`, en el directorio de trabajo.

Archivos del artículo (idénticos a los blobs de `origin/main`):

| Archivo | sha256 | bytes |
|---|---|---|
| PDF (105 páginas, "Updated 18 November 2026") | `1b36d8ad6c7b674b0e4aa2ad01c621b4e8371ff3bd4285a38ad89a36066b5f46` | 4,019,802 |
| `.tex` | `473fc53304e829f8a8edc445e7ea78aeb1c809ac749033cda74556a77c53cdbc` | 380,754 |
| ZIP de fuentes | `305eed10cba5a69f73975ea4cf4e8fc9da247b81f7c636770a74e15e40c5a0ad` | 41,641,075 |

## Cambios de contenido

- Subtítulo en el visor EN/ES: "Causal Event–RGB Fusion for Signed Time-to-Contact
  Estimation" → "A Causal Event–RGB System for Signed Time-to-Contact Estimation".
  La página ES ya mostraba el subtítulo en inglés, como título del artículo, y se mantiene así.
  Meta description ES: "Fusión causal de eventos y RGB…" → "Un sistema causal de eventos y RGB…".
- "Rank 1" (proyecto y portada, EN/ES) → "first of three entries on the archived
  GarlTTC Test leaderboard (28 October 2026)". Se mantiene 35.80 frente a 45.03 y se añade la
  diferencia de soporte: hasta 650 ms con las regiones suministradas, frente a dos observaciones separadas 0.1 s.
- Atribución: el resultado mide el sistema completo, y los controles emparejados no muestran
  beneficio del acceso a eventos ni de la parametrización por fase. Esto se refleja en
  la nota del benchmark, en el resumen de la portada y en la sección de transferencia
  (antes "Where fusion helps", ahora "What the controls show").
- Nuevo bloque "Post-submission controls, 18 November" en "New findings"/"Novedades":
  85.2 → 42.0 (41.3 sin eventos y con pérdida ponderada por bandas). Los niveles de los folds
  no son comparables con el test y la variante no se ha puntuado. El historial de eventos
  ajustado no aporta un beneficio medible. La historia más allá de dos observaciones supone
  unos 3 de 85 puntos. Enlace a las páginas 103–105.
- Latencia: 29.14 ms p95 en la reproducción de 5 s queda por encima de 20 ms. Con cola de 64,
  el p95 por escena es de 104–227 ms. Ninguna configuración cumple 20 ms. La cola de dos
  paquetes completó 1 de 24, cifra que ya figuraba. Capítulo de inferencia: páginas 91–102.
- Fechas, páginas y caché: 13 → 18 de noviembre de 2026, 97 → 105 páginas, `?v=20261119-claim-scope`.

## Verificación

Antes del cambio, con `verify_release.py` en el directorio de trabajo:

- Inventario: 15 cambiados, 0 añadidos, 0 eliminados, 0 inesperados y 0 diferencias de modo o tipo.
- HTML: las 6 páginas cambiadas pasan `html.parser`, sin etiquetas desparejadas ni sin
  cerrar y sin ids duplicados. Todos los enlaces internos (`href`/`src`/`poster`) y
  fragmentos se resuelven dentro de la release.
- Restos: no queda "Rank 1", "Event–RGB Fusion", "Fusión causal", "97 pages/páginas",
  "13 November" ni `20261113-inference` en los archivos de texto cambiados.
  Repeticiones intencionadas: el PDF conserva "Rank 1" en el gráfico de la Figura 1 (página 2)
  del propio manuscrito. "fusion head", "RGB-anchored fusion" y otros nombres de
  componentes siguen en las descripciones de la arquitectura.
- `expanded-edition.json`: los hashes y tamaños coinciden con los archivos publicados.

Después del cambio, con curl externo:

```
200 15669  /                       (antes 15445; +224 B solo por el bloque PHASE-TTC)
200 16584  /es/                    (antes 16322; +262 B, ídem)
308 → /PHASE-TTC/      200 49590   (antes 48584)
308 → /es/PHASE-TTC/   200 52834   (antes 51638)
200 4260   /papers/phase-ttc/      (antes 4251)
200 4432   /es/papers/phase-ttc/   (antes 4422)
200 2189   /sitemap.xml            (idéntico)
200 10729  alvaro.schwiedop.com/   (CV idéntico byte a byte)
PDF servido     sha256 1b36d8ad6c7b674b0e4aa2ad01c621b4e8371ff3bd4285a38ad89a36066b5f46 = local
.tex servido    sha256 473fc53304e829f8a8edc445e7ea78aeb1c809ac749033cda74556a77c53cdbc = local
ZIP servido     sha256 305eed10cba5a69f73975ea4cf4e8fc9da247b81f7c636770a74e15e40c5a0ad = local
```

Las 9 respuestas servidas (6 HTML, JSON, `paper-viewer.js` y portada) coinciden en
sha256 con los archivos de la release. Las cadenas nuevas están presentes en EN/ES.
Cabeceras sin cambios: `Cache-Control: no-cache` en páginas y PDF; CSP del portfolio intacta.

## No cambiado

- Figuras `phase-publication-*-20261111.png`: diagramas de arquitectura, no afirmaciones;
  regenerarlas cambiaría assets fuera del alcance pedido.
- El aviso fechado del 3 de noviembre en la portada y las descripciones de la
  arquitectura ("Fuse explicit quantities", "fusion head"), que describen el mecanismo,
  no atribuyen el resultado.
- `~/schwiedop-site`, que las entregas del 11 de noviembre sincronizaban, queda fuera
  de las rutas permitidas. La fuente vigente son los builders de la release nueva
  (copias anteriores en `source-before/`).
