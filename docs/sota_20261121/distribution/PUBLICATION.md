# Estado de distribución a 21 de noviembre de 2026

Este directorio parte de estados públicos leídos el 21 de noviembre, sin publicar
cambios:

| Destino | Revisión pública fuente |
|---|---|
| GitHub `Kripta-Studios/phase-ttc` | `main@90040a1b1de5f76d26bdc2c2491ff013d4a2ea1c` |
| Modelo `AlvaroSchwiedop/PHASE-TTC` | `9c7ec9b886597cb3e744f148c51f6a16f2d29520` |
| Modelo `AlvaroSchwiedop/PHASE-TTC-Adapt` | `9d30a6ccbfff1d461525e87edc74e9f4531d1e70` |
| Space `AlvaroSchwiedop/PHASE-TTC` | `2c5fc66e84f7d5421aa17b6e6543d624ffcd1bc8` |
| Web activa | `/var/www/schwiedop/releases/20261120T173106Z-test-result` |

Los materiales añaden el alcance A/B1/B2/C de la reproducción Geo, el tercer envío
`1063261` de solo cajas, la brecha de 4,50325 puntos y las cuatro bandas, además de
la auditoría de cajas, FCWD y latencia aislada. También corrigen `gradio_app.zip`.
No se afirma una posición pública posterior al tercer envío. Los enlaces nuevos se
fijan a la revisión pública verificada
`8fa8dda64bcdf846331191eb4ad4c8171b979699`; no usan `tree/main`.

Los cinco assets de Hugging Face se publicaron y se leyeron de nuevo como usuario
el 21 de noviembre. Sus revisiones resultantes y SHA-256 leídos están en
[`HF_PUBLISHED.json`](HF_PUBLISHED.json). La web permanece preparada en staging,
pero todavía no desplegada; la release activa indicada arriba sigue siendo el punto
de rollback. Si SSH continúa devolviendo el banner y agotando el tiempo, el
directorio de staging validado constituye el paquete web homologado para que el
propietario lo transfiera y active con su inventario y cambio de symlink.

## Web

`scripts/stage_qualification_web.py` copia y parchea de forma fail-closed los tres
builders, `paper-viewer.js` y `expanded-edition.json`; cuenta las páginas reales,
instala el PDF ampliado, el `.tex`, el ZIP de fuentes y el PDF corto, y emite
`STAGING_MANIFEST.json` con hashes. No activa ninguna release remota.

Comando de staging desde PowerShell (la comprobación debe devolver exactamente el
SHA fijado antes de continuar):

```powershell
$publicCommit = '8fa8dda64bcdf846331191eb4ad4c8171b979699'
$remoteCommit = (git ls-remote origin refs/heads/research/phase-ttc-qualification-20261121).Split("`t")[0]
if ($remoteCommit -ne $publicCommit) { throw "public revision mismatch: $remoteCommit" }
$stage = Join-Path $env:TEMP ('phase-qualification-web-stage-' + [guid]::NewGuid().ToString('N'))
& '..\phase-ttc-main-integration\.venv\Scripts\python.exe' scripts/stage_qualification_web.py `
  --source-dir "$env:TEMP/phase-qualification-web-20261121" `
  --output-dir $stage `
  --expanded-pdf output/pdf/PHASE_TTC_From_Motion_to_Time_Expanded.pdf `
  --expanded-tex output/pdf/PHASE_TTC_From_Motion_to_Time_Expanded.tex `
  --expanded-zip output/pdf/PHASE_TTC_From_Motion_to_Time_Expanded_sources.zip `
  --short-pdf output/pdf/PHASE_TTC_From_Motion_to_Time_Short.pdf `
  --public-commit $publicCommit
```

El manifiesto de QA más reciente detectó 109 páginas en la edición ampliada y 9
en la corta. Los hashes deben tomarse siempre del manifiesto de la ejecución que
se vaya a desplegar, porque la construcción editorial puede cambiar. El activador
remoto, su inventario y el rollback por symlink los prepara el propietario; la
release anterior que debe conservarse es
`/var/www/schwiedop/releases/20261120T173106Z-test-result`.

## Hugging Face

Estado: **publicado y verificado por lectura**. Revisión del modelo principal
`5d1b1e7baeb10281cfcd5f177d6d7e86e9201543`, revisión del modelo Adapt
`513ed0331949863546c72d6fc36c368e2c63e5ac` y revisión del Space
`40b79c547746efac2caed3bf8e01eb5b813f8d50`. El recibo legible por máquina es
[`HF_PUBLISHED.json`](HF_PUBLISHED.json).

`HF_ASSET_DIFF.json` limita el cambio a cinco rutas remotas y registra para cada
una revisión fuente, SHA-256 anterior, SHA-256 preparado y tamaño. Es también el
inventario de rollback; no se debe sustituir por una subida de carpeta.

Revisar primero que los remotos sigan en las revisiones fuente de la tabla. Después
de publicar el commit de evidencia en GitHub, ejecutar desde la raíz del repositorio:

```powershell
hf upload AlvaroSchwiedop/PHASE-TTC docs/sota_20261121/distribution/hf/PHASE-TTC/README.md README.md --type model --commit-message "Document Geo reproduction scope"
hf upload AlvaroSchwiedop/PHASE-TTC-Adapt docs/sota_20261121/distribution/hf/PHASE-TTC-Adapt/README.md README.md --type model --commit-message "Document Geo reproduction scope"
hf upload AlvaroSchwiedop/PHASE-TTC docs/sota_20261121/distribution/space/README.md README.md --type space --commit-message "Document qualification and reproduction scope"
hf upload AlvaroSchwiedop/PHASE-TTC docs/sota_20261121/distribution/space/index.html index.html --type space --commit-message "Document qualification and reproduction scope"
hf upload AlvaroSchwiedop/PHASE-TTC docs/sota_20261121/distribution/space/gradio_app.zip gradio_app.zip --type space --commit-message "Correct local app scope"
```

Si `hf upload --type space` rechaza la cuenta, usar `HfApi.upload_file` con
`repo_type="space"` para esos tres archivos exactos. No subir el directorio entero
ni `space/gradio_app/`; este último se conserva solo como fuente revisable del ZIP.

Verificación posterior:

```powershell
hf models info AlvaroSchwiedop/PHASE-TTC --format json
hf models info AlvaroSchwiedop/PHASE-TTC-Adapt --format json
hf spaces info AlvaroSchwiedop/PHASE-TTC --format json
hf cache verify AlvaroSchwiedop/PHASE-TTC --type model --revision <NEW_MODEL_SHA> --local-dir docs/sota_20261121/distribution/hf/PHASE-TTC
```

Para rollback, descargar por revisión los archivos anteriores y volver a subirlos,
lo que crea revisiones nuevas con el contenido anterior sin tocar otros assets:

```powershell
$rollback = Join-Path $env:TEMP ('phase-hf-rollback-' + [guid]::NewGuid().ToString('N'))
hf download AlvaroSchwiedop/PHASE-TTC README.md --type model --revision 9c7ec9b886597cb3e744f148c51f6a16f2d29520 --local-dir "$rollback/model"
hf download AlvaroSchwiedop/PHASE-TTC-Adapt README.md --type model --revision 9d30a6ccbfff1d461525e87edc74e9f4531d1e70 --local-dir "$rollback/adapt"
hf download AlvaroSchwiedop/PHASE-TTC README.md index.html gradio_app.zip --type space --revision 2c5fc66e84f7d5421aa17b6e6543d624ffcd1bc8 --local-dir "$rollback/space"
hf upload AlvaroSchwiedop/PHASE-TTC "$rollback/model/README.md" README.md --type model --commit-message "Rollback card to 9c7ec9b"
hf upload AlvaroSchwiedop/PHASE-TTC-Adapt "$rollback/adapt/README.md" README.md --type model --commit-message "Rollback card to 9d30a6c"
hf upload AlvaroSchwiedop/PHASE-TTC "$rollback/space/README.md" README.md --type space --commit-message "Rollback Space README to 2c5fc66"
hf upload AlvaroSchwiedop/PHASE-TTC "$rollback/space/index.html" index.html --type space --commit-message "Rollback Space index to 2c5fc66"
hf upload AlvaroSchwiedop/PHASE-TTC "$rollback/space/gradio_app.zip" gradio_app.zip --type space --commit-message "Rollback local app to 2c5fc66"
```

Ningún comando de esta página se ejecutó contra los remotos.
