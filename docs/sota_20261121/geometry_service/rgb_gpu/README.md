# Lectura y verificación

Este directorio contiene los resúmenes legibles de la campaña. `MANIFEST.json` es una copia
del manifiesto sellado del bundle de ejecución; sus ocho trazas por oferta no se duplican aquí.

El paquete portable completo está en
`../rgb_gpu_long_contended_traces.zip`, acompañado de su `.sha256`. Contiene el manifiesto,
la población física, las ocho trazas, resultados, informe y evaluación del gate. Se verifica
con:

```powershell
python -X utf8 -m operational.geometry_service.verify `
  docs/sota_20261121/geometry_service/rgb_gpu_long_contended_traces.zip
```
