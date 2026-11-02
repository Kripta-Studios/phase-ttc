# Viabilidad de MonoCoP para una comparación posterior

La investigación ya avanzó a GPU: el gate técnico se recuperó mediante replay
exacto en un proceso independiente, sin repetir entrenamiento, y se registró
el [primer screening de calidad](MONOCOP_SCREEN_PLAN_20261101.md).
Los fallos técnicos, la recuperación y las copias verificadas están en
[la evidencia GPU](evidence/monocop_gpu_execution_20261101/ARCHIVE.json).
Las restricciones y observaciones siguientes describen sus etapas anteriores;
no sustituyen los estados reales del nuevo ensayo.

Actualización posterior a la revisión estática: el [port CPU v2](evidence/monocop_cpu_port_20261101/report/REPORT.md)
pasó construcción, atención, forward/backward, calibración y replay en 56,126 s.
El intento v1 falló por dimensiones de la pirámide y se conserva con su corrección.
Estas pruebas usan pesos aleatorios y cero pasos de optimizador; no miden mAP
ni admiten otro entrenamiento GPU. El texto siguiente documenta la inspección
estática anterior, cuyo recibo permanece intacto.

También se completó posteriormente la [transferencia CPU de los pesos multiclase](evidence/monocop_transfer_20261101/report/REPORT.md).
La descarga quedó fijada por revisión y SHA-256, y las tres clases se mapearon
explícitamente a eAP. Pasaron gradientes, datos TRAIN reales y replay exacto,
incluida una corrección de API con igualdad del checkpoint y todos los resultados
numéricos. Esa transferencia CPU no midió mAP ni admitió por sí sola una cola GPU.

MonoCoP queda como un competidor condicional. Esta revisión estática no admite
otro entrenamiento: primero deben terminar los ensayos de rango e inicialización
KITTI ya registrados. No hay una métrica eAP de MonoCoP ni evidencia de que vaya
a superar al candidato actual.

Se revisa `alanzhangcs/MonoCoP` en el commit
`7d3932444f7ea94be564ae61a0b334fc6673213b`. El recibo conserva URL, Git blob SHA-1,
SHA-256 y tamaño de cada archivo descargado. No se importó ese código, no se
construyó el modelo y no se descargaron pesos durante esta revisión.

El [resultado de la inspección](evidence/monocop_static_feasibility_20261101/AUDIT.json)
verifica los 65 archivos recibidos y registra las líneas de cada hallazgo; el
[recibo original](evidence/monocop_static_feasibility_20261101/SOURCE_RECEIPT.json)
permite recuperar los mismos bytes desde la revisión fijada.

El artículo correcto es [Unleashing the Power of Chain-of-Prediction for Monocular
3D Object Detection](https://arxiv.org/abs/2505.04594), enlazado por la
[página de los autores](https://alanzhangcs.github.io/monocop-page/). El enlace del
README fijado apunta por error a arXiv:2503.11651, que corresponde a VGGT.
Las afirmaciones de resultados en KITTI, nuScenes y Waymo pertenecen a sus autores;
no son resultados nuestros ni una comparación en eAP.

## Qué aporta y qué cambia

El código encadena características de tamaño, orientación y profundidad. En
entrenamiento elige por grupo entre dos ramas de regresión de profundidad; en
inferencia elige según la incertidumbre de la rama encadenada. Ambas ramas suman
el mismo término geométrico derivado del tamaño. Por tanto, la rama paralela no
elimina toda dependencia del tamaño estimado.

También cambia el backbone: su ResNet-vd usa tres convoluciones iniciales y pesos
SSLD convertidos de Paddle. Nuestro release construye el ResNet de torchvision y
carga los pesos del checkpoint propio. Una comparación de ambos sistemas no
aislaría el efecto del encadenamiento: arquitectura e inicialización cambiarían.

El [inventario oficial de pesos](https://huggingface.co/zhihao406/MonoCoP/tree/91e1119b4463a83a9f64b7c2838be4bc5314834f)
contiene `MonoCoP_All.pth` y `MonoCoP_Car.pth`, aproximadamente 502 MB cada uno.
El model zoo del código anuncia versiones KITTI Car/Pedestrian/Cyclist y Car.
La primera es una candidata más pertinente para investigar clases pequeñas que
otra inicialización exclusivamente Car. Es una hipótesis: no se descargaron ni
inspeccionaron los tensores de esos checkpoints. Se descargó únicamente el README
fijado y se verificó su Git blob; véase el
[recibo de catálogo](evidence/monocop_static_feasibility_20261101/HF_INTAKE.json).
La ficha muestra un distintivo MIT, pero el inventario del repositorio de pesos
no incluye un archivo LICENSE; no se certifica por ello la licencia de los pesos.

## Adaptaciones necesarias

- La configuración nuScenes incluida usa tres clases, evaluación KITTI y lista
  de salida `car`. No es una receta lista para las ocho clases eAP.
- La pérdida DDNLoss discretiza con valores por defecto de 60 m y 80 bins; su
  `forward` no pasa el rango configurable del predictor. Cambiar solo el YAML
  dejaría desalineados objetivos y predictor.
- La profundidad geométrica usa `fx` con altura vertical. Nuestro MonoDGP
  congelado ya usa `fy`. El port debe respetar calibración, reescalado anisótropo,
  centro de píxel y transformación ego/cámara; no hay una corrección nueva que
  aplicar retrospectivamente al candidato actual por este hallazgo.
- El constructor de DDNLoss consulta CUDA y el backbone descarga pesos
  incondicionalmente, aunque recibe un argumento `pretrained`. Se necesita carga
  local explícita y construcción CPU sin efectos de red.
- La atención importa una extensión compilada. Hay implementación PyTorch de
  referencia, pero todavía no se han medido su paridad, gradientes ni coste en
  este modelo y equipo.
- El decodificador nativo añade dimensiones medias, convierte centro geométrico
  a base de caja y multiplica la confianza por un factor de incertidumbre. Estas
  convenciones deben reconciliarse con el contrato eAP y el ranking registrado;
  copiar las salidas KITTI directamente no constituye una adaptación correcta.

El README recomienda Python 3.9, PyTorch 2.4.1, torchvision 0.19.1 y CUDA 12.1.
No se da por comprobada su compatibilidad con nuestro entorno. La licencia raíz
es MIT; los archivos derivados de atención conservan avisos Apache 2.0. Hay que
preservar esos avisos. Esto no certifica licencias de pesos ni su admisión en el
benchmark.

## Criterios antes de gastar GPU

Si los ensayos pendientes no producen un candidato promovible, preparar una sola
adaptación aislada: ocho clases, contrato geométrico eAP, rango coherente entre
targets y predictor, pesos con procedencia fijada y carga sin red. Primero deben
pasar construcción CPU, forward y backward finitos, cajas sintéticas con
`fx != fy`, fronteras de bins y reproducción del checkpoint.

Después se necesita un ensayo corto que mida VRAM, tiempo y paridad de la
atención antes de registrar población, control, receta, presupuesto y umbral de
promoción. No se fija una ETA ni calidad antes de esa medición. Las arquitecturas
y los historiales de preentrenamiento deben figurar en la comparación; si se
quisiera atribuir la mejora al encadenamiento, haría falta una ablation separada.

## Reproducción de la inspección

```powershell
python scripts/audit_detection_monocop_source.py `
  --intake D:/PHASE_TTC_COMPETITOR_20261101/monocop_7d393244 `
  --release D:/PHASE_TTC_RELEASE_20261101/dfull-rgb-ranked-seed7 `
  --output D:/PHASE_TTC_COMPETITOR_20261101/audit_replay/AUDIT.json
```

El verificador usa solo la biblioteca estándar. Comprueba los archivos contra el
recibo, localiza los anclajes revisados y contrasta el código del candidato actual
con su manifest inmutable. Los anclajes documentan la inspección; no sustituyen
pruebas de ejecución. El archivo de salida debe ser nuevo.
