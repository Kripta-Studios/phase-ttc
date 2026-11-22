# PHASE-TTC · Traspaso para continuar hacia el estado del arte

Corte del 22 de noviembre de 2026, rama `research/phase-ttc-qualification-20261121`. Este directorio es el punto de entrada para quien retome el trabajo. Contiene el contexto mínimo, las reglas, el entorno y un paquete de trabajo por cada línea abierta. El texto para lanzar el enjambre está en [PROMPT.md](PROMPT.md).

## 1. Objetivo

Que PHASE-TTC pueda defenderse como estado del arte en estimación de tiempo hasta el contacto (TTC). Hoy eso significa dos cosas distintas:

- **Con regiones suministradas** (benchmark GarlTTC): ya somos primeros con 23,47 Overall MiD. Falta demostrar que la ventaja es del método y no de la información usada, que se transfiere a datos independientes y que cabe en el plazo de 20 ms.
- **Sin regiones** (localizar, seguir y estimar desde el sensor): la cadena de eventos del piloto solo dio un TTC bien asociado en 3 de 165 observaciones. Falta casi todo.

## 2. Estado en una tabla

| Hecho | Valor | Fuente |
|---|---|---|
| Garl-TTC, referencia publicada en test | 45,03 MiD | [registro de envíos](../../sota_20261118/SUBMISSION_LEDGER.md) |
| PHASE-TTC original, envío `1016512` | 35,80 | ídem |
| RGB + geometría, envío `1058935` | **23,47** | ídem |
| Solo cajas, envío `1063261` | 27,98 | ídem |
| Garl eventos + cajas frente a PHASE RGB + cajas, en pliegues | 0,70 [−0,31; 1,75], Garl dentro de muestra | [comparador](../../sota_20261118/matched_comparator/REPORT.md) |
| Geometría robusta + escala de contenido sobre RGB, en pliegues | −2,42 [−2,90; −1,93] frente a RGB + geometría | [escala de contenido](../../sota_20261118/saturation_scale/REPORT.md) |
| Puerta de fiabilidad, en pliegues de cabeza | −0,57 [−0,77; −0,38] | [puerta](../reliability_gate/REPORT.md) |
| FCWD, mejor brazo propio (desarrollo expuesto) | 16,1 % RTE; Garl local 26,8 % | [revisión](../../sota_20261118/REVIEW_20261119.md) |
| Latencia RGB + geometría, cómputo aislado | 28,8 ms de mediana | [latencias](../../sota_20261118/variant_latency/REPORT.md) |
| Servicio de solo cajas en CPU | 23.992 de 24.000 ofertas en 20 ms | [servicio](../geometry_service/REPORT.md) |
| TTC por objeto desde eventos | 3 de 165 observaciones | [piloto](../per_object_events/REPORT.md) |

## 3. Lectura obligatoria, en este orden

1. `AGENTS.md`, sección 3 (integridad científica). Es contrato.
2. [Hallazgos y límites](../HALLAZGOS_Y_LIMITES.md) y [Trabajo realizado](../TRABAJO_REALIZADO.md).
3. [Revisión independiente del 19 de noviembre](../../sota_20261118/REVIEW_20261119.md): qué se puede afirmar y qué no.
4. Los dos planes: [benchmark](../COMO_CONTINUAR_BENCHMARK.md) y [sin cajas](../COMO_CONTINUAR_SIN_CAJAS.md).
5. [ADR-0047](../../decisions/ADR-0047-garlttc-submission-discipline.md): disciplina de envíos al test.
6. El paquete de trabajo que te toque y los informes que enlaza.

## 4. Paquetes de trabajo

| Paquete | Qué resuelve | Depende de | Complejidad |
|---|---|---|---|
| [WP1](WP1_COMPARACION_EXTERIOR.md) | PHASE frente a Garl-TTC con la misma información, pliegues exteriores completos | disco y GPU | alta |
| [WP2](WP2_CANDIDATO_TEST.md) | Un candidato nuevo y prerregistrado para el test | WP1 (productores por pliegue) y WP3 | alta |
| [WP3](WP3_IDEAS_DE_MODELO.md) | Mejoras de la cabeza medidas en pliegues | nada | media |
| [WP4](WP4_COHORTE_INDEPENDIENTE.md) | Datos independientes para afirmar generalización | nada | alta |
| [WP5](WP5_LATENCIA.md) | Modelo ganador dentro de 20 ms | GPU libre | media |
| [WP6](WP6_TOLERANCIA_A_CAJAS.md) | Cuánto ruido de caja toleran las cabezas | nada | media |
| [WP7](WP7_DETECTOR.md) | Detector y seguidor entrenados | WP6 | alta |
| [WP8](WP8_CABEZAS_Y_EVALUACION_FINAL.md) | Cabezas con cajas de detector y evaluación sellada | WP7 | alta |

Pueden arrancar en paralelo WP3, WP4, WP6 y la parte de preparación de WP1. WP5 y el entrenamiento de WP1 compiten por la GPU y no deben solaparse: las medidas de latencia con la GPU ocupada no valen.

## 5. Reglas que no admiten excepción

**Integridad**
- No inventar resultados ni escribir cifras a mano: toda tabla sale de ficheros generados por código.
- Antes de calcular cualquier comparación, escribir `PROTOCOL.md` con brazos, métrica, regla de decisión y población. Las desviaciones se declaran en el informe.
- Particiones siempre por secuencia completa. Intervalos por remuestreo de secuencias completas, no de consultas.
- Un resultado negativo se informa igual que uno positivo.
- No leer etiquetas privadas del test. No abrir etiquetas de un conjunto sellado antes de fijar las predicciones por huella.
- FCWD, EvTTC y TRAIN40 son datos de desarrollo ya expuestos. Nada de lo medido en ellos es una confirmación independiente.
- No usar la expresión «estado del arte» en ningún informe sin la comparación reproducida que la sostenga.

**Envíos al test**
- Nadie sube nada a CodaBench salvo el propietario. Los agentes solo preparan el paquete y su recibo con `scripts/archive_submission.py`.
- Un candidato solo se empaqueta si cumple la regla escrita antes de medirlo.

**Git**
- Los subagentes no ejecutan `git add`, `git commit`, `git push` ni ninguna otra escritura de git.
- Solo el orquestador hace commits, con autor Kripta-Studios y sin ninguna línea de atribución a Claude ni a otra herramienta. El remoto ya usa la clave `github-kripta`.
- Commits pequeños, un asunto por commit, en la rama de trabajo. No fusionar a `main` sin orden del propietario.

**Código**
- Python tipado, `ruff check` y `pytest` limpios con el `.venv` del repositorio.
- Código nuevo en `operational/<paquete>/`, pruebas en `tests/unit/test_<paquete>_*.py`, informes en `docs/sota_20261121/<paquete>/`.
- Sin rutas absolutas ni nombres de usuario en ficheros versionados: las raíces de datos entran por variables de entorno.
- Informes en español; código e identificadores en inglés.

**Privacidad y exterior**
- No enviar datos del proyecto ni del propietario a servicios externos. Descargar datos públicos está permitido; darse de alta, aceptar licencias o contactar con terceros requiere al propietario.

## 6. Entorno

**Repositorio.** Es un árbol de trabajo enlazado de git. Trabajar siempre desde su raíz.

**Intérpretes**
- `.venv\Scripts\python.exe` del repositorio: `pytest`, `ruff`, `numpy`. Su `torch` no funciona.
- `..\phase-ttc-main-integration\.venv\Scripts\python.exe`: `torch` con CUDA. Es el que se usa para entrenar e inferir.
- OpenCV no está en ese entorno; se añade por `PYTHONPATH` desde un directorio `runtime_deps` (ver sección 11 del [informe del comparador](../../sota_20261118/matched_comparator/REPORT.md)).

**Variables de entorno de datos** (los valores están en los informes citados; no escribirlos en el código)

| Variable | Contenido |
|---|---|
| `PHASE_TTC_GARLTTC_DATASET` | Parquets públicos de GarlTTC y entradas del test |
| `PHASE_TTC_GARLTTC_BENCHMARK` | Código oficial del benchmark y evaluador |
| `PHASE_TTC_EAP_RAW_ROOT`, `PHASE_TTC_EAP_ROOT`, `PHASE_TTC_EAP_DATASET` | eAP crudo: `data/train/<secuencia>/events.h5` e imágenes |
| `PHASE_TTC_POD_RETURN_ROOT` | Modelos y características selladas del sistema original |
| `PHASE_TTC_FCWD_ROOT` | Secuencias FCWD |
| `PHASE_TTC_V12_ARTIFACTS` | Punto de control verificado de Garl-TTC completo |
| `PHASE_TTC_BASELINES_ROOT` | Clones y pesos de los métodos de la literatura |

**Máquina.** Portátil con Ryzen 9 9955HX, 31 GiB de RAM y RTX 5070 Ti de 12 GiB. Es compartida:
- Memoria: lanzar como mucho 4 procesos pesados a la vez. Una ejecución anterior con 12 procesos dejó el sistema sin memoria.
- **Disco: es el cuello de botella.** Al corte hay unos 57 GB libres en `C:` y 31 GB en `D:`. WP1 pide 57 GB temporales y WP4 y WP7 descargan o generan datos. Antes de lanzar nada grande, medir el espacio libre y, si no alcanza, parar y pedir al propietario que libere o indique otro disco. No borrar nada existente por iniciativa propia.
- Trabajo temporal en `artifacts/sota_20261121/<paquete>/` (ignorado por git) o en un directorio nuevo `D:\PHASE_TTC_<PAQUETE>_<fecha>`.

**Artículo.** Fuentes en `paper/from_motion_to_time_20261102/`; se compila con `scripts/build_expanded.py` y se empaqueta con `scripts/package_expanded.py`. No tocar el artículo hasta que el revisor final apruebe los resultados.

## 7. Modelos de los agentes

- Implementaciones complejas: Opus 5.5 con esfuerzo alto.
- Implementaciones sencillas y búsquedas: Opus 5.5 con esfuerzo medio.
- Revisión final independiente, de solo lectura: Fable 5.2 con esfuerzo medio. Se le da el código y los ficheros de resultados, no las conclusiones.

## 8. Qué debe entregar cada paquete

En `docs/sota_20261121/<paquete>/`:
- `PROTOCOL.md`, escrito antes de medir, con su SHA-256 citado en el informe.
- `REPORT.md` generado o respaldado por `RESULTS.json`, con: veredicto frente a la regla previa, tablas con intervalos, coste, desviaciones y comandos de reproducción.
- Sello (`seal.json`) cuando haya un conjunto que se abre una sola vez.

Y un mensaje final al orquestador con el veredicto, las tablas principales, las desviaciones y las rutas de lo escrito.

## 9. Cierre

Cuando terminen los paquetes lanzados, el orquestador:
1. Verifica pruebas y `ruff`, y busca rutas privadas en lo nuevo.
2. Hace los commits y los sube.
3. Lanza la revisión final y registra su dictamen en `docs/sota_20261121/REVIEW_<fecha>.md`.
4. Actualiza [Hallazgos y límites](../HALLAZGOS_Y_LIMITES.md) y los dos planes con lo que haya cambiado.
5. Informa al propietario de las decisiones que le corresponden: envíos al test, altas en servicios, disco.
