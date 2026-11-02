# Plan de investigación y mensaje para Codex: estado al 2 de noviembre de 2026

## 1. Situación que debe respetar la siguiente sesión

La ronda terminó a petición del usuario y la investigación quedó pausada.
Este documento prepara una posible reanudación; no autoriza a lanzar entrenamientos
solo por leerlo. La solicitud posterior pidió documentación, git add, commit y push.

El candidato conservado es Dfull RGB + ranker TRAIN:
0.06204229 mAP en selección completa alineada y
0.05129508 AP oficial anterior. MonoCoP terminó en
0.03476030 sobre el panel, cuyo incumbente logra
0.06533347. No hay un modelo cercano a 1,0.

Leer primero [ESTADO_RESULTADOS_Y_ARTEFACTOS_EAP_20261102.md](ESTADO_RESULTADOS_Y_ARTEFACTOS_EAP_20261102.md).
La implementación y evidencia del cierre llegan a 543fa399.
La rama de trabajo es dev/eap-diagnostics-20261101; el repositorio es
Kripta-Studios/phase-ttc. Verificar el HEAD remoto y local al reanudar.

## 2. Qué se consiguió y qué sigue sin resolver

Completamos el diagnóstico raw, la auditoría de rango, las comparaciones de ranking,
la confirmación 150 m, el control 300 m, la transferencia KITTI, el port y screening
MonoCoP, la recuperación tras reinicio y las dos auditorías geométricas.
Conservamos un paquete ejecutable, checkpoints, resultados por clase, fuentes y hashes.

Ninguna variante nueva superó al candidato conservado. El fallo dominante combina
profundidad/centros, cobertura por rango y clasificación de clases raras.
Los oracles orientan el diagnóstico; no representan mejoras ya conseguidas.

La inversión geométrica eliminó el sesgo con targets perfectos, pero su uso directo
con atributos predichos empeoró TRAIN: 0.21064779 →
0.16478485. No trasladar la conclusión del caso
ideal a inferencia ni añadir una corrección al residual de un checkpoint antiguo
sin comprobar qué compensación ya aprendió.

## 3. Experimentos cerrados: no repetir por rutina

| Intervención | Conclusión de esta ronda | Evidencia |
|---|---|---|
| Expansión de clases, top 50 y rankers lineal/MLP | Sin mejora macro suficiente | multiclass_20261101 y query_quality_20261101 |
| 150 m | Mejor que el control sin ranker, inferior al candidato conservado en selección completa | range150_confirmation_20261101 |
| 300 m | Inferior al incumbente y al mejor 150 m en el panel; sin confirmación completa | range300_screen_20261101 |
| Combinación por profundidad y unión | Sin mejora completa | range_ensemble_20261101 |
| Inicialización KITTI MonoDGP | Screening negativo bajo el presupuesto registrado | kitti_screen_20261101 |
| MonoCoP | Screening negativo; arquitectura y pesos cambiaron a la vez | monocop_screen_final_20261102 |
| Inversión geométrica directa | Empeora la salida aprendida en TRAIN | predicted_geometry_20261102 |
| Prueba semántica RGB | Demasiado débil y pequeña para justificar una campaña | semantic_probe_20261101 |

Las carpetas citadas están bajo docs/detection_sota_20261029/evidence.
No generalizar estos negativos a toda temporalidad, eventos, mayor resolución,
otras arquitecturas o recetas más largas: no las hemos descartado en general.
Tampoco ampliar épocas por la sola caída de loss TRAIN.

## 4. Primer trabajo si el usuario reanuda

1. Verificar el worktree, modificaciones ajenas, HEAD y recibos de entrega.
2. Confirmar que no existe un proceso escritor vivo. La presencia de un lock no
   acredita actividad: comprobar PID y fecha de creación.
3. Usar el run final monocop_screen_resume_v1. El STATUS de monocop_screen_v1
   quedó dañado en el reinicio y se preserva como evidencia, no como estado vigente.
4. Verificar los hashes de pesos y del perfil antes de cargar un productor.
5. Reutilizar los JSON y sidecars de predicciones. Repetir inferencia solo si un
   cambio concreto requiere salidas que aún no se guardaron.
6. Declarar una hipótesis falsable, población, control, receta, presupuesto y gate
   antes de reservar GPU. No relajar contratos congelados para hacer pasar un run.

El script de handoff de estado admite --monocop-run para señalar el run final.
Los snapshots antiguos contienen decisiones que entonces estaban pendientes.
No interpretar esas frases históricas como una cola todavía activa.

## 5. Hipótesis priorizadas, aún no ejecutadas

### A. Geometría consistente durante el aprendizaje

La inversión por vértices justifica estudiar otra parametrización, no reemplazar
la profundidad de un checkpoint como si fuera una corrección demostrada.
Antes de una campaña, separar con outputs TRAIN guardados el efecto del error en
tamaño, yaw, centro proyectado y bordes. Examinar recortes, oclusión y dispersión
entre las profundidades inferidas por cada borde.

Una variante entrenable debe declarar cómo reinicializa o conserva el residual,
cómo trata bordes inválidos y qué supervisión geométrica usa. El control debe compartir
datos, inicialización, rango, resolución y presupuesto. Cualquier tensor derivado de
GT pertenece a la pérdida o al diagnóstico, nunca al forward de inferencia.
Primero exigir pérdida finita, gradientes útiles y replay del checkpoint; después,
medir mAP alineada y errores a 0,5/1/2/4 m en el mismo panel.

### B. Clases raras y diversidad de datos

El diagnóstico muestra confusión semántica además de falta de cobertura.
La diferencia TRAIN/selección no distingue por sí sola memorización, distribución
de escenas y ruido. Antes de otro sampler, revisar IDs por secuencia, tamaño,
visibilidad y asociación. No contar ventanas correlacionadas como objetos nuevos.

Un balance nuevo necesita cambiar una causa documentada y mantener ocho clases en
la evaluación. No repetir R2-rare o las mismas cabezas de logits con otro nombre.
No elegir una variante porque mejora una clase rara si deteriora el resultado macro
sin una justificación y un criterio registrados.

### C. Resolución, representación o contexto temporal

La sensibilidad geométrica de cajas pequeñas permite plantear un ensayo de resolución,
si cabe en el presupuesto y conserva la calibración de resize.
Una comparación temporal o con eventos debe separar el valor del contexto del
objetivo JEPA y usar información disponible de forma causal en inferencia.
Los screenings históricos no constituyen un descarte general de esas vías.
No portar simultáneamente otro backbone, pérdida, resolución y sampler si se quiere
atribuir la mejora a un componente.

### D. Duración de entrenamiento

KITTI y MonoCoP alcanzaron su mejor punto al final de sus screenings, pero ambos
quedaron lejos del modelo conservado. Eso no demuestra convergencia ni que ampliar
el entrenamiento sea rentable. Una extensión futura necesita curva por clase,
cobertura suficiente, señal de validación y un presupuesto nuevo declarado.
El buen resultado del smoke sobre dos imágenes no estima calidad ni ETA de una época.

## 6. Objetivos y reglas de decisión

Una mAP de 1,0 es una aspiración, no una expectativa respaldada por estos datos.
Superar 0,10 en selección completa alineada puede servir como objetivo de investigación;
no es un resultado ni una promesa. El primer objetivo operativo sería superar al
mejor candidato con una ganancia defendible y sin depender solo de car.

Para una campaña comparable, predefinir antes de ejecutar:

- Control y población idénticos, score y protocolo raw-GT de ocho clases.
- Cribado con mejora de al menos +0,005 frente al incumbente y al control pertinente.
  Los controles y el umbral deben quedar en el plan; no copiarlos a otra población.
- Confirmación completa de todo candidato que supere el cribado.
- Revisión explícita de caídas de AP por clase mayores de 0,01, GT y TP/FP/recall.
- Incertidumbre por secuencia y confirmación independiente; tres semillas para
  resultados principales cuando una primera receta justifique ese coste.
- Checkpoint elegido por validación, inferencia sin etiquetas, guardado/carga,
  paridad del evaluador y paquete reproducible antes de promoverlo.

El panel actual no contiene GT de other_vehicle. Puede rechazar un candidato
globalmente débil, pero no certificar esa clase. Si se cambia el panel para darle
cobertura, medir de nuevo los controles en él y conservar los resultados anteriores.
No retocar el test ni el perfil después de ver el score oficial.

## 7. Recursos y límites

La ronda se ejecutó localmente en una RTX 5070 Ti Laptop de 12 GiB, con 32 GiB RAM.
El entorno probado usa Python 3.11.15 y PyTorch 2.11.0+cu128.
Verificar el hardware y el entorno actual antes de reutilizar esos datos.
Los presupuestos de 2.048 updates/7.200 s pertenecen al screening cerrado; no son
un saldo abierto para otro experimento.

No existe autorización de gasto cloud nueva en este cierre. No deducir presupuesto
de pods, IPs o acuerdos antiguos. Si el usuario pide cloud, solicitar el límite que
falte una sola vez y preparar antes el experimento concreto.

Medir preparación, forward, matching, backward, optimizador, evaluación y checkpoint.
Usar caché/prefetch solo con pruebas de paridad y consumo de RAM/VRAM.
Evitar procesos GPU competidores. Una observación que vence no acredita que el
entrenamiento haya muerto; consultar el mismo proceso antes de reiniciarlo.

## 8. Artefactos imprescindibles y recuperación

El paquete listo para usar está en Downloads/PHASE_TTC_ENTREGA_20261102.
El candidato conservado y el MonoCoP final son modelos distintos.
Los hashes y rutas exactas están en el documento de estado y en DELIVERY.json.
Los backups en C: conservan checkpoints intermedios/finales y diagnósticos; D:
conserva los runs originales. Git contiene la evidencia y el código, no los pesos
ni los datasets.

Mantener el directorio monocop_recovery_predecessor_v1 y su RECOVERY.json.
Las losses repetidas tras la recuperación no fueron bit a bit iguales; no afirmar
reanudación numérica exacta. El contrato, el estado del optimizador/sampler/RNG y
el orden de tokens sí se conservaron y verificaron.
No sobrescribir el STATUS dañado ni eliminar los fallos técnicos para simplificar
la presentación.

## 9. Reproducción y publicación

Desde la raíz del worktree:

~~~powershell
python scripts/build_eap_research_handoff_20261102.py
~~~

Ese comando usa evidencia versionada y no inicializa CUDA. Escribe ambos Markdown
y HANDOFF_EAP_20261102.sources.json. Los valores de tablas proceden de los JSON.
Copiar ambos Markdown a Descargas para una sesión futura.

La rama de publicación es dev/eap-diagnostics-20261101.
Verificar que el commit remoto coincide con el local después del push.
No inferir un merge a main, una release pública de pesos o una nueva presentación
al benchmark a partir de ese push.

## 10. Estado de las afirmaciones

Podemos defender el paquete, las cifras medidas, el diagnóstico de sesgo del prior,
la mejora de la inversión frente al prior aislado y los resultados negativos de
las variantes probadas. No podemos defender excelencia, SOTA, seguridad para
conducción, generalización independiente ni una mejora TTC/JEPA.

Si una próxima campaña no alcanza su gate, conservar el candidato anterior y
documentar qué se aprendió. Terminar una cola de GPU no prueba éxito científico.
Si el usuario pide cerrar, finalizar el trabajo pendiente y entregar resultados
con sus límites; no seguir entrenando para poder escribir una conclusión positiva.

## 11. Mensaje completo para una sesión futura

Usar este mensaje únicamente cuando el usuario decida reanudar:

~~~text
Continúa la investigación del detector PHASE-TTC para eAP 3D Detection a partir
del cierre de 2 de noviembre de 2026. Quiero mejoras defendibles en ocho clases.

Lee AGENTS.md y estos dos documentos completos:
docs/detection_sota_20261029/ESTADO_RESULTADOS_Y_ARTEFACTOS_EAP_20261102.md
docs/detection_sota_20261029/PLAN_INVESTIGACION_Y_PROMPT_CODEX_EAP_20261102.md

Repositorio: Kripta-Studios/phase-ttc.
Rama de continuidad: dev/eap-diagnostics-20261101.
El cierre de implementación llegó a 543fa399; hay un commit posterior que publica
estos documentos. Verifica HEAD, rama remota y recibos. El worktree utilizado fue:
C:/Users/Álvaro Schwiedop/Desktop/KriptaStudios/EVOCON_JEPA_Codex_Handoff/phase-ttc-eap-diagnostics
Conserva cambios ajenos y usa un worktree aislado si la rama actual es otra.

El modelo conservado es Dfull RGB + ranker TRAIN, no MonoCoP. Sus métricas exactas
están en los JSON y en el documento de estado. No se logró un modelo excelente,
no hay resultados cercanos a 1,0 y no hay etiquetas TRAIN a 400 m.
No confundas métricas TRAIN, panel reducido, selección completa y test oficial.

La ronda anterior terminó: 150 m, 300 m, rankers, combinaciones, KITTI y MonoCoP
no sustituyeron al incumbente. La inversión por vértices funciona con targets
perfectos, pero su sustitución directa con predicciones empeora TRAIN respecto
a la salida aprendida. No repitas esos screenings ni apliques la corrección
como si hubiera mejorado un modelo.

Verifica procesos e integridad antes de actuar. El run final MonoCoP es
D:/PHASE_TTC_COMPETITOR_20261101/monocop_screen_resume_v1.
El STATUS del run original quedó dañado tras un reinicio y se preservó.
Los checkpoints y diagnósticos tienen backups verificados en Descargas.
No interpretes un lock antiguo o un timeout como prueba de actividad o de muerte.

Reutiliza predicciones y sidecars ya guardados. Elige una hipótesis nueva según
los errores medidos: geometría aprendida con residual compatible, diversidad
de clases raras, resolución o contexto. Cambia una causa identificada y fija
control, población, receta, presupuesto y criterio de promoción antes de GPU.
Los screenings negativos no descartan toda temporalidad, eventos o recetas largas.

Mantén ocho clases, GT raw y splits por secuencia. No uses etiquetas privadas,
el score test para ajustar hiperparámetros ni GT en el forward de inferencia.
No modifiques fuentes congeladas ni relajes hashes. Un productor nuevo requiere
contrato nuevo y paridad con el evaluador alineado.

No hay gasto cloud autorizado por los acuerdos antiguos. Comprueba el límite
vigente si necesitas un servicio de pago. Mide el tiempo completo y los recursos.
No añadas semillas ni épocas sin una razón y un presupuesto concretos.

Trabaja solo por defecto y conserva pruebas negativas, varios checkpoints,
recibos, configuración, semillas y tablas regenerables desde JSON.
Avanza hasta el siguiente resultado defendible dentro de los recursos autorizados,
con validación y entrega reproducible. No prometas SOTA, producción ni mAP 1,0.
Si el usuario vuelve a pedir cerrar, entrega el mejor candidato verificado y
explica lo que sigue sin resolver.
~~~
