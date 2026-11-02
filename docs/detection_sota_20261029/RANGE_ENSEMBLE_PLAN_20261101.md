# Ensayo CPU de combinación por distancia

Registro posterior a la confirmación completa de 150 m y anterior a medir esta
combinación. Es exploración sobre selección pública ya expuesta, no confirmación
independiente. La confirmación mostró mayor cobertura lejana y menor cobertura
entre 40 y 60 m; el candidato no superó al incumbente. Ver los JSON y el informe
en `evidence/range150_confirmation_20261101/`.

Hipótesis: usar el detector original para objetos próximos y el de 150 m para
objetos lejanos puede conservar las ventajas complementarias observadas.

## Receta fijada

- Dos checkpoints congelados: Dfull original y range150/update1024.
- Candidato primario: conservar las cajas originales con profundidad óptica
  **predicha** <= 60 m y las cajas del segundo modelo con profundidad predicha
  > 60 m. Usar la calibración disponible en inferencia. No usar profundidad GT.
- La frontera corresponde al límite de entrenamiento original; no barrerla.
- Scores sigmoid de clase originales en ambos modelos, sin ranker, calibración
  aprendida, NMS, cambio de clase, mezcla de coordenadas ni filtrado de targets.
- Orden de cajas: original primero, segundo modelo después; preservar orden
  interno. Se conserva así una regla explícita para empates de score.
- Control adicional: unión sin filtro de las mismas dos listas. Permite observar
  el efecto de aumentar las hipótesis. El máximo es 100 cajas, dentro del límite
  de 500 del perfil; el presupuesto de propuestas no es el de un solo modelo.
- Recalcular los controles de un solo detector y exigir paridad exacta con sus
  métricas completas guardadas antes de aceptar el informe.

Población: las mismas 23.958 imágenes, ocho secuencias de selección pública, GT
raw y ocho clases. Sin etiquetas privadas, cambios de split ni acceso a test.
No hay entrenamiento, sampling aleatorio ni semilla de optimización.

Métrica primaria: mAP alineada agrupada. El candidato primario debe superar en
al menos 0,005 tanto al incumbente con ranking como al mejor control individual,
y mejorar respecto a la unión sin filtro para atribuir ventaja al filtro.
Revisar caídas por clase > 0,01 AP. No promocionar automáticamente el control
unión si resulta mejor; informar todas las variantes, incluidas las negativas.

Presupuesto: una pasada CPU de las predicciones guardadas, hasta 900 segundos
con comprobación del límite entre shards. Sin inferencia GPU nueva ni coste
cloud. Guardar recibos, entradas verificadas, fuente ejecutada, predicciones
combinadas comprimidas y métricas globales/por secuencia. Parar ante integridad
inválida, valores no finitos, población distinta o divergencia de los controles.

Una mejora exigiría verificar inferencia integrada y coste antes de promover.
Dos detectores implican dos forwards; no se presupone latencia igual al modelo
actual. La distancia predicha puede estar sesgada, los scores pueden diferir
entre modelos y puede haber duplicados alrededor de la frontera. Este ensayo no
resuelve por sí solo la debilidad en clases raras ni demuestra excelencia.
