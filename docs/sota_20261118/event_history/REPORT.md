# Historial de eventos alineado con el entrenamiento: resultado

Fecha: 2026-11-18. Protocolo: [PROTOCOL.md](PROTOCOL.md), escrito antes de calcular
cualquier resultado del brazo B. Tablas completas: [TABLES.md](TABLES.md),
regenerables con `python -m operational.event_history_match.score`.

## Conclusión

La hipótesis no se sostiene en EvTTC. Con un historial de eventos que reproduce el
de entrenamiento, la fusión publicada (F0) no mejora de forma medible, y la
estimación puntual apunta a un ligero empeoramiento. EvTTC ya se usó durante el
desarrollo: esto es un diagnóstico retrospectivo y condicional, no una confirmación
independiente. Todos los modelos están congelados; el experimento no dice qué
haría un modelo reentrenado con el historial de inferencia.

## Qué difiere realmente entre entrenamiento e inferencia

- **Entrenamiento:** ocho huecos con retardos de 350 a 0 ms cada 50 ms, todos
  codificados en el recuadro de la consulta actual. En 88.617 de 88.744 filas hay
  siete huecos válidos (el de 350 ms queda fuera del presupuesto causal por el
  desfase de exposición de 0,9 a 20 ms).
- **Inferencia en EvTTC:** cuatro huecos a 300, 200, 100 y 0 ms, también
  codificados en el recuadro de la consulta actual. En EvTTC la diferencia es la
  rejilla de retardos y los campos de tiempo, no el recuadro.
- **Inferencia en Test12:** historial a 100 ms construido con el recuadro propio
  de cada observación anterior. El recuadro por observación solo se da en Test12.

## Brazos

- **A:** predicciones entregadas. **A-replay:** A recalculado desde eventos crudos.
- **B (primario):** siete huecos cada 50 ms (300 a 0 ms), intervalo 0,05 s, 7/8 válidos.
- **B8 (sensibilidad):** la regla de entrenamiento aplicada literalmente; en EvTTC
  da 8/8 huecos, un valor que no existe en entrenamiento.

Eventos, recuadro, codificador, productores, E_CTX, cabeza y bloque RGB son
idénticos entre brazos. A-replay reproduce A con diferencia máxima de 5,7e-6 s en
F0 y 0 en E_CTX sobre las 460 consultas.

## Resultados (460 consultas, 32 grabaciones)

Bootstrap pareado por grabación completa, 10.000 remuestreos, semilla 20261118.

**Unidad de la columna MiD:** `score.py` calcula 100·|Δφ|, no los 10⁴·|Δφ| del
benchmark. Para leerla en puntos MiD oficiales hay que multiplicar por 100: el
contraste primario es +5,2 [+0,9; +9,1] MiD, un empeoramiento con intervalo que
excluye el cero. La regla prefijada se escribió sobre el MAE en segundos, por eso
devuelve «sin efecto medible»; en la coordenada del benchmark el historial
alineado es entre neutro y perjudicial.

| Lectura | MAE (s) | Mediana (s) | p95 (s) | RTE (%) | MiD |
|---|---:|---:|---:|---:|---:|
| F0, A | 0,8130 | 0,4456 | 2,7420 | 29,11 | 1,0296 |
| F0, B | 0,8712 | 0,4831 | 2,8373 | 32,02 | 1,0812 |
| F0, B8 | 0,8941 | 0,4740 | 2,9779 | 32,98 | 1,0989 |
| E_CTX, A | 1,3763 | 0,8443 | 3,7073 | 58,15 | 1,6838 |
| E_CTX, B | 1,3526 | 0,9359 | 3,5689 | 56,45 | 1,7318 |
| R_CTX (solo RGB) | 0,8513 | 0,3903 | 3,0383 | 29,02 | 0,9822 |

| Contraste | ΔMAE (s) [IC 95 %] | ΔMiD [IC 95 %] | Regla prefijada |
|---|---|---|---|
| **Primario: F0 B − A** | +0,058 [−0,004; +0,120] | +0,052 [+0,009; +0,091] | sin efecto medible |
| E_CTX B − A | −0,024 [−0,153; +0,058] | +0,048 [+0,014; +0,079] | sin efecto medible |
| F0 B − R_CTX | +0,020 [−0,055; +0,092] | +0,099 [+0,013; +0,194] | sin efecto medible |
| F0 B8 − A | +0,081 [+0,002; +0,160] | +0,069 [+0,016; +0,118] | empeora |
| E_CTX B − R_CTX | +0,501 [+0,304; +0,722] | +0,750 [+0,469; +1,074] | empeora |

F0 mejora en 9 de 32 grabaciones. Observado a posteriori, sin valor de decisión:
B alarga la TTC de F0 en el 98 % de las consultas (mediana +0,16 s) cuando A ya
sobreestimaba (error con signo de +0,10 s a +0,24 s); B mejora las seis
grabaciones CCRm y empeora la mayoría de las demás familias.

## ¿Elimina B el desajuste?

- Intervalo entre los dos últimos huecos: A da 0,10 s (ausente en entrenamiento);
  B da 0,05 s (dentro del soporte).
- Huecos válidos: A da 0,5 (ausente); B da 0,875 (dentro); B8 da 1,0 (fuera).
- **Desajuste residual en todos los brazos:** los campos de desfase consulta−ancla
  y llegada−ancla valen 0 en EvTTC frente a 0,0009–0,020 s en entrenamiento. Está
  declarado en el protocolo, no corregido, y su efecto no está medido.

## Test12 (sin etiquetas: desacuerdo, no precisión)

- Los tensores congelados de Test12 ya contienen los ocho huecos de 50 ms; el
  predictor entregado solo usaba el último de cada observación.
- Se generaron entradas de fusión del brazo B para las 5.805 consultas físicas
  (`TEST12_ARM_B_FUSION_INPUTS.npz`, fuera de git; su descripción y hash están en
  [TEST12_ARM_B_FUSION_INPUTS.json](TEST12_ARM_B_FUSION_INPUTS.json)). No se
  leyeron etiquetas privadas, no se construyó ningún paquete y no se envió nada.
- Con B, los campos de intervalo y de huecos válidos caen dentro del soporte de
  entrenamiento en todas las consultas; los de desfase quedan fuera en el 19,8 %.
  Con el historial entregado, el intervalo cae fuera en el 98,2 % y los de desfase
  en el 100 %.
- Desacuerdo de F0 con lo enviado: mediana 0,468 s, p95 2,12 s, mediana relativa
  8,9 %. B predice más largo en el 98,4 % de las consultas (mediana +0,46 s), el
  mismo sesgo que en EvTTC.

## Qué significa

1. Alinear la rejilla del historial de eventos, con todo lo demás congelado, no
   recupera un beneficio de los eventos en EvTTC. El desajuste documentado en el
   apéndice E.8 del artículo es real, pero corregirlo no basta.
2. En los dos conjuntos el brazo B alarga sistemáticamente la TTC. Como en EvTTC
   eso empeora el error, no hay base para preferir B en Test12.
3. La rama de eventos sigue siendo claramente peor que la RGB (E_CTX frente a
   R_CTX: +0,50 s de MAE con intervalo que excluye el cero).

## Pendiente

- Datos independientes; no se conocen grabaciones compatibles sin usar.
- El desfase de exposición en EvTTC (brazo declarado de antemano o reentrenamiento
  con aumentos de historial y desfase).
- Si un modelo reentrenado con el historial de inferencia aprovecharía los eventos.

## Coste

EvTTC: 1.898 s de cómputo, menos de 0,5 GB de GPU; dos ejecuciones murieron por
falta de RAM en el equipo compartido y la tercera reanudó desde fragmentos por
grabación con los mismos hashes de código y pesos. Test12: 3.812 s.
