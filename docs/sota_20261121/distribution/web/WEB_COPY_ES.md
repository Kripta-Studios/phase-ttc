# Resultado y estado de reproducción de PHASE-TTC

PHASE-TTC-Geo obtuvo **23,47238 de Overall MiD** en GarlTTC Test el 18 de
noviembre de 2026, con **0% de fallos según el evaluador** en 6.762 tokens. Sus
cuatro bandas fueron 24,37 / 23,95 / 22,89 / 18,13. El sistema enviado usó
estimaciones de la rama RGB y hasta 600 ms de historia causal de las regiones de
objeto suministradas; no recibió eventos. El sistema original de nueve
componentes obtuvo 35,80145 y la referencia Garl-TTC, 45,03 en sus protocolos
registrados.

El paquete PHASE-TTC-Geo recuperado de CodaBench es idéntico byte a byte al ZIP
prerregistrado (SHA-256 `8590f115…`). La reproducción desde endpoints PHASE20
congelados y autenticados recupera las 5.805 predicciones físicas, los 957 alias y
el ZIP sin deriva TTC. El reajuste de la cabeza desde esos mismos inputs reproduce
todas las predicciones físicas de la semilla 7. Las semillas 13 y 21 tienen colas
inestables; el resultado de seed 7 no demuestra insensibilidad a la semilla.

La reproducción acotada desde medios cubre 12/5.805 consultas físicas, una por
secuencia. Regenera RGB desde los medios conservados y reutiliza el tensor público
congelado de eventos. El gate de endpoint `1e-6` pasa 0/12 y el gate de TTC
geométrico `1e-4 s` pasa 5/12 (deriva mediana/máxima
`1,893e-4/8,430e-4 s`). No se leyeron targets privados ni se dedujo un Overall
privado nuevo. No es paridad extremo a extremo.

El resultado mide un sistema completo del benchmark con regiones suministradas.
No demuestra ventaja del acceso a eventos, paridad sensor-a-salida, robustez con
detector, generalización a una cohorte independiente ni cumplimiento de 20 ms.

El historial de la cuenta recoge un tercer envío terminado el 21 de noviembre de
2026: el envío `1063261`, una cabeza solo-cajas que usa 600 ms de historia de las
regiones suministradas y ningún modelo de imagen o eventos, obtuvo **27,97563 de
Overall MiD** con cero fallos según el evaluador. Sus bandas crucial / pequeña /
grande / negativa fueron 26,38540 / 28,00007 / 32,93790 / 30,89121. El envío RGB
+ geometría sigue siendo mejor con 23,47238, una **brecha de 4,50325 puntos**, y
mejora las cuatro bandas (24,36932 / 23,95223 / 22,89423 / 18,12626). Esta
comparación aísla la contribución de las estimaciones RGB más allá de la historia
de cajas suministradas que comparten; no compara empresas ni sistemas
sensor-a-salida emparejados. No se ha archivado la vista del ranking público
posterior al tercer envío, por lo que no se afirma una posición actual.
