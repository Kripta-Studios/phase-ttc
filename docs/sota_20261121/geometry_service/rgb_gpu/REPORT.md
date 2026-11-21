# Servicio largo RGB + geometría suministrada

Replay host-paced de 60 s por celda, 50 Hz, con uno o cuatro objetos físicos de pistas
distintas en el mismo instante, rama persistente RGB_GEO_BAND y cabeza congelada. Los frames
RGB se decodifican y el historial antiguo se cachea antes del reloj; cada oferta ejecuta el
crop más reciente, productores GPU, contexto RGB, geometría de cajas y cabeza. El reloj
empieza en el deadline absoluto, antes de adquirir el puntero a la oferta precargada y de la
cola, y termina tras obtener el TTC como float host. La cola y la sincronización GPU quedan
incluidas. No es una lectura sensor-a-salida ni una certificación de sensor físico.

| celda | obj. | ofertas | válidas | drops | <20 ms | p50 | p95 | p99 | máx | fría | paridad |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| rgb_geo__supplied_050ms__objects_01__queue_64 | 1 | 3000 | 408 | 2592 | 0.00% | 11204.24 | 11677.995009999997 | 12083.059838 | 12115.905 | 138.529 | sí |
| rgb_geo__supplied_000ms__objects_04__queue_02 | 4 | 3000 | 90 | 2910 | 0.00% | 2049.96325 | 2208.9738899999998 | 2252.765911 | 2261.8974 | 151.783 | sí |
| rgb_geo__supplied_000ms__objects_04__queue_64 | 4 | 3000 | 136 | 2864 | 0.00% | 50268.500499999995 | 55792.243775 | 55885.08047 | 55908.0947 | 196.255 | sí |
| rgb_geo__supplied_050ms__objects_04__queue_64 | 4 | 3000 | 144 | 2856 | 0.00% | 47862.956000000006 | 48717.528615 | 48820.17434 | 48913.9181 | 191.960 | sí |
| rgb_geo__supplied_050ms__objects_04__queue_02 | 4 | 3000 | 80 | 2920 | 0.00% | 2305.21435 | 2527.95146 | 2541.649606 | 2570.0332 | 161.176 | sí |
| rgb_geo__supplied_000ms__objects_01__queue_02 | 1 | 3000 | 313 | 2687 | 0.00% | 572.4496 | 632.47498 | 653.554256 | 673.849 | 180.042 | sí |
| rgb_geo__supplied_000ms__objects_01__queue_64 | 1 | 3000 | 394 | 2606 | 0.00% | 11727.32025 | 12018.932825 | 12050.589413 | 12060.0557 | 160.622 | sí |
| rgb_geo__supplied_050ms__objects_01__queue_02 | 1 | 3000 | 321 | 2679 | 0.00% | 563.5501 | 628.5278 | 644.91022 | 677.637 | 196.467 | sí |

La condición de 50 ms desplaza tanto la disponibilidad del pool geométrico como la de cada
observación RGB: se eliminan del contexto las observaciones cuyos recortes necesitan cajas
que aún no estarían disponibles. La paridad compara el servicio CUDA-graph con una ejecución
eager independiente, preparada fuera del reloj sobre los mismos bytes y relojes causales.

Gate de capacidad de las celdas: **FAIL**; cumplen 0/8. Gate efectivo de la
campaña: **INVALIDATED**. No se afirma capacidad garantizada en las condiciones que no
cumplen todas sus ofertas antes de 20 ms sin drops ni cuando la campaña queda invalidada.

Auditoría de pausa/suspensión: **VÁLIDA**; lateness máximo del planificador
97.184 ms frente al umbral invalidante de 1000 ms.
Validez de atribución: **INVALIDATED**; runner GPU visible=True, carga de
modelo dedicada=False.

El detector/tracker localizado pertenece a otro protocolo y no se incluye en estas ofertas;
véase `DETECTOR_AUDIT.md`. Cada capacidad de objetos cicla una población fija de 2
instantes reales durante 3.000 ofertas para medir capacidad sostenida, no independencia
estadística.
