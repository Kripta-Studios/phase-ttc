# Prompt para lanzar el enjambre

Copiar el bloque siguiente tal cual en una sesión nueva abierta en la raíz del repositorio, en la rama `research/phase-ttc-qualification-20261121`.

```text
Vas a continuar el proyecto PHASE-TTC (estimación de tiempo hasta el contacto con cámaras de eventos y RGB) como orquestador de un enjambre de agentes. El objetivo es que el modelo pueda defenderse como estado del arte: hoy es primero en el benchmark GarlTTC con 23,47 Overall MiD usando las regiones que el benchmark suministra, pero falta demostrar que la ventaja es del método, que se transfiere a datos independientes, que cabe en 20 ms y que funciona sin regiones suministradas.

Todo lo que necesitas está escrito en el repositorio. Antes de lanzar ningún agente, lee tú mismo, en este orden:
1. AGENTS.md, sección 3 (integridad científica).
2. docs/sota_20261121/handoff/README.md (estado, reglas, entorno y lista de paquetes de trabajo). Es el contrato de esta sesión.
3. docs/sota_20261121/HALLAZGOS_Y_LIMITES.md y docs/sota_20261121/TRABAJO_REALIZADO.md.
4. docs/sota_20261118/REVIEW_20261119.md (qué se puede afirmar y qué no).
5. docs/sota_20261121/COMO_CONTINUAR_BENCHMARK.md y docs/sota_20261121/COMO_CONTINUAR_SIN_CAJAS.md.
6. Los ocho paquetes docs/sota_20261121/handoff/WP1 a WP8.

Después comprueba el estado real antes de fiarte de lo escrito: rama y árbol limpio, espacio libre en C: y D:, memoria libre, que la GPU esté sin procesos ajenos, y que pasan las pruebas existentes de los módulos que vas a tocar. Si algo de lo escrito ya no coincide con el repositorio, manda el repositorio y anótalo.

Cómo trabajar:
- Lanza primero, en paralelo, los paquetes que no dependen de nada ni compiten por la GPU: WP3 (mejoras de la cabeza), WP4 fase A (investigación de cohortes, sin descargas grandes ni altas), WP6 (tolerancia a ruido de caja) y la preparación de WP1 (integraciones y medición de presupuesto, sin entrenar todavía).
- Cuando terminen, sigue el orden de dependencias del README: WP7 tras WP6; WP8 tras WP7; WP2 tras WP1 y WP3. WP5 y cualquier entrenamiento en GPU nunca a la vez.
- A cada agente dale un encargo autocontenido: el objetivo, la ruta de su fichero WP y de los informes que debe leer, lo que ya está descartado, sus límites de memoria y disco, y qué debe devolverte. No le des tus conclusiones para que las confirme.
- Modelos: Opus 5.5 con esfuerzo alto para implementaciones complejas (WP1, WP2, WP4, WP7, WP8), Opus 5.5 con esfuerzo medio para las sencillas (WP3, WP5, WP6 y búsquedas), y Fable 5.2 con esfuerzo medio como revisor final independiente, de solo lectura, al que le das código y ficheros de resultados.
- Como mucho cuatro procesos pesados a la vez en total: la máquina es compartida y ya se quedó sin memoria una vez.

Reglas que no admiten excepción:
- Protocolo escrito antes de medir; particiones e intervalos por secuencia completa; resultados negativos informados igual que los positivos; ninguna cifra escrita a mano.
- Nadie lee etiquetas privadas del test. Nadie sube nada a CodaBench: solo se preparan paquete y recibo con scripts/archive_submission.py, y el envío lo decide y lo hace el propietario.
- FCWD, EvTTC y TRAIN40 son desarrollo ya expuesto; no se presentan como confirmación. No se escribe «estado del arte» sin la comparación reproducida que lo sostenga.
- Los subagentes tienen prohibido git add, git commit, git push y cualquier escritura de git. Solo tú haces commits, en la rama actual, con autor Kripta-Studios y sin ninguna línea de atribución a Claude ni a ninguna herramienta, y los subes con el remoto ya configurado. No fusiones a main.
- Sin rutas absolutas ni nombres de usuario en ficheros versionados. No se envían datos del proyecto a servicios externos. Altas, aceptación de licencias y contacto con terceros corresponden al propietario.
- No borres nada existente para ganar espacio. Si un paquete no cabe en disco o en tiempo, para ese paquete, deja escrito el presupuesto medido y sigue con los demás.

Al terminar cada paquete: verifica pytest y ruff, busca rutas privadas en lo nuevo, haz el commit y súbelo. Al terminar la tanda: lanza la revisión final, registra su dictamen en docs/sota_20261121/REVIEW_<fecha>.md, corrige lo que señale, y actualiza HALLAZGOS_Y_LIMITES.md y los dos planes de continuación con lo que haya cambiado.

No te detengas a pedir confirmación entre paquetes. Solo necesitas al propietario para: enviar algo al test, darte de alta o aceptar términos en un servicio, liberar disco o autorizar una GPU externa. En esos casos sigue con todo lo demás y deja la petición clara en tu informe final, que debe decir, por paquete, el veredicto frente a su regla previa, las cifras principales con sus intervalos, las desviaciones y lo que queda pendiente.
```

## Variantes

- **Solo el frente del benchmark:** añadir al final «Limítate a WP1, WP2, WP3 y WP5».
- **Solo el frente sin regiones:** añadir «Limítate a WP6, WP7 y WP8».
- **Solo investigación, sin gastar GPU:** añadir «Limítate a WP3, WP4 fase A, WP6 y la preparación de WP1; no entrenes productores ni detectores».
