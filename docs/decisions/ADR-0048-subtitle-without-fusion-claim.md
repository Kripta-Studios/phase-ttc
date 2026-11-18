# ADR-0048: Subtítulo sin afirmación de fusión

Fecha: 2026-11-19. Estado: aceptada. Complementa a [ADR-0042](ADR-0042-phase-ttc-name.md).

## Contexto

El subtítulo público era «Causal Event–RGB Fusion for (Signed) Time-to-Contact
Estimation». Los controles emparejados del artículo no muestran beneficio medible
del acceso a eventos ni de la parametrización por fase, las cabezas reajustadas no
se distinguen de la rama RGB, y los controles del 18 de noviembre sitúan la
precisión disponible en la historia de las regiones suministradas
(`docs/sota_20261118/`). Un título centrado en «Fusion» promete una contribución
que la evidencia propia no respalda.

## Decisión

El subtítulo pasa a ser **«A Causal Event–RGB System for (Signed) Time-to-Contact
Estimation»**. Describe lo que el sistema es, que recibe eventos y RGB y es causal,
sin afirmar que la fusión sea la fuente de su rendimiento. El nombre PHASE-TTC, el
título «From Motion to Time», el espacio de nombres `e_jepa_ttc`, los
identificadores históricos y las rutas no cambian.

## Alcance

Se actualizan `AGENTS.md`, `README.md`, `CITATION.cff` y el manuscrito vigente
(`paper/from_motion_to_time_20261102`). No se modifican ADR-0042, los documentos
históricos ni la instantánea del 28 de octubre en `paper/`, que conservan el
subtítulo con el que se publicaron.

## Pendiente fuera del repositorio

La descripción del repositorio en GitHub, la página del proyecto y la ficha de
Hugging Face conservan el subtítulo anterior hasta que el propietario los cambie.
