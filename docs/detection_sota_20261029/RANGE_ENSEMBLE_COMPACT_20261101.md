# Revisión técnica de almacenamiento del ensayo de combinación

El intento v1, registrado en `8cd65b9a`, se detuvo de forma controlada entre shards
antes de generar métricas completas. Su estado final, predicciones parciales,
plan, fuente y perfil quedan en `D:/PHASE_TTC_RANGE_20261101/range_ensemble_v1`.
No es un resultado científico negativo de la combinación.

Los primeros bloques tardaban unos treinta segundos cada uno. Un muestreo pasivo
de diez segundos con py-spy (990 muestras, cero errores) encontró 291 muestras en
JSON raw_decode, 250 en iterencode y 107 en gzip.compress. Son muestras de pilas
de Python, no una descomposición exacta del tiempo de pared. El ritmo observado
indicaba que el presupuesto inicial de 900 segundos sería insuficiente.

La versión compacta conserva la receta científica del plan original, incluidos
la frontera de 60 m, los dos modelos, controles, scores, población y criterios.
Cambia la representación de los artefactos: guarda por token la lista ordenada
de pares (modelo, índice de propuesta), comprimida, con hashes de los archivos
originales. Esto permite reconstruir las cajas completas sin volver a escribir
sus numerosos floats. La unión sin filtro se reconstruye concatenando las dos
listas originales verificadas. No se pierde score, geometría, clase u orden.

Antes de aceptar resultados se exige igualdad de predicciones y reconstrucción
desde el archivo compacto en las primeras 512 imágenes guardadas por v1, además
de la paridad completa de ambos controles individuales. La función de selección
de cajas no se modifica. Los resultados de v1 no se reinterpretan como completos.

Se registra antes de medir la combinación un presupuesto técnico de hasta 1.800
segundos CPU para v2, con límite comprobado entre bloques, para cubrir el coste
observado de parsing y puntuación. Esto amplía el margen temporal de ejecución;
no amplía modelos, fronteras, variantes, entrenamiento ni inferencia GPU. No hay
coste cloud. Se conservan todos los intentos y el tiempo de cada uno.
