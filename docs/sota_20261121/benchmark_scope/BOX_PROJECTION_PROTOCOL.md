# Procedencia geométrica de las cajas de TRAIN40

Comprobar si se pueden reconstruir las cajas 2D públicas a partir de las esquinas 3D en cámara y los intrínsecos que distribuye eAP. La motivación viene de la figura 4 y la sección III-C del artículo de los organizadores. Antes de congelar este protocolo se inspeccionó el esquema del Parquet y el primer registro para identificar unidades y orientación. El ensayo es un diagnóstico de procedencia en datos de desarrollo.

Usar todas las consultas de `GarlTTC/data/train.parquet`. Emparejar por `sample_token` con `GarlTTC/annotations/train.parquet`, leyendo de este último únicamente el token y las esquinas `box3d_Fcam`. Puntuar el último fotograma de cada consulta. Obtener `K_event` de la fila de medios eAP identificada por secuencia, shard y miembro. Usar el plano de 1280 × 720 del release. Proyectar las ocho esquinas, tomar sus extremos y limitar a [0,1279] × [0,719]. No estimar intrínsecos, correcciones ni redondeos a partir de las cajas observadas.

Registrar conteos totales, sin calibración y con proyección inválida (valores no finitos, esquina detrás del plano de cámara o área degenerada). Sobre filas válidas, calcular IoU continua en coordenadas xyxy, máximo error absoluto entre los cuatro bordes en píxeles, fracción a ≤1 y ≤2 píxeles y coincidencia exacta tras truncar las coordenadas proyectadas. Informar resultados agregados y por secuencia. Registrar hashes de los tres parquets y del script. No abrir columnas TTC ni usar datos de Test12.

Un alto acuerdo permitiría afirmar equivalencia geométrica bajo la proyección probada en TRAIN40. No demuestra qué herramienta produjo cada anotación, cómo se generaron las cajas de Test12 ni qué precisión alcanzaría un detector.
