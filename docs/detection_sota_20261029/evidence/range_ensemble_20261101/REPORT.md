# Combinación fija por distancia predicha

23.958 imágenes de selección pública ya expuesta; GT raw, ocho clases. Sin inferencia GPU nueva. Ambos controles individuales reproducen exactamente sus métricas completas. Ver PLAN.json y el plan de investigación para los criterios.
La representación compacta reconstruye exactamente las predicciones de las primeras 512 imágenes del intento v1. Los recibos ligan cada índice a los archivos de cajas originales mediante SHA-256. El intento v1 se detuvo por coste de JSON antes de producir métricas completas; se conserva su evidencia.

| Variante | mAP | Cajas totales |
|---|---:|---:|
| original_unranked | 0.05701726 | 1197900 |
| range150_unranked | 0.06105242 | 1197900 |
| gated_60m | 0.06048236 | 1832614 |
| ungated_union | 0.04055616 | 2395800 |
| Incumbente con ranking | 0.06204229 | 1197900 |

Delta primario frente al incumbente: -0.00155993; frente al mejor control individual: -0.00057006. Criterio macro completo: **False**. Promoción: **False**.

| Clase | Incumbente AP | Gated AP | Unión AP |
|---|---:|---:|---:|
| car | 0.34789333 | 0.34014157 | 0.20618380 |
| bicycle | 0.00000000 | 0.00000000 | 0.00000000 |
| truck | 0.04317640 | 0.05000593 | 0.03251438 |
| pedestrian | 0.00048178 | 0.00005804 | 0.00022873 |
| motorcycle | 0.04575330 | 0.04577926 | 0.03933050 |
| bus | 0.03563948 | 0.03251191 | 0.03194144 |
| tricycle | 0.02339401 | 0.01536219 | 0.01425044 |
| other_vehicle | 0.00000000 | 0.00000000 | 0.00000000 |

La combinación conserva el score de cada detector sin recalibrarlo. Permite hasta 100 propuestas y requiere dos forwards en despliegue. No se midió esa latencia integrada. Pueden persistir duplicados cerca de la frontera. No se ajusta la frontera tras ver estos resultados. No hay prueba de generalización independiente ni nuevo score oficial.
