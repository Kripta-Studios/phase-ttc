# Semántica del segundo componente de profundidad

El nombre histórico `depth_log_variance` no describe literalmente la pérdida
implementada. El productor congelado usa, para el error de profundidad `e` y
su segundo output `s`:

```text
L(e, s) = 1.4142 * exp(-s) * abs(e) + s
```

Salvo una constante aditiva, es una NLL de Laplace con escala
`b = exp(s) / 1.4142`. Su desviación estándar es
`sqrt(2) * exp(s) / 1.4142`, aproximadamente `exp(s)`. Por tanto, `s` se interpreta
como un logaritmo de escala/desviación, no como la log-varianza de una NLL
gaussiana. La expresión no demuestra que esa escala esté calibrada en datos.

Esta precisión corrige la terminología de los informes que llamaban al output
«log-varianza». Se conservan intactos el nombre del campo, las features, los
pesos, los paquetes sellados y sus resultados. Las correlaciones publicadas
siguen siendo correlaciones del mismo valor `s`; no hay una reevaluación ni
un cambio de algoritmo. El ranker aprende directamente de ese valor.

En particular, `exp(-s)` no debe describirse como inversa de una varianza
gaussiana. Tampoco se deben construir intervalos gaussianos usando `exp(s/2)`
a partir de este nombre heredado. Cualquier calibración futura necesita
correspondencias y errores fuera de los datos de ajuste.

Fuente ejecutada: `loss_depths` en el `monodgp.py` del paquete congelado.
El hash y la expresión extraída están en
`evidence/range_campaign_20261101/DEPTH_SCALE_SEMANTICS.json`.

## Geometría pendiente

El artículo de eAP describe un remapeo RGB al plano distorsionado de eventos
mediante profundidad infinita. El Parquet local conserva `K_event` y
`T_event_ego`, pero no coeficientes de distorsión. La prueba de ida/vuelta del
adaptador pinhole acredita consistencia interna, no una corrección independiente
de esa aproximación. Esta lectura confirma la limitación ya documentada;
no justifica modificar la calibración sin más evidencia.
[Fuente primaria, sección III-B](https://arxiv.org/html/2603.16303v1#S3.SS2).
