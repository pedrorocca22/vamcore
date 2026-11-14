# Estilo APTUM Editorial

Este documento define el aspecto aprobado y se puede leer por separado del skill.
Es un sistema visual reutilizable; no fija las funciones de ningún software.

## Identidad

| Uso | Valor |
|---|---|
| Fondo | Cloud `#EEF2F3`, o blanco cuando la marca lo requiera |
| Texto principal | Aptum Navy `#102633` |
| Acento principal | Deep Blue `#225B7A` |
| Acento de apoyo | Aptum Teal `#63B3AF` |
| Apoyo suave | Soft Mint `#A9D8D1` |
| Blanco | `#FFFFFF` |
| Texto secundario | `#3C535E`; etiquetas discretas `#526973` |
| Reglas y separadores | `#D2DFE1` |

Usa el azul profundo para cifras, selección y progreso. Teal y mint sirven para
series secundarias y estados pendientes. El degradado azul→teal se utiliza con
moderación, vinculado al lenguaje del logo. El ámbar puede identificar una
simulación o un aviso real; no añadas avisos ficticios para decorar.

Conserva la semántica de los datos. Un mapa científico, una escala de grises o un
resultado con colores propios no se recolorea si ello altera su lectura. Los
renderizados de geometría y las máscaras ilustrativas pueden usar el acento de
marca, identificando ese mapeo de presentación cuando sea relevante.

## Tipografía

| Elemento | Familia | Peso orientativo | Tamaño en 1080p |
|---|---|---|---|
| Titulares | Newsreader | 500 | 54–60 px; 58 px de partida |
| Frase de portada | Newsreader | 400–500 | 44–48 px |
| Texto general | Libre Franklin | 300–600 | 26–28 px |
| Etiquetas y secciones | IBM Plex Mono | 400–500 | 18–22 px |
| Cifra principal | IBM Plex Mono | 400–500 | 86–100 px |
| Pie y notas | Libre Franklin o IBM Plex Mono | 400 | 18–22 px |

Las fuentes y licencias abiertas están en `assets/aptum/fonts/`. Newsreader
variable tiene ejes de peso y tamaño óptico: configura sus ejes por nombre,
no asumas un orden. Para titulares grandes usa tamaño óptico 72. No sustituyas
las fuentes silenciosamente; si falta un archivo, resuelve el recurso antes de
producir la versión final o comunica la sustitución.

Un logotipo suministrado conserva su dibujo. La elección tipográfica anterior
se aplica a textos, no autoriza redibujar un logo de otra organización.

## Composición de referencia: 1920×1080

- Visual: `x=0..1190`; divisor fino en `x=1190`.
- Columna de texto: inicio `x=1258`; margen derecho al menos 64–76 px.
- Etiqueta del capítulo: alrededor de `y=202`.
- Titular: alrededor de `y=266`, interlínea 70 px, máximo dos líneas.
- Apoyo: alrededor de `y=455`, interlínea 39 px, máximo tres líneas breves.
- Cifra: alrededor de `y=663`; explicación alrededor de `y=788`.
- Progreso por capítulos: alrededor de `y=936`; marca discreta en el pie.
- Etiqueta de procedencia del visual: `x≈76, y≈92`, cuando sea necesaria.

Son coordenadas de partida, no una obligación de encajar contenido cortado.
Acorta el texto o reduce ligeramente el tamaño manteniendo el sistema. Comprueba
la caja real de cada fuente: dos familias del mismo tamaño no ocupan lo mismo.
Evita escalas independientes por eje y conserva la relación de aspecto del visual.

## Portada y cierre

La primera imagen ya debe mostrar el nombre del producto y una propuesta de
valor legible. Puede ser una portada editorial o una pantalla real si se pide.
Exporta esa misma imagen como PNG. No dependas de metadatos de póster: algunas
plataformas construyen la miniatura con el fotograma cero del MP4.

Usa el logotipo original del producto en la portada y conserva sus proporciones.
Termina cada vídeo con el MP4 completo `assets/aptum/aptum_logo_animation.mp4`.
Conserva el encuadre, los colores y el lema. Adapta el formato de codificación
al vídeo final cuando sea necesario. El contenido visible y audible va en inglés.

## Movimiento

Escenas de unos 4–7 s y fundidos de unos 0,3–0,45 s funcionan como punto de
partida. Usa aceleración suave y pequeños desplazamientos. Evita rebotes,
destellos, efectos de partículas y giros de cámara sin relación con la función.

| Función que se explica | Cambio que debe verse |
|---|---|
| Biblioteca de patrones | El patrón principal cambia; selección y nombre lo siguen |
| Parámetros por altura | La selección recorre la altura y el corte muestra el valor actual |
| Escala de grises | El corte cambia de tono, sincronizado con curva y cifra |
| Análisis o diagnóstico | Se revelan datos, regiones o problemas realmente detectados |
| Diseño paramétrico | La geometría se recalcula y se ve su transformación real |
| Exportación | Se muestran archivos efectivamente escritos y su resultado |

Los fundidos entre dos resultados reales son transiciones editoriales. No
presentes estados interpolados como nuevas simulaciones o diseños calculados.
Una cámara que gira no sustituye el cambio de patrón, parámetro o estado descrito.

## Copia comercial y rigor

El titular expresa el beneficio demostrado; el apoyo explica el comportamiento.
Habla al público del producto, sin rutas internas ni marcas de dependencias
innecesarias. La documentación técnica sí conserva su procedencia.

Utiliza como máximo una cifra principal por capítulo. Si una cifra no ayuda o
no está disponible, omítela. Las configuraciones programadas y los valores de
una calibración guardada se etiquetan como tales. No conviertas una imagen bonita
en una afirmación de rendimiento, eficacia clínica, certificación o resultado físico.
