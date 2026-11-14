---
name: software-showcase-video
description: Crea o revisa vídeos comerciales y demostraciones editoriales de software a partir de resultados reales del proyecto. Úsalo para vídeos MP4 de producto, animaciones de funciones y recorridos de software con capítulos, métricas verificables y una identidad de marca consistente. Incluye el estilo APTUM y admite otras marcas. No es un flujo para modificar la interfaz de la aplicación.
---

# Software Showcase Video

Produce un vídeo que explique lo que el software puede hacer hoy, con la limpieza
editorial y la dinámica aprobadas para los vídeos de APTUM. Cambia el contenido
según el proyecto; conserva el lenguaje visual cuando se pida el mismo estilo.

## Resolver el encargo

Revisa primero la carpeta activa, sus instrucciones, README, código y recursos
de marca. Averigua las funciones ejecutables y los ejemplos disponibles antes
de escribir el guion. El código y los resultados prevalecen sobre el marketing.

Si el encargo no especifica otros valores, parte de un comercial web de 30–60 s,
1920×1080, 24 fps y sin audio. Todo el contenido del vídeo debe estar siempre en
inglés, incluidos rótulos, cifras con texto, subtítulos y locución si se incluye.
Conserva los nombres propios y el logotipo original de la aplicación.

Consulta [el estilo editorial](references/estilo-editorial.md). Cada vídeo debe
terminar con el archivo suministrado `assets/aptum/aptum_logo_animation.mp4`,
completo y como último segmento. La marca de la aplicación identifica el producto
durante el anuncio y el clip APTUM funciona como firma de cierre. Si falta el
archivo, informa del bloqueo.

## Preparar una demostración verdadera

Lee [el flujo de producción](references/produccion.md) para preparar datos,
elegir el motor gráfico y renderizar. Sus apartados de integración son relevantes
cuando el programa tiene efectos secundarios, hardware o servicios externos.

- Ejecuta el núcleo, API o CLI con un ejemplo representativo. Conserva los
  archivos de entrada, salidas, parámetros y hashes que sostengan lo mostrado.
- Una cifra en pantalla debe proceder de esa ejecución, del archivo realmente
  generado o de mediciones guardadas claramente identificadas como tales.
  El tiempo programado no equivale a tiempo medido; los píxeles no equivalen a
  resolución física; una simulación no equivale a un resultado físico.
- Identifica en pantalla hardware simulado, prototipos y mediciones históricas.
  Los fundidos editoriales entre dos resultados no son estados calculados nuevos.
- Si algo no puede ejecutarse, no inventes su salida. Usa datos auténticos
  existentes con su procedencia, omite esa función o solicita el recurso faltante.
- No modifiques el núcleo ni actúes sobre dispositivos para producir un anuncio
  cuando sea suficiente un adaptador de grabación o una ejecución aislada.

## Construir la secuencia

Ordena normalmente 5–8 cualidades como las vive el usuario: entrada, preparación,
proceso, inspección y resultado. Ajusta el número a las funciones demostrables;
no añadas capítulos vacíos para cumplir una cifra.

Cada capítulo utiliza una etiqueta corta, un titular de hasta dos líneas,
1–3 frases breves y, cuando aporte valor, una cifra verificable. El visual ocupa
aproximadamente el 62 % izquierdo y el texto el 38 % derecho. Evita marcas de
terceros, extensiones y detalles internos en el texto comercial salvo que ayuden
al público o el usuario los pida; sí documenta esas dependencias en la procedencia.

El comportamiento descrito debe verse en el objeto principal: si hablas de
patrones, cambia el patrón grande; si hablas de intensidad, cambia el brillo del
corte; si hablas de capas, recorre las capas. Un contador animado junto a una
imagen inmóvil no demuestra una transición. Sincroniza datos, selección, gráfico
y visual con un único estado de escena.

La portada debe estar completa desde el primer fotograma y mostrar la identidad
original del producto. Antes de entregar, verifica que el vídeo termina con
el clip APTUM incluido, con su encuadre y proporciones originales.

`scripts/editorial_frame.py` ofrece tipografía, márgenes y composición reutilizables;
úsalo si ayuda a mantener el estilo. La geometría, imágenes y animaciones de cada
función deben seguir viniendo del nuevo software. No reutilices resultados de
Biolight, cifras de ejemplo o guiones clínicos para otro proyecto.

## Revisar y entregar

Haz una pasada de 3–6 fps. Genera una hoja de contactos y varias muestras de cada
escena dinámica. Mira las imágenes: corrige cortes de texto, solapes, encuadres,
contraste, estados estáticos y cifras incoherentes antes de renderizar a 24 fps.
Mantén el proceso final en segundo plano y comunica avances cuando tarde.

Usa `scripts/verify_video.py` para decodificar el resultado, comprobar formato y
extraer portada y fotogramas. Las comprobaciones numéricas son auxiliares: no
sustituyen la revisión visual ni prueban la veracidad de las funciones.

Entrega el MP4 reproducible, una portada PNG, el script o fuentes de generación,
la hoja de contactos y la procedencia de capítulos y métricas. Indica si faltan
audio o validación física. Conserva versiones previas al revisar un vídeo aprobado.
No publiques ni envíes archivos a terceros sin una instrucción que lo autorice.

Hay ejemplos de invocación y traspaso entre proyectos en
[peticiones reutilizables](references/peticiones.md).
