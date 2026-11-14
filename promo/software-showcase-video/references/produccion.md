# Producción con resultados del nuevo software

## Inspección y guion

Revisa instrucciones locales, entrada del programa, funciones centrales, salidas,
ejemplos y dependencias antes de elegir capítulos. Distingue implementación,
prototipo, opción de UI sin conexión funcional y objetivo de roadmap.

Guarda un inventario pequeño: función, entrada, llamada real, archivo o dato que
la prueba, limitación y propuesta visual. Selecciona un recorrido entrada→proceso
→resultado adecuado al público. No trasplantes los capítulos de un programa a
otro cuando sus funciones sean distintas.

Registra por escena la función y llamada ejecutadas, hashes de entradas, rutas
de las salidas, cifra y unidad, tipo de dato (resultado, configuración o medición
guardada) y limitaciones. No uses números de una plantilla como cifras del vídeo.

## Integración sin alterar el proyecto

Prefiere importaciones del núcleo sin efectos secundarios, CLI o API oficiales.
No lances automáticamente servidores que conecten hardware, modifiquen bases de
datos o borren trabajos. Un adaptador que registre comandos permite ejecutar la
lógica de orquestación cuando no hace falta un dispositivo real. Etiquétalo como
simulación del hardware y no como impresión o fabricación completada.

En Python, la extracción AST de funciones concretas es una alternativa cuando
importar el módulo inicializa servicios. Conserva las definiciones originales,
inyecta dependencias explícitas, registra qué se extrajo y revisa sus dependencias.
No copies y reescribas el algoritmo para fabricar una salida más vistosa.

Guarda los datos del anuncio en una carpeta propia, por ejemplo `comercial/<producto>/`.
Reutiliza la misma preparación entre vista previa y final, comprobando los hashes
de código y entradas. No reemplaces el historial de la aplicación para generar
una demostración ni hagas un despliegue como efecto colateral del anuncio.

Si falta un ejecutable local, busca su instalación conocida o usa resultados
auténticos existentes. No sustituyas silenciosamente un laminador, un solver o
una medición por un dato falso. Documenta las versiones utilizadas; si una
alternativa cambia el significado del resultado, resuelve esa diferencia antes.

## Motores gráficos

- Pillow + NumPy: composición tipográfica y máscaras; bajo coste por fotograma.
- Matplotlib con backend Agg: gráficos y diagramas derivados de datos reales.
- VTK o PyVista: geometría, cortes, superficies reconstruidas y cámaras.
- Captura de la aplicación: cuando la interfaz misma sea el objeto del anuncio.
- FFmpeg: codificación, recorte y validación de vídeo.

Elige las dependencias disponibles que reproduzcan fielmente el resultado. Si
PyVista falla por incompatibilidad con VTK, se puede usar VTK directamente; no
actualices todo el entorno del usuario por defecto. Las imágenes y la geometría
del programa no necesitan generación de imágenes por IA para ser comerciales.

## Recursos incluidos

`scripts/editorial_frame.py` compone el marco editorial. Ejemplo de uso desde un
script del nuevo proyecto, resolviendo la ruta real del skill:

```python
import importlib.util
from pathlib import Path

helper = Path(skill_directory) / "scripts/editorial_frame.py"
spec = importlib.util.spec_from_file_location("editorial", helper)
editorial = importlib.util.module_from_spec(spec)
spec.loader.exec_module(editorial)
frame = editorial.Composer()

# actual_visual debe salir del nuevo software; no es una maqueta.
image = frame.chapter(
    visual=actual_visual,
    index=1, count=6, label="Preparation",
    title="Your real benefit.\nYour actual result.",
    body=["Describe the demonstrated behavior."],
    product="Product name",
    metric=None,
)
```

El helper admite `style_file` y `fonts_dir` para otra identidad. Los estilos
personalizados deben definir las mismas claves del JSON de referencia. Los
recursos del skill son de lectura; escribe resultados en el nuevo proyecto.

Concatena al final `assets/aptum/aptum_logo_animation.mp4` completo. Conserva su
encuadre y proporciones, y ajusta los fps o el formato de píxel al MP4 de destino.
Comprueba que el último fotograma decodificado pertenece al cierre suministrado.

Conserva un script de producción junto al MP4. Si importa helpers del skill,
incluye una copia de esos helpers y recursos en la entrega para que pueda
regenerarse fuera de Codex; alternativamente integra sus funciones en un único
script. Documenta los comandos de preparación, preview y final.

## Codificación y revisión

Usa de partida MP4 / H.264, `yuv420p`, 1920×1080, 24 fps, calidad aproximada
CRF 17–18 y `+faststart`. Un logo independiente puede usar 30 fps. El audio es
opcional: no inventes una pista o una licencia; si se solicita voz o música,
incluye sus recursos y procedencia.

1. Prepara los datos reales una vez y guarda evidencia.
2. Renderiza a 3–6 fps y saca contactos de todos los capítulos.
3. Para las escenas de cambio, revisa al menos inicio, medio y final del visual.
4. Corrige tipografía, márgenes, solapes, encuadres, cifras y ritmo.
5. Renderiza la versión final en segundo plano, con logs y archivo de salida nuevo.
6. Comprueba los datos del contenedor, decodifica el archivo y revisa muestras del
   resultado codificado, incluyendo el primer fotograma y el cierre.

Puedes ejecutar:

```text
python <skill>/scripts/verify_video.py <video.mp4> --out <carpeta-qa> --size 1920x1080 --fps 24 --duration-min 30 --duration-max 60 --chapters <chapters.json>
```

El JSON opcional es una lista de objetos con `label`, `start` y `end` en segundos
del vídeo entregado. Tras recortar una introducción, actualiza esos tiempos.

La herramienta extrae portada, contactos por capítulo y muestras de movimiento,
además de comprobar codec, duración, dimensiones y decodificación. Detecta un
primer frame casi plano como señal de revisión, no como prueba de diseño correcto.
La comprobación de contenido real, texto cortado y contraste sigue siendo visual.

Conserva las versiones aprobadas. Si solo se recortan fotogramas iniciales, usa
un filtro preciso de frames y una nueva salida; verifica que empieza en la imagen
correcta y que el cierre permanece íntegro. No elimines automáticamente un número
fijo de fotogramas: el requisito es que la portada ya sea visible.

## Entrega

Entrega MP4, portada PNG, fuente reproducible, contactos y un documento conciso
con capítulos, datos utilizados y limitaciones. Registra hashes del código,
fuentes, logos y entradas relevantes. Comunica qué falta realmente, por ejemplo
locución o validación física, sin añadir hipótesis legales ajenas al encargo.
Una duda real sobre una afirmación o un recurso se resuelve antes de publicarlo.
