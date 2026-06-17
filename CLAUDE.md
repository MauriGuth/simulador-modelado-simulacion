# Preferencias del proyecto

## Resolución de ejercicios (preferencia permanente del usuario)

Cuando el usuario envíe una **foto con ejercicios** (consignas de matemática / modelado
y simulación, sistemas dinámicos, etc.), **siempre** entregar la resolución así:

- **Resolución completa y paso a paso** de todos los ejercicios, con todos los cálculos
  intermedios, lista para **transcribir a papel**.
- Entregarla como **PDF compilado** (matemática renderizada, **no** LaTeX crudo ni
  Markdown con `$...$`): el usuario no puede pasar a papel el código LaTeX.
- Si algún ejercicio requiere **gráfico** (retrato de fase, nulclinas, campo vectorial,
  trayectorias, etc.), **generarlo** y **embeberlo** en el mismo PDF.
- Enviar el PDF al usuario con `SendUserFile`.

### Cómo se genera el PDF en este entorno (reproducible)

El contenedor es efímero: en cada sesión nueva probablemente haya que reinstalar las
herramientas. Receta usada:

1. Gráficos: `pip install numpy matplotlib` y un script de Python (ver
   `graficos_examen/plots.py`) que dibuja los retratos de fase con un integrador RK4
   propio (sin dependencias extra).
2. PDF: `apt-get install -y --no-install-recommends texlive-latex-base
   texlive-latex-recommended texlive-fonts-recommended poppler-utils`, luego documento
   LaTeX con `amsmath`, `amssymb`, `graphicx` y compilar con `pdflatex` (dos pasadas).
   - No usar `babel` con opción `spanish` (no está `texlive-lang-spanish`); los acentos
     funcionan igual con `inputenc utf8` + `fontenc T1`.
   - No usar `enumitem` (no está `texlive-latex-extra`); usar `itemize` estándar.
3. Verificar visualmente renderizando las páginas con `pdftoppm -png` y revisar que no
   haya `Overfull \hbox` (texto que se sale del margen).

Ejemplo de referencia ya resuelto: carpeta `graficos_examen/` (`resolucion.tex`,
`resolucion.pdf` y los 4 PNG de los retratos de fase).
