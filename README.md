# Modelado y Simulación — UADE (3.1.025)

Repositorio con todos los simuladores y la app interactiva de la materia.

## App interactiva (Streamlit) — `resto1.py`
Cubre Parcial 1 y Parcial 2:
- **P1:** calculadora analítica, raíces (bisección, Newton, punto fijo, Aitken),
  interpolación (Lagrange, Newton-Gregory), integración (trapecio, Simpson 1/3 y 3/8,
  Monte Carlo 2D), derivación numérica, EDOs (Euler, RK4).
- **P2:** sistemas autónomos 1D, bifurcaciones, sistemas lineales 2D, no homogéneos 2D,
  conversión EDO orden superior → sistema, sistemas no lineales 2D, aplicaciones
  (Hamilton, Lotka-Volterra, competencia, combate Lanchester, Romeo-Julieta).

### Correr local
```
pip install -r requirements.txt
streamlit run resto1.py
```

### Deploy (trabajar desde el teléfono)
- **Streamlit Community Cloud:** repo + branch `main` + main file `resto1.py`.
- **Railway:** usa el `Procfile` incluido.

## Otros archivos
- `proyecto_cohete_spacex.html` — simulador Falcon 9 (etapas, gravity turn, sensibilidad, Monte Carlo).
- `simulador.html`, `simulador_avanzado.html` — simuladores web de métodos numéricos.
- `simulador_notebook.ipynb`, `simulador_tkinter.py` — versiones notebook y escritorio.
- `parcial2_resuelto.html` — resolución completa del 2do parcial (fórmulas + gráficos).
- `graficos/` — figuras generadas (parcial 1 y 2).
