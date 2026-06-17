import streamlit as st
import numpy as np
import pandas as pd
import sympy as sp
import matplotlib.pyplot as plt
import math

# ==========================================
# FUNCIONES BASE, TECLADO Y PARSER MATEMÁTICO
# ==========================================
if "funcion_input" not in st.session_state:
    st.session_state.funcion_input = ""

def agregar_texto(texto):
    st.session_state.funcion_input += texto

def borrar_texto():
    st.session_state.funcion_input = ""

def parse_math(expr):
    try:
        if not expr: return None
        expr_str = str(expr).replace(',', '.')
        return float(sp.sympify(expr_str).evalf())
    except Exception:
        return None

def teclado_matematico():
    with st.expander("⌨️ Teclado Matemático (Ayuda memoria)"):
        st.caption("Hacé clic en los botones para armar tu función con la sintaxis correcta.")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.button("sin()", on_click=agregar_texto, args=("sin(",))
        c2.button("cos()", on_click=agregar_texto, args=("cos(",))
        c3.button("tan()", on_click=agregar_texto, args=("tan(",))
        c4.button("exp()", on_click=agregar_texto, args=("exp(",))
        c5.button("log()", on_click=agregar_texto, args=("log(",))
        
        c1_2, c2_2, c3_2, c4_2, c5_2 = st.columns(5)
        c1_2.button(" + ", on_click=agregar_texto, args=(" + ",))
        c2_2.button(" - ", on_click=agregar_texto, args=(" - ",))
        c3_2.button(" * ", on_click=agregar_texto, args=(" * ",))
        c4_2.button(" / ", on_click=agregar_texto, args=(" / ",))
        c5_2.button(" ** (Pot)", on_click=agregar_texto, args=("**",))
        
        c1_3, c2_3, c3_3, c4_3, c5_3 = st.columns(5)
        c1_3.button(" ( ", on_click=agregar_texto, args=("(",))
        c2_3.button(" ) ", on_click=agregar_texto, args=(")",))
        c3_3.button(" x ", on_click=agregar_texto, args=("x",))
        c4_3.button(" pi ", on_click=agregar_texto, args=("pi",))
        c5_3.button("Borrar", on_click=borrar_texto, type="primary")

# ==========================================
# MÓDULO 1: RAÍCES
# ==========================================
def biseccion(f_str, a, b, tol, max_iter):
    x = sp.Symbol('x')
    f = sp.lambdify(x, sp.sympify(f_str), 'numpy')
    if f(a) == 0: return pd.DataFrame([{"Iteración": 0, "a": a, "b": b, "Raíz (c)": a, "Error": 0.0}]), None
    if f(b) == 0: return pd.DataFrame([{"Iteración": 0, "a": a, "b": b, "Raíz (c)": b, "Error": 0.0}]), None
    if f(a) * f(b) > 0: return None, "El intervalo no cambia de signo."
        
    resultados = []
    for i in range(max_iter):
        c = (a + b) / 2
        error = abs(b - a) / 2
        resultados.append({"Iteración": i, "a": a, "b": b, "Raíz (c)": c, "Error": error})
        if f(c) == 0 or error < tol: break
        if f(a) * f(c) < 0: b = c
        else: a = c
        if i == max_iter - 1 and error >= tol:
            return pd.DataFrame(resultados), "El método alcanzó el límite de iteraciones y NO convergió a la tolerancia pedida."
    return pd.DataFrame(resultados), None

def newton_raphson(f_str, x0, tol, max_iter):
    x = sp.Symbol('x')
    f_sym = sp.sympify(f_str)
    df_sym = sp.diff(f_sym, x)
    f = sp.lambdify(x, f_sym, 'numpy')
    df = sp.lambdify(x, df_sym, 'numpy')
    resultados = []
    xi = x0
    for i in range(max_iter):
        f_xi = f(xi)
        df_xi = df(xi)
        if df_xi == 0: return pd.DataFrame(resultados), "Falla: La derivada es cero."
        x_next = xi - (f_xi / df_xi)
        error = abs(x_next - xi)
        resultados.append({"Iteración": i, "x_n": xi, "f(x_n)": f_xi, "f'(x_n)": df_xi, "x_n+1": x_next, "Error": error})
        if error < tol: break
        xi = x_next
        if i == max_iter - 1 and error >= tol:
            return pd.DataFrame(resultados), "El método alcanzó el límite de iteraciones y NO convergió (Probablemente divergió)."
    return pd.DataFrame(resultados), None

def punto_fijo(g_str, x0, tol, max_iter):
    x = sp.Symbol('x')
    try: g_expr = sp.sympify(g_str)
    except Exception as e: return pd.DataFrame(), f"Error de sintaxis: {e}"
    resultados = []
    xi = float(x0)
    for i in range(max_iter):
        try:
            x_next = float(g_expr.subs(x, xi).evalf())
            if np.isinf(x_next) or np.isnan(x_next) or abs(x_next) > 1e10: 
                return pd.DataFrame(resultados), "El método DIVERGIÓ matemáticamente (creció hacia el infinito)."
            error = abs(x_next - xi)
        except Exception as e: return pd.DataFrame(resultados), f"Error de cálculo: {e}"
        resultados.append({"Iteración": i, "x_n": xi, "x_n+1": x_next, "Error": error})
        if error < tol: break
        xi = x_next
        if i == max_iter - 1 and error >= tol:
            return pd.DataFrame(resultados), "El método alcanzó el límite de iteraciones y NO convergió. Revisá tu g(x)."
    return pd.DataFrame(resultados), None

def punto_fijo_aitken(g_str, x0, tol, max_iter):
    x = sp.Symbol('x')
    try: g_expr = sp.sympify(g_str)
    except Exception as e: return pd.DataFrame(), f"Error de sintaxis: {e}"
    resultados = []
    xi = float(x0)
    for i in range(max_iter):
        try:
            x1 = float(g_expr.subs(x, xi).evalf())
            x2 = float(g_expr.subs(x, x1).evalf())
            denominador = x2 - 2*x1 + xi
            if denominador == 0: x_acc = x2
            else: x_acc = xi - ((x1 - xi)**2) / denominador
            error = abs(x_acc - xi)
            resultados.append({"Iteración": i, "x_n": xi, "x_1 = g(x_n)": x1, "x_2 = g(x_1)": x2, "x_acelerado": x_acc, "Error": error})
            if error < tol: break
            xi = x_acc
        except Exception as e: return pd.DataFrame(resultados), f"Error de cálculo: {e}"
        if i == max_iter - 1 and error >= tol:
            return pd.DataFrame(resultados), "El método no convergió."
    return pd.DataFrame(resultados), None

# ==========================================
# MÓDULO 2: INTERPOLACIÓN
# ==========================================
def lagrange_interp(x_vals, y_vals):
    x = sp.Symbol('x')
    poly = 0
    n = len(x_vals)
    for i in range(n):
        L_i = 1
        for j in range(n):
            if i != j: L_i *= (x - x_vals[j]) / (x_vals[i] - x_vals[j])
        poly += y_vals[i] * L_i
    return sp.simplify(poly)

def newton_gregory_adelante(x_vals, y_vals):
    diffs = np.diff(x_vals)
    h = diffs[0]
    if not np.allclose(diffs, h):
        raise ValueError("Para usar Newton-Gregory el paso (h) entre los X debe ser constante.")
        
    n = len(y_vals)
    diff_table = np.zeros((n, n))
    diff_table[:, 0] = y_vals
    
    for j in range(1, n):
        for i in range(n - j):
            diff_table[i][j] = diff_table[i+1][j-1] - diff_table[i][j-1]
            
    x = sp.Symbol('x')
    s = (x - x_vals[0]) / h
    poly = diff_table[0, 0]
    s_term = 1
    for k in range(1, n):
        s_term = s_term * (s - (k - 1))
        poly += (diff_table[0, k] * s_term) / math.factorial(k)
        
    return sp.simplify(poly), diff_table

def cota_error_interp(x_vals, x_eval, M_next):
    n_plus_1 = len(x_vals)
    productoria = 1
    for xi in x_vals: productoria *= abs(x_eval - xi)
    return (M_next / math.factorial(n_plus_1)) * productoria

# ==========================================
# MÓDULO 3: INTEGRACIÓN Y MONTE CARLO
# ==========================================
def eval_func(f_str, x_vals):
    x = sp.Symbol('x')
    f_sym = sp.sympify(f_str)
    f = sp.lambdify(x, f_sym, 'numpy')
    
    # Evaluamos ignorando alertas de división por cero
    with np.errstate(divide='ignore', invalid='ignore'):
        y_vals = np.array(f(np.array(x_vals)), dtype=float)
        
    # Si hay un NaN, aplicamos L'Hôpital automáticamente
    for i, val in enumerate(y_vals):
        if np.isnan(val) or np.isinf(val):
            punto_fallado = x_vals[i]
            limite_real = sp.limit(f_sym, x, punto_fallado)
            y_vals[i] = float(limite_real)
            
    return y_vals

def trapecio_compuesto(f_str, a, b, n):
    x_vals = np.linspace(a, b, n + 1)
    y_vals = eval_func(f_str, x_vals)
    h = (b - a) / n
    integral = (h / 2) * (y_vals[0] + 2 * np.sum(y_vals[1:-1]) + y_vals[-1])
    return integral, x_vals, y_vals

def simpson_13_compuesta(f_str, a, b, n):
    if n % 2 != 0: return None, None, None, "El número de subintervalos (n) debe ser par."
    x_vals = np.linspace(a, b, n + 1)
    y_vals = eval_func(f_str, x_vals)
    h = (b - a) / n
    suma_impares = np.sum(y_vals[1:-1:2])
    suma_pares = np.sum(y_vals[2:-2:2])
    integral = (h / 3) * (y_vals[0] + 4 * suma_impares + 2 * suma_pares + y_vals[-1])
    return integral, x_vals, y_vals, None

def simpson_38_compuesta(f_str, a, b, n):
    if n % 3 != 0: return None, None, None, "El número de subintervalos (n) debe ser múltiplo de 3."
    x_vals = np.linspace(a, b, n + 1)
    y_vals = eval_func(f_str, x_vals)
    h = (b - a) / n
    suma = y_vals[0] + y_vals[-1]
    for i in range(1, n):
        if i % 3 == 0: suma += 2 * y_vals[i]
        else: suma += 3 * y_vals[i]
    integral = (3 * h / 8) * suma
    return integral, x_vals, y_vals, None

# ==========================================
# MÓDULO 4: DERIVACIÓN NUMÉRICA
# ==========================================
def tabla_diferencias_finitas(f_str, x_start, h, n_points, metodo):
    x_sym = sp.Symbol('x')
    try:
        f_expr = sp.sympify(f_str)
        f = sp.lambdify(x_sym, f_expr, 'numpy')
        df_expr = sp.diff(f_expr, x_sym)
        d2f_expr = sp.diff(df_expr, x_sym)
        f1_exact = sp.lambdify(x_sym, df_expr, 'numpy')
        f2_exact = sp.lambdify(x_sym, d2f_expr, 'numpy')
    except Exception as e:
        return None, f"Error analizando la función: {e}"

    x_vals = [x_start + i * h for i in range(n_points)]
    resultados = []
    
    for xv in x_vals:
        try:
            if metodo == "Hacia Adelante":
                d1 = (f(xv + h) - f(xv)) / h
                d2 = (f(xv + 2*h) - 2*f(xv + h) + f(xv)) / (h**2)
            elif metodo == "Hacia Atrás":
                d1 = (f(xv) - f(xv - h)) / h
                d2 = (f(xv) - 2*f(xv - h) + f(xv - 2*h)) / (h**2)
            elif metodo == "Central":
                d1 = (f(xv + h) - f(xv - h)) / (2*h)
                d2 = (f(xv + h) - 2*f(xv) + f(xv - h)) / (h**2)

            ex1 = float(f1_exact(xv))
            ex2 = float(f2_exact(xv))
            
            resultados.append({
                "x": xv,
                "f'(x) Num.": d1, "f'(x) Real": ex1, "Error Abs. 1ra": abs(ex1 - d1),
                "f''(x) Num.": d2, "f''(x) Real": ex2, "Error Abs. 2da": abs(ex2 - d2)
            })
        except Exception as e:
            return None, f"Error al calcular en x={xv}: {e}"
            
    return pd.DataFrame(resultados), None

# ==========================================
# MÓDULO 5: EDOs
# ==========================================
def euler_edo(f_str, x0, y0, h, x_final, exact_func=None):
    x_sym, y_sym = sp.symbols('x y')
    f = sp.lambdify((x_sym, y_sym), sp.sympify(f_str), 'numpy')
    x_vals, y_vals = [x0], [y0]
    exact_vals = []
    if exact_func: exact_vals.append(exact_func(x0))

    n_steps = int(np.round((x_final - x0) / h))
    for i in range(n_steps):
        y_next = y_vals[-1] + h * f(x_vals[-1], y_vals[-1])
        x_next = x_vals[-1] + h
        x_vals.append(x_next)
        y_vals.append(y_next)
        if exact_func: exact_vals.append(exact_func(x_next))
            
    df = pd.DataFrame({"x": x_vals, "y_{n+1} (Euler)": y_vals})
    if exact_func:
        df["y Real"] = exact_vals
        df["Error (Euler vs Real)"] = abs(df["y Real"] - df["y_{n+1} (Euler)"])
    return df

def rk4_edo(f_str, x0, y0, h, x_final, exact_func=None):
    x_sym, y_sym = sp.symbols('x y')
    f = sp.lambdify((x_sym, y_sym), sp.sympify(f_str), 'numpy')
    x_vals, y_vals = [x0], [y0]
    k1_l, k2_l, k3_l, k4_l = [None], [None], [None], [None]
    exact_vals = []
    if exact_func: exact_vals.append(exact_func(x0))

    n_steps = int(np.round((x_final - x0) / h))
    for i in range(n_steps):
        xi, yi = x_vals[-1], y_vals[-1]
        k1 = f(xi, yi)
        k2 = f(xi + h/2, yi + (h/2)*k1)
        k3 = f(xi + h/2, yi + (h/2)*k2)
        k4 = f(xi + h, yi + h*k3)
        y_next = yi + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
        x_next = xi + h
        
        k1_l.append(k1); k2_l.append(k2); k3_l.append(k3); k4_l.append(k4)
        x_vals.append(x_next)
        y_vals.append(y_next)
        if exact_func: exact_vals.append(exact_func(x_next))
            
    df = pd.DataFrame({
        "x": x_vals, "k1": k1_l, "k2": k2_l, "k3": k3_l, "k4": k4_l, "y_{n+1} (RK4)": y_vals
    })
    
    if exact_func:
        df["y Real"] = exact_vals
        df["Error (RK4 vs Real)"] = abs(df["y Real"] - df["y_{n+1} (RK4)"])
    return df

# ==========================================
# INTERFAZ STREAMLIT PRO
# ==========================================
st.set_page_config(page_title="Simulador Numérico PRO", layout="wide")
st.title("🚀 Laboratorio de Modelado y Simulación PRO")

st.sidebar.header("Módulos")
st.sidebar.markdown("**Parcial 1**")
st.sidebar.markdown("**Parcial 2**")
categoria = st.sidebar.radio("Seleccioná la categoría:", [
    # --- Parcial 1 ---
    "Calculadora Analítica", "Búsqueda de Raíces", "Interpolación",
    "Integración Numérica", "Derivación Numérica", "Ecuaciones Diferenciales",
    # --- Parcial 2 ---
    "Sistemas Autónomos 1D", "Bifurcaciones", "Sistemas Lineales 2D",
    "Sistemas No Homogéneos 2D", "EDO Orden Superior → Sistema",
    "Sistemas No Lineales 2D", "Aplicaciones (Hamilton/Volterra)"
])
st.markdown("---")

# ----------------- CALCULADORA ANALÍTICA -----------------
if categoria == "Calculadora Analítica":
    st.header("🧮 Calculadora Analítica Exacta")
    f_input = st.text_input("Función f(x)", value="sin(x) + exp(x)", key="funcion_input")
    teclado_matematico()
    operacion = st.radio("¿Qué querés calcular?", ["Derivada Primera f'(x)", "Integral Indefinida ∫f(x)dx"])
    if st.button("Calcular Exacto"):
        x = sp.Symbol('x')
        try:
            f_sym = sp.sympify(f_input)
            if operacion == "Derivada Primera f'(x)": res = sp.diff(f_sym, x)
            else: res = sp.integrate(f_sym, x)
            st.latex(sp.latex(res))
        except Exception as e: st.error(f"Error de sintaxis: {e}")

# ----------------- BÚSQUEDA DE RAÍCES -----------------
elif categoria == "Búsqueda de Raíces":
    metodo = st.sidebar.selectbox("Método:", ["Bisección", "Newton-Raphson", "Punto Fijo", "Aceleración de Aitken"])
    f_input = st.text_input("Función (o g(x) para Punto Fijo)", value="exp(x) - 2", key="funcion_input")
    teclado_matematico()
    c1, c2, c3 = st.columns(3)
    a_in, b_in, x0_in = None, None, None
    if metodo == "Bisección":
        a_in = c1.text_input("a", value="0")
        b_in = c2.text_input("b", value="2")
    else: x0_in = c1.text_input("x0", value="1")
    tol_in = c3.text_input("Tolerancia", value="1e-5")
    
    if st.button("Calcular Raíz"):
        a, b, x0, tol = parse_math(a_in), parse_math(b_in), parse_math(x0_in), parse_math(tol_in)
        if tol is None: st.error("Error en datos.")
        else:
            df, err = None, None
            if metodo == "Bisección": df, err = biseccion(f_input, a, b, tol, 100)
            elif metodo == "Newton-Raphson": df, err = newton_raphson(f_input, x0, tol, 100)
            elif metodo == "Punto Fijo": df, err = punto_fijo(f_input, x0, tol, 100)
            elif metodo == "Aceleración de Aitken": df, err = punto_fijo_aitken(f_input, x0, tol, 100)
                
            if err: st.error(f"⚠️ {err}")
            else:
                st.dataframe(df, use_container_width=True)
                st.success("Cálculo Finalizado.")

# ----------------- INTERPOLACIÓN -----------------
elif categoria == "Interpolación":
    metodo = st.sidebar.selectbox("Método:", ["Lagrange", "Diferencias Finitas (Newton-Gregory Adelante)"])
    c1, c2 = st.columns(2)
    x_input = c1.text_input("Valores X (separados por coma)", value="0, 1, 2")
    y_input = c2.text_input("Valores Y (separados por coma)", value="0, 0.6931, 1.0986")
    
    st.markdown("### 🎯 Evaluación y Error")
    ce1, ce2 = st.columns(2)
    x_eval_in = ce1.text_input("Punto a evaluar (x)", value="0.45")
    f_original = ce2.text_input("Función original f(x) (opcional)", value="log(x+1)")

    if st.button("Calcular Polinomio"):
        x_eval = parse_math(x_eval_in)
        if x_eval is None: st.error("Punto inválido.")
        else:
            try:
                x_vals = [float(parse_math(x.strip())) for x in x_input.split(',')]
                y_vals = [float(parse_math(y.strip())) for y in y_input.split(',')]
                n = len(x_vals) - 1
                if metodo == "Lagrange": poly_sym = lagrange_interp(x_vals, y_vals)
                else: 
                    poly_sym, tabla_diff = newton_gregory_adelante(x_vals, y_vals)
                    st.write("**Tabla de Diferencias Finitas:**")
                    st.dataframe(pd.DataFrame(tabla_diff))
                    
                st.latex(f"P(x) = {sp.latex(poly_sym)}")
                x_sym = sp.Symbol('x')
                p_x_eval = float(poly_sym.subs(x_sym, x_eval).evalf())
                st.success(f"Valor aproximado P({x_eval}) = **{p_x_eval:.6f}**")
                
                if f_original:
                    f_sym = sp.sympify(f_original)
                    f_real_val = float(f_sym.subs(x_sym, x_eval).evalf())
                    error_local = abs(f_real_val - p_x_eval)
                    f_deriv = f_sym
                    for _ in range(n + 1): f_deriv = sp.diff(f_deriv, x_sym)
                    f_deriv_lamb = sp.lambdify(x_sym, f_deriv, 'numpy')
                    t_vals = np.linspace(min(x_vals), max(x_vals), 1000)
                    try: M_auto = np.max(np.abs(f_deriv_lamb(t_vals)))
                    except: M_auto = 1.0
                    cota = cota_error_interp(x_vals, x_eval, M_auto)
                    col_err1, col_err2 = st.columns(2)
                    col_err1.metric("Error Local |f(x)-P(x)|", f"{error_local:.6e}")
                    col_err2.metric(f"Cota de Error Teórica (M={M_auto:.2f})", f"{cota:.6e}")
                
                f_lamb = sp.lambdify(x_sym, poly_sym, 'numpy')
                x_plot = np.linspace(min(x_vals) - 0.5, max(x_vals) + 0.5, 400)
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(x_plot, f_lamb(x_plot), label="P(x)", color="blue")
                if f_original: ax.plot(x_plot, sp.lambdify(x_sym, f_sym, 'numpy')(x_plot), label="f(x)", color="gray", linestyle="--")
                ax.scatter(x_vals, y_vals, color="red")
                ax.scatter([x_eval], [p_x_eval], color="green", marker="X", s=100)
                ax.legend(); st.pyplot(fig)
            except Exception as e: st.error(f"Error: {e}")

# ----------------- INTEGRACIÓN NUMÉRICA -----------------
elif categoria == "Integración Numérica":
    metodo = st.sidebar.selectbox("Método:", ["Métodos Clásicos (Trapecio/Simpson)", "Monte Carlo (Doble 2D)"])
    
    if metodo == "Métodos Clásicos (Trapecio/Simpson)":
        f_input = st.text_input("Función a integrar", value="log(x+1)/x", key="funcion_input")
        teclado_matematico()
        metodo_clasico = st.selectbox("Sub-método", ["Regla del Trapecio", "Simpson 1/3 Compuesta", "Simpson 3/8 Compuesta"])
        c1, c2 = st.columns(2)
        a_in = c1.text_input("a (Límite Inferior)", value="0")
        b_in = c2.text_input("b (Límite Superior)", value="1")
        n = st.number_input("Subintervalos (n)", min_value=1, value=4, step=1)
        
        st.markdown("### 📉 Error de Truncamiento Teórico")
        xi_in = st.text_input("Punto ξ (xi) para evaluar el error (Opcional)", value="0.5")
        
        if st.button("Calcular Integral"):
            a, b = parse_math(a_in), parse_math(b_in)
            xi_eval = parse_math(xi_in) if xi_in else None
            
            if None in (a, b): st.error("Límites inválidos.")
            else:
                res, x_int, y_int, err = None, None, None, None
                if metodo_clasico == "Regla del Trapecio": res, x_int, y_int = trapecio_compuesto(f_input, a, b, n)
                elif metodo_clasico == "Simpson 1/3 Compuesta": res, x_int, y_int, err = simpson_13_compuesta(f_input, a, b, n)
                elif metodo_clasico == "Simpson 3/8 Compuesta": res, x_int, y_int, err = simpson_38_compuesta(f_input, a, b, n)
                
                if err: st.error(err)
                else: 
                    try:
                        x_sym = sp.Symbol('x')
                        f_sym = sp.sympify(f_input)
                        exact_val = float(sp.integrate(f_sym, (x_sym, a, b)).evalf())
                        error_abs = abs(exact_val - res)
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Integral Aproximada", f"{res:.6f}")
                        col2.metric("Valor Exacto Analítico", f"{exact_val:.6f}")
                        col3.metric("Error Absoluto", f"{error_abs:.6e}")
                    except Exception: st.success(f"Integral ≈ **{res:.6f}**")
                    
                    if xi_eval is not None:
                        try:
                            h_val = (b - a) / n
                            x_sym = sp.Symbol('x')
                            f_sym = sp.sympify(f_input)
                            if metodo_clasico == "Regla del Trapecio":
                                f_deriv = sp.diff(f_sym, x_sym, 2)
                                deriv_val = float(f_deriv.subs(x_sym, xi_eval).evalf())
                                error_trunc = -((b - a) * (h_val**2) / 12) * deriv_val
                                st.info(f"**Error Teórico (Trapecio):** $E_T = -\\frac{{(b-a)h^2}}{{12}} f''({xi_eval}) \\approx {error_trunc:.6e}$")
                            elif metodo_clasico == "Simpson 1/3 Compuesta":
                                f_deriv = sp.diff(f_sym, x_sym, 4)
                                deriv_val = float(f_deriv.subs(x_sym, xi_eval).evalf())
                                error_trunc = -((b - a) * (h_val**4) / 180) * deriv_val
                                st.info(f"**Error Teórico (Simpson 1/3):** $E_S = -\\frac{{(b-a)h^4}}{{180}} f^{{(4)}}({xi_eval}) \\approx {error_trunc:.6e}$")
                            elif metodo_clasico == "Simpson 3/8 Compuesta":
                                f_deriv = sp.diff(f_sym, x_sym, 4)
                                deriv_val = float(f_deriv.subs(x_sym, xi_eval).evalf())
                                error_trunc = -((b - a) * (h_val**4) / 80) * deriv_val
                                st.info(f"**Error Teórico (Simpson 3/8):** $E_{{S3/8}} = -\\frac{{(b-a)h^4}}{{80}} f^{{(4)}}({xi_eval}) \\approx {error_trunc:.6e}$")
                        except Exception as e:
                            st.warning(f"No se pudo calcular el error de truncamiento analítico: {e}")

                    fig, ax = plt.subplots(figsize=(10, 4))
                    x_curv = np.linspace(a, b, 200)
                    y_curv = eval_func(f_input, x_curv)
                    ax.plot(x_curv, y_curv, 'k-', linewidth=2, label='f(x)')
                    for i in range(n):
                        xs = [x_int[i], x_int[i], x_int[i+1], x_int[i+1]]
                        ys = [0, y_int[i], y_int[i+1], 0]
                        ax.fill(xs, ys, 'skyblue', edgecolor='blue', alpha=0.4)
                    ax.legend(); st.pyplot(fig)

    elif metodo == "Monte Carlo (Doble 2D)":
        st.subheader("🎲 Integración por Monte Carlo (2D)")
        st.info("Calcula la integral doble de f(x,y) sobre una región rectangular [a,b] × [c,d]")
        
        f_input = st.text_input("Función f(x,y)", value="x * exp(y)", key="funcion_input_mc")
        teclado_matematico() 
        
        col1, col2 = st.columns(2)
        ax_in = col1.text_input("Límite Inferior X (a)", value="0")
        bx_in = col2.text_input("Límite Superior X (b)", value="1")
        
        col3, col4 = st.columns(2)
        ay_in = col3.text_input("Límite Inferior Y (c)", value="1")
        by_in = col4.text_input("Límite Superior Y (d)", value="3")
        
        cc1, cc2 = st.columns(2)
        n_in = cc1.number_input("Cantidad de Muestras (n)", min_value=10, value=10000, step=100)
        seed_in = cc2.number_input("Semilla (Opcional, -1 para aleatorio)", value=0, step=1)
        
        if st.button("Ejecutar Monte Carlo"):
            ax_val, bx_val = parse_math(ax_in), parse_math(bx_in)
            ay_val, by_val = parse_math(ay_in), parse_math(by_in)
            
            if None in (ax_val, bx_val, ay_val, by_val):
                st.error("Error: Revisá los límites de integración.")
            else:
                x_sym, y_sym = sp.symbols('x y')
                try:
                    f_sym = sp.sympify(f_input)
                    f_lamb = sp.lambdify((x_sym, y_sym), f_sym, 'numpy')
                    
                    if seed_in != -1: np.random.seed(int(seed_in))
                        
                    x_r = np.random.uniform(ax_val, bx_val, int(n_in))
                    y_r = np.random.uniform(ay_val, by_val, int(n_in))
                    
                    f_vals = f_lamb(x_r, y_r)
                    if isinstance(f_vals, (int, float)): f_vals = np.full(int(n_in), f_vals)
                        
                    area = (bx_val - ax_val) * (by_val - ay_val)
                    muestras_integrales = area * f_vals
                    
                    media = np.mean(muestras_integrales)
                    varianza = np.var(muestras_integrales, ddof=1)
                    desvio = np.std(muestras_integrales, ddof=1)
                    error_estandar = desvio / np.sqrt(n_in)
                    
                    ic_min = media - 1.96 * error_estandar
                    ic_max = media + 1.96 * error_estandar
                    
                    st.success(f"### Integral Aproximada: **{media:.6f}**")
                    
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Error Estándar (EE)", f"{error_estandar:.6f}")
                    m2.metric("Desviación Estándar (S)", f"{desvio:.6f}")
                    m3.metric("Varianza Muestral (S²)", f"{varianza:.6f}")
                    
                    st.info(f"**Intervalo de Confianza (95%):** [ {ic_min:.6f}  ,  {ic_max:.6f} ]")
                    
                    try:
                        exact_val = float(sp.integrate(sp.integrate(f_sym, (y_sym, ay_val, by_val)), (x_sym, ax_val, bx_val)).evalf())
                        st.write(f"*(Valor Exacto Analítico de referencia: {exact_val:.6f})*")
                    except: pass
                except Exception as e: st.error(f"Error procesando la función: {e}")

# ----------------- DERIVACIÓN NUMÉRICA -----------------
elif categoria == "Derivación Numérica":
    f_input = st.text_input("Función", value="sin(x)", key="funcion_input")
    teclado_matematico()
    c1, c2, c3, c4 = st.columns(4)
    x_val_in = c1.text_input("x Inicial ($x_0$)", value="0")
    h_in = c2.text_input("Paso (h)", value="pi/10")
    n_points = c3.number_input("Cant. de Puntos (n)", min_value=1, value=5, step=1)
    tipo = c4.selectbox("Método", ["Hacia Adelante", "Hacia Atrás", "Central"])
    
    if st.button("Generar Tabla de Derivadas"):
        x_val, h = parse_math(x_val_in), parse_math(h_in)
        if None in (x_val, h): st.error("Error: Escribí constantes válidas.")
        else:
            df, err = tabla_diferencias_finitas(f_input, x_val, h, n_points, tipo)
            if err: st.error(err)
            else: st.dataframe(df, use_container_width=True)

# ----------------- ECUACIONES DIFERENCIALES -----------------
elif categoria == "Ecuaciones Diferenciales":
    st.info("Ingresá la Ecuación Diferencial en la forma y' = f(x,y). Por ejemplo, y - x**2 + 1")
    f_input = st.text_input("f(x,y)", value="y * sin(x)", key="funcion_input")
    teclado_matematico()
    c1, c2, c3, c4 = st.columns(4)
    x0_in = c1.text_input("x0 inicial", value="0")
    y0_in = c2.text_input("y0 inicial", value="2")
    h_in = c3.text_input("Paso (h)", value="pi/10")
    x_final_in = c4.text_input("x final a evaluar", value="pi")
    
    if st.button("Resolver PVI"):
        x0, y0, h, x_final = parse_math(x0_in), parse_math(y0_in), parse_math(h_in), parse_math(x_final_in)
        if None in (x0, y0, h, x_final): st.error("Error en datos.")
        else:
            x_sym = sp.Symbol('x')
            y_func = sp.Function('y')
            exact_lamb = None
            try:
                f_expr = sp.sympify(f_input).subs(sp.Symbol('y'), y_func(x_sym))
                eq = sp.Eq(y_func(x_sym).diff(x_sym), f_expr)
                sol = sp.dsolve(eq, y_func(x_sym), ics={y_func(x0): y0})
                exact_expr = sol.rhs
                exact_lamb = sp.lambdify(x_sym, exact_expr, 'numpy')
                st.success(f"🎯 Solución Exacta: $$y(x) = {sp.latex(exact_expr)}$$")
            except Exception: pass
            
            df_euler = euler_edo(f_input, x0, y0, h, x_final, exact_lamb) 
            df_rk4 = rk4_edo(f_input, x0, y0, h, x_final, exact_lamb)     
            
            st.subheader("⚖️ Tabla Comparativa General")
            df_final = pd.merge(df_euler[["x", "y_{n+1} (Euler)"]], df_rk4[["x", "y_{n+1} (RK4)"]], on="x")
            if exact_lamb: df_final["y Real (Exacto)"] = df_euler["y Real"]
            st.dataframe(df_final, use_container_width=True)
            
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(df_final["x"], df_final["y_{n+1} (Euler)"], label="Euler", marker="o", alpha=0.7)
            ax.plot(df_final["x"], df_final["y_{n+1} (RK4)"], label="RK4", marker="x", alpha=0.7)
            if exact_lamb:
                x_dense = np.linspace(x0, x_final, 200)
                ax.plot(x_dense, exact_lamb(x_dense), 'k--', label="Exacta", linewidth=2)
            ax.legend(); st.pyplot(fig)

# ==========================================
# UTILIDADES PARCIAL 2
# ==========================================
def parse_lista_numeros(s):
    """'1, 2, pi/3, sqrt(2)' -> [1.0, 2.0, 1.047..., 1.414...]"""
    try:
        return [float(parse_math(v.strip())) for v in s.split(",") if v.strip()]
    except Exception:
        return None

def clasificar_eq_1d(f_prime_val):
    """Clasifica equilibrio en 1D según f'(x*)."""
    if abs(f_prime_val) < 1e-9: return "No hiperbólico (f'=0): usar análisis no lineal"
    return "Estable (atractor)" if f_prime_val < 0 else "Inestable (repulsor)"

def clasificar_2d_lineal(A_np):
    """Recibe matriz 2x2 numpy. Devuelve (tipo, autovalores, autovectores)."""
    eigvals, eigvecs = np.linalg.eig(A_np)
    l1, l2 = eigvals
    tr = np.trace(A_np); det = np.linalg.det(A_np)
    disc = tr**2 - 4*det
    # Casos canónicos
    if np.isclose(det, 0):
        tipo = "Degenerado (det=0, hay una recta de equilibrios)"
    elif np.iscomplex(l1) and not np.isclose(l1.imag, 0):
        if np.isclose(tr, 0): tipo = "Centro (oscilación pura)"
        elif tr < 0: tipo = "Foco/Espiral estable"
        else: tipo = "Foco/Espiral inestable"
    else:
        # Reales
        l1r, l2r = float(np.real(l1)), float(np.real(l2))
        if l1r * l2r < 0: tipo = "Silla (saddle)"
        elif l1r < 0 and l2r < 0:
            tipo = "Nodo estable propio" if np.isclose(l1r, l2r) and not np.isclose(disc, 0) else "Nodo estable"
            if np.isclose(disc, 0): tipo = "Nodo estable degenerado/impropio"
        elif l1r > 0 and l2r > 0:
            tipo = "Nodo inestable"
            if np.isclose(disc, 0): tipo = "Nodo inestable degenerado/impropio"
    return tipo, (l1, l2), eigvecs, (tr, det, disc)

def dibujar_campo_2d(ax, fx, fy, xlim, ylim, n=22, color="gray", alpha=0.6):
    X, Y = np.meshgrid(np.linspace(xlim[0], xlim[1], n), np.linspace(ylim[0], ylim[1], n))
    with np.errstate(divide='ignore', invalid='ignore'):
        U = fx(X, Y); V = fy(X, Y)
        if np.isscalar(U): U = np.full_like(X, U)
        if np.isscalar(V): V = np.full_like(Y, V)
        M = np.hypot(U, V); M[M==0] = 1
        ax.quiver(X, Y, U/M, V/M, M, cmap="viridis", alpha=alpha, pivot="mid")

def integrar_trayectoria(fx, fy, x0, y0, t_max=20, dt=0.02):
    """RK4 para sistema (xdot, ydot) = (fx(x,y), fy(x,y))."""
    n = int(t_max/dt); xs = [x0]; ys = [y0]
    for _ in range(n):
        x, y = xs[-1], ys[-1]
        try:
            k1x, k1y = fx(x,y), fy(x,y)
            k2x, k2y = fx(x+dt*k1x/2, y+dt*k1y/2), fy(x+dt*k1x/2, y+dt*k1y/2)
            k3x, k3y = fx(x+dt*k2x/2, y+dt*k2y/2), fy(x+dt*k2x/2, y+dt*k2y/2)
            k4x, k4y = fx(x+dt*k3x, y+dt*k3y), fy(x+dt*k3x, y+dt*k3y)
            xn = x + dt/6*(k1x+2*k2x+2*k3x+k4x)
            yn = y + dt/6*(k1y+2*k2y+2*k3y+k4y)
            if abs(xn) > 1e6 or abs(yn) > 1e6: break
            xs.append(xn); ys.append(yn)
        except Exception: break
    return np.array(xs), np.array(ys)


# ==========================================
# MÓDULO 6: SISTEMAS AUTÓNOMOS 1D
# ==========================================
if categoria == "Sistemas Autónomos 1D":
    st.header("📈 Sistemas Dinámicos Autónomos 1D — ẋ = f(x)")
    st.info("Encuentra puntos de equilibrio, clasifica estabilidad por f'(x*), grafica diagrama de fase 1D y soluciones x(t).")
    f_input = st.text_input("f(x) — lado derecho de ẋ = f(x)", value="x*(100 - x)", key="funcion_input")
    teclado_matematico()
    c1, c2, c3 = st.columns(3)
    x_min_in = c1.text_input("x mín (rango fase)", value="-10")
    x_max_in = c2.text_input("x máx (rango fase)", value="110")
    x0s_in   = c3.text_input("Condiciones iniciales x₀ (coma)", value="-5, 20, 50, 95, 120")
    t_final_in = st.text_input("Tiempo final para integrar x(t)", value="0.2")

    if st.button("Analizar"):
        x = sp.Symbol('x', real=True)
        try:
            f_sym = sp.sympify(f_input)
            fp_sym = sp.diff(f_sym, x)
            eqs = sp.solve(sp.Eq(f_sym, 0), x)
            eqs_reales = [complex(e).real for e in eqs if abs(complex(e).imag) < 1e-9]

            st.subheader("Puntos de equilibrio y clasificación")
            rows = []
            for e in eqs_reales:
                fp_val = float(fp_sym.subs(x, e))
                rows.append({"x* (equilibrio)": e, "f'(x*)": fp_val, "Clasificación": clasificar_eq_1d(fp_val)})
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

            f_lamb = sp.lambdify(x, f_sym, 'numpy')
            x_min, x_max = parse_math(x_min_in), parse_math(x_max_in)
            x_grid = np.linspace(x_min, x_max, 600)
            y_grid = f_lamb(x_grid)
            if np.isscalar(y_grid): y_grid = np.full_like(x_grid, y_grid)

            # Diagrama de fase 1D (x vs xdot)
            fig, ax = plt.subplots(figsize=(10,4))
            ax.axhline(0, color='gray', lw=0.8)
            ax.plot(x_grid, y_grid, 'b-', lw=2, label=r'$\dot{x}=f(x)$')
            # flechas sobre eje x indicando signo
            xs_arr = np.linspace(x_min, x_max, 18)
            for xi in xs_arr:
                yi = float(f_lamb(xi))
                ax.annotate('', xy=(xi + np.sign(yi)*(x_max-x_min)*0.015, 0),
                            xytext=(xi, 0),
                            arrowprops=dict(arrowstyle='->', color='red' if yi>0 else 'blue'))
            for e in eqs_reales:
                color = 'green' if float(fp_sym.subs(x, e)) < 0 else 'red'
                ax.plot(e, 0, 'o', color=color, markersize=12, zorder=5)
                ax.annotate(f'x*={e:.3f}', xy=(e,0), xytext=(0,15), textcoords='offset points',
                            ha='center', color=color, fontsize=10)
            ax.set_xlabel('x'); ax.set_ylabel(r'$\dot{x}$'); ax.set_title('Diagrama de fase 1D')
            ax.grid(alpha=0.3); ax.legend()
            st.pyplot(fig)

            # x(t) para varias condiciones iniciales (Euler RK4 simple)
            x0s = parse_lista_numeros(x0s_in) or []
            t_final = parse_math(t_final_in)
            if x0s and t_final:
                dt = max(t_final/2000, 1e-4); N = int(t_final/dt)
                ts = np.linspace(0, t_final, N+1)
                fig2, ax2 = plt.subplots(figsize=(10,4))
                for x0 in x0s:
                    xs = [x0]
                    for _ in range(N):
                        xi = xs[-1]
                        try:
                            k1 = f_lamb(xi); k2 = f_lamb(xi + dt*k1/2)
                            k3 = f_lamb(xi + dt*k2/2); k4 = f_lamb(xi + dt*k3)
                            xn = xi + dt/6*(k1+2*k2+2*k3+k4)
                            if abs(xn) > 1e10: xs.append(np.nan); continue
                            xs.append(xn)
                        except Exception: xs.append(np.nan)
                    ax2.plot(ts, xs, label=f'x₀={x0:g}')
                for e in eqs_reales:
                    color = 'green' if float(fp_sym.subs(x, e)) < 0 else 'red'
                    ax2.axhline(e, color=color, linestyle='--', alpha=0.5)
                ax2.set_xlabel('t'); ax2.set_ylabel('x(t)'); ax2.set_title('Soluciones x(t)')
                ax2.legend(); ax2.grid(alpha=0.3)
                st.pyplot(fig2)

            # Intento solución analítica
            try:
                t = sp.Symbol('t', real=True); xf = sp.Function('x')
                eq_sym = sp.Eq(xf(t).diff(t), f_sym.subs(x, xf(t)))
                sol = sp.dsolve(eq_sym, xf(t))
                st.latex(f"Solución general: {sp.latex(sol)}")
            except Exception:
                st.caption("Sin solución cerrada elemental — análisis cualitativo aplicado.")
        except Exception as e:
            st.error(f"Error: {e}")


# ==========================================
# MÓDULO 7: BIFURCACIONES
# ==========================================
elif categoria == "Bifurcaciones":
    st.header("🌿 Bifurcaciones 1D — ẋ = f(x, r)")
    st.info("Encuentra puntos fijos como función del parámetro r, clasifica la bifurcación y grafica el diagrama.")
    f_input = st.text_input("f(x, r)", value="r*x - x**3", key="funcion_input")
    teclado_matematico()
    c1, c2, c3 = st.columns(3)
    r_min_in = c1.text_input("r mín", value="-2")
    r_max_in = c2.text_input("r máx", value="2")
    n_r_in = c3.number_input("N puntos r", min_value=50, value=400, step=50)

    if st.button("Analizar Bifurcación"):
        x, r = sp.symbols('x r', real=True)
        try:
            f_sym = sp.sympify(f_input)
            fp_sym = sp.diff(f_sym, x)
            r_min, r_max = parse_math(r_min_in), parse_math(r_max_in)
            r_vals = np.linspace(r_min, r_max, int(n_r_in))

            # Ramas de equilibrio: x* en función de r
            sol = sp.solve(sp.Eq(f_sym, 0), x)
            st.write("**Ramas simbólicas x*(r):**")
            for k, s in enumerate(sol):
                st.latex(f"x_{k+1}^*(r) = {sp.latex(s)}")

            # Evaluar cada rama
            ramas_estables_x = []; ramas_estables_r = []
            ramas_inestables_x = []; ramas_inestables_r = []
            for rv in r_vals:
                for s in sol:
                    try:
                        xv = complex(s.subs(r, rv))
                        if abs(xv.imag) > 1e-7: continue
                        xv = xv.real
                        fpv = float(fp_sym.subs({x: xv, r: rv}))
                        if fpv < 0:
                            ramas_estables_x.append(xv); ramas_estables_r.append(rv)
                        else:
                            ramas_inestables_x.append(xv); ramas_inestables_r.append(rv)
                    except Exception: continue

            fig, ax = plt.subplots(figsize=(10,6))
            ax.scatter(ramas_estables_r, ramas_estables_x, s=4, color='green', label='Estable (f\'<0)')
            ax.scatter(ramas_inestables_r, ramas_inestables_x, s=4, color='red', label='Inestable (f\'>0)')
            ax.axhline(0, color='gray', lw=0.5); ax.axvline(0, color='gray', lw=0.5)
            ax.set_xlabel('r'); ax.set_ylabel('x*')
            ax.set_title('Diagrama de bifurcación')
            ax.legend(); ax.grid(alpha=0.3)
            st.pyplot(fig)

            # Detección heurística del tipo
            st.subheader("Tipo de bifurcación (heurístico)")
            # Buscar valor de r donde aparece/desaparece una rama (saddle-node) o coinciden ramas (transcritical/pitchfork)
            r_bifs = sp.solve([sp.Eq(f_sym,0), sp.Eq(fp_sym,0)], [x, r], dict=True)
            if r_bifs:
                for sol_b in r_bifs:
                    st.write(f"- Punto crítico: x* = {sol_b.get(x, '?')}, r = {sol_b.get(r, '?')}")
                    # Segunda derivada en x para clasificación
                    fxx = sp.diff(f_sym, x, 2).subs(sol_b)
                    fr = sp.diff(f_sym, r).subs(sol_b)
                    fxxx = sp.diff(f_sym, x, 3).subs(sol_b)
                    try:
                        st.write(f"  - $f_{{xx}}={sp.simplify(fxx)}$, $f_r={sp.simplify(fr)}$, $f_{{xxx}}={sp.simplify(fxxx)}$")
                        if sp.simplify(fr) != 0 and sp.simplify(fxx) != 0:
                            st.success("Tipo: **Saddle-node (silla-nodo)**")
                        elif sp.simplify(fr) == 0 and sp.simplify(fxx) != 0:
                            st.success("Tipo: **Transcrítica**")
                        elif sp.simplify(fr) == 0 and sp.simplify(fxx) == 0 and sp.simplify(fxxx) != 0:
                            st.success("Tipo: **Pitchfork (tridente)**")
                        else:
                            st.warning("Forma normal no inmediata — revisar manualmente.")
                    except Exception: pass
            else:
                st.caption("Sin punto crítico donde f=0 y f'=0 simultáneamente — sistema sin bifurcación en este rango.")
        except Exception as e:
            st.error(f"Error: {e}")


# ==========================================
# MÓDULO 8: SISTEMAS LINEALES 2D
# ==========================================
elif categoria == "Sistemas Lineales 2D":
    st.header("📐 Sistemas Lineales 2D — Ẋ = A·X")
    st.info("Calcula autovalores, autovectores, clasifica equilibrio y grafica retrato de fase + campo vectorial + nullclines.")
    st.markdown("Matriz A = [[a, b], [c, d]]:")
    c1, c2, c3, c4 = st.columns(4)
    a_in = c1.text_input("a", value="-3"); b_in = c2.text_input("b", value="-4")
    c_in = c3.text_input("c", value="2");  d_in = c4.text_input("d", value="1")
    c5, c6, c7 = st.columns(3)
    rng_in = c5.text_input("Rango |x|,|y|", value="5")
    n_traj_in = c6.number_input("Nº trayectorias", min_value=0, value=10, step=1)
    t_max_in = c7.text_input("t máx integración", value="10")

    if st.button("Analizar 2D Lineal"):
        try:
            a, b, c, d = parse_math(a_in), parse_math(b_in), parse_math(c_in), parse_math(d_in)
            A = np.array([[a, b], [c, d]], dtype=float)
            tipo, (l1, l2), V, (tr, det, disc) = clasificar_2d_lineal(A)
            st.subheader("Análisis")
            st.write(f"**A** = {A.tolist()}")
            st.write(f"- Traza tr(A) = **{tr:.6f}**")
            st.write(f"- Determinante det(A) = **{det:.6f}**")
            st.write(f"- Discriminante tr² - 4·det = **{disc:.6f}**")
            st.write(f"- Autovalores: λ₁ = **{l1}**, λ₂ = **{l2}**")
            st.write(f"- Autovectores (columnas): {V.tolist()}")
            st.success(f"**Clasificación: {tipo}**")
            st.caption("Eq único en origen (0,0) salvo det=0.")

            rng = parse_math(rng_in)
            fx = lambda X, Y: a*X + b*Y
            fy = lambda X, Y: c*X + d*Y
            fig, ax = plt.subplots(figsize=(8,8))
            dibujar_campo_2d(ax, fx, fy, (-rng, rng), (-rng, rng))
            # Nullclines: f_x=0 -> a x + b y = 0; f_y=0 -> c x + d y = 0
            xs = np.linspace(-rng, rng, 100)
            if abs(b) > 1e-9: ax.plot(xs, -a*xs/b, 'b--', alpha=0.5, label='Nulclina ẋ=0')
            else: ax.axvline(0, color='b', ls='--', alpha=0.5, label='Nulclina ẋ=0')
            if abs(d) > 1e-9: ax.plot(xs, -c*xs/d, 'r--', alpha=0.5, label='Nulclina ẏ=0')
            else: ax.axvline(0, color='r', ls='--', alpha=0.5, label='Nulclina ẏ=0')
            # Autovectores reales
            for k, lk in enumerate([l1, l2]):
                if abs(np.imag(lk)) < 1e-9:
                    v = np.real(V[:,k]); v = v / (np.linalg.norm(v) + 1e-12) * rng
                    ax.plot([-v[0], v[0]], [-v[1], v[1]], 'k-', lw=2, alpha=0.7,
                            label=f'Autovector λ={np.real(lk):.3f}')
            # Trayectorias
            n_traj = int(n_traj_in); t_max = parse_math(t_max_in)
            for k in range(n_traj):
                ang = 2*np.pi*k/max(n_traj,1)
                x0, y0 = rng*0.8*np.cos(ang), rng*0.8*np.sin(ang)
                xs_t, ys_t = integrar_trayectoria(fx, fy, x0, y0, t_max=t_max, dt=0.02)
                ax.plot(xs_t, ys_t, color='purple', lw=1, alpha=0.7)
                # backwards en tiempo
                xs_b, ys_b = integrar_trayectoria(lambda X,Y: -fx(X,Y), lambda X,Y: -fy(X,Y), x0, y0, t_max=t_max, dt=0.02)
                ax.plot(xs_b, ys_b, color='purple', lw=1, alpha=0.4, ls=':')
            ax.plot(0, 0, 'ko', markersize=10, zorder=6, label='Eq (0,0)')
            ax.set_xlim(-rng, rng); ax.set_ylim(-rng, rng)
            ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_title(f'Retrato de fase — {tipo}')
            ax.legend(fontsize=8, loc='upper right'); ax.grid(alpha=0.3); ax.set_aspect('equal')
            st.pyplot(fig)

            # Intento solución analítica simbólica
            try:
                t = sp.Symbol('t', real=True)
                xs_f, ys_f = sp.Function('x'), sp.Function('y')
                sysym = [sp.Eq(xs_f(t).diff(t), a*xs_f(t) + b*ys_f(t)),
                         sp.Eq(ys_f(t).diff(t), c*xs_f(t) + d*ys_f(t))]
                sol = sp.dsolve(sysym)
                st.subheader("Solución general (sympy)")
                for s in sol: st.latex(sp.latex(s))
            except Exception as e:
                st.caption(f"Sin solución cerrada: {e}")
        except Exception as e:
            st.error(f"Error: {e}")


# ==========================================
# MÓDULO 9: SISTEMAS NO HOMOGÉNEOS 2D
# ==========================================
elif categoria == "Sistemas No Homogéneos 2D":
    st.header("➕ Sistemas Lineales No Homogéneos — Ẋ = A·X + f(t)")
    st.info("Resuelve simbólicamente, identifica si el forzado preserva o rompe el comportamiento del sistema homogéneo.")
    st.markdown("Matriz A = [[a, b], [c, d]]:")
    c1, c2, c3, c4 = st.columns(4)
    a_in = c1.text_input("a", value="0"); b_in = c2.text_input("b", value="-1")
    c_in = c3.text_input("c", value="-9"); d_in = c4.text_input("d", value="0")
    c5, c6 = st.columns(2)
    f1_in = c5.text_input("f₁(t) (forzado en ẋ)", value="1")
    f2_in = c6.text_input("f₂(t) (forzado en ẏ)", value="9")
    c7, c8, c9 = st.columns(3)
    x0_in = c7.text_input("x(0)", value="0"); y0_in = c8.text_input("y(0)", value="0")
    t_max_in = c9.text_input("t máx", value="10")

    if st.button("Resolver"):
        try:
            t = sp.Symbol('t', real=True)
            a, b, c, d = parse_math(a_in), parse_math(b_in), parse_math(c_in), parse_math(d_in)
            f1 = sp.sympify(f1_in); f2 = sp.sympify(f2_in)
            A_np = np.array([[a,b],[c,d]], dtype=float)
            tipo_h, evals, evecs, _ = clasificar_2d_lineal(A_np)
            st.write(f"**Sistema homogéneo Ẋ=AX:** {tipo_h}, autovalores {evals}")

            xs_f, ys_f = sp.Function('x'), sp.Function('y')
            x0, y0 = parse_math(x0_in), parse_math(y0_in)
            sysym = [sp.Eq(xs_f(t).diff(t), a*xs_f(t) + b*ys_f(t) + f1),
                     sp.Eq(ys_f(t).diff(t), c*xs_f(t) + d*ys_f(t) + f2)]
            try:
                sol = sp.dsolve(sysym, ics={xs_f(0): x0, ys_f(0): y0})
                st.subheader("Solución particular con CI")
                for s in sol: st.latex(sp.latex(s))
                x_sol_expr = sol[0].rhs; y_sol_expr = sol[1].rhs
                x_lamb = sp.lambdify(t, x_sol_expr, 'numpy')
                y_lamb = sp.lambdify(t, y_sol_expr, 'numpy')
                t_max = parse_math(t_max_in)
                ts = np.linspace(0, t_max, 600)
                fig, ax = plt.subplots(figsize=(10,4))
                ax.plot(ts, x_lamb(ts), label='x(t)'); ax.plot(ts, y_lamb(ts), label='y(t)')
                ax.set_xlabel('t'); ax.grid(alpha=0.3); ax.legend()
                st.pyplot(fig)
                # Retrato de fase 2D
                fig2, ax2 = plt.subplots(figsize=(7,7))
                ax2.plot(x_lamb(ts), y_lamb(ts), 'b-', lw=2)
                ax2.plot(x0, y0, 'go', markersize=10, label='Inicio')
                ax2.plot(x_lamb(ts)[-1], y_lamb(ts)[-1], 'rs', markersize=10, label='Final')
                ax2.set_xlabel('x'); ax2.set_ylabel('y'); ax2.grid(alpha=0.3); ax2.legend()
                ax2.set_title('Trayectoria forzada en plano (x,y)')
                st.pyplot(fig2)
            except Exception as e:
                st.error(f"Sympy no resolvió simbólicamente: {e}. Usando RK4 numérico.")
                f1_lamb = sp.lambdify(t, f1, 'numpy'); f2_lamb = sp.lambdify(t, f2, 'numpy')
                t_max = parse_math(t_max_in); dt = 0.01; N = int(t_max/dt)
                ts = [0.0]; xs = [x0]; ys = [y0]
                for _ in range(N):
                    tn, xn, yn = ts[-1], xs[-1], ys[-1]
                    fx = lambda T,X,Y: a*X + b*Y + float(f1_lamb(T))
                    fy = lambda T,X,Y: c*X + d*Y + float(f2_lamb(T))
                    k1x, k1y = fx(tn,xn,yn), fy(tn,xn,yn)
                    k2x, k2y = fx(tn+dt/2,xn+dt*k1x/2,yn+dt*k1y/2), fy(tn+dt/2,xn+dt*k1x/2,yn+dt*k1y/2)
                    k3x, k3y = fx(tn+dt/2,xn+dt*k2x/2,yn+dt*k2y/2), fy(tn+dt/2,xn+dt*k2x/2,yn+dt*k2y/2)
                    k4x, k4y = fx(tn+dt,xn+dt*k3x,yn+dt*k3y), fy(tn+dt,xn+dt*k3x,yn+dt*k3y)
                    xs.append(xn + dt/6*(k1x+2*k2x+2*k3x+k4x))
                    ys.append(yn + dt/6*(k1y+2*k2y+2*k3y+k4y))
                    ts.append(tn + dt)
                fig, ax = plt.subplots(figsize=(10,4))
                ax.plot(ts, xs, label='x(t)'); ax.plot(ts, ys, label='y(t)')
                ax.set_xlabel('t'); ax.legend(); ax.grid(alpha=0.3)
                st.pyplot(fig)

            # Análisis preservación / ruptura
            st.subheader("Preservación o ruptura del comportamiento")
            # Si f(t) acotada y sistema estable -> preservación; si f(t) crece sin cota o sistema centro forzado en resonancia -> ruptura
            es_acotado = True
            try:
                lim_pos = sp.limit(f1, t, sp.oo); lim2 = sp.limit(f2, t, sp.oo)
                if any(abs(complex(l)) > 1e10 for l in [lim_pos, lim2] if l.is_number):
                    es_acotado = False
            except Exception: es_acotado = False
            estable_h = all(np.real(l) < 0 for l in evals)
            if estable_h and es_acotado:
                st.success("Sistema **preserva** comportamiento: homogéneo estable + forzado acotado.")
            elif not estable_h and es_acotado:
                st.warning("Sistema homogéneo no estable: el forzado solo agrega solución particular pero las trayectorias divergen.")
            elif estable_h and not es_acotado:
                st.warning("Forzado no acotado puede dominar la dinámica → **ruptura** del régimen estacionario homogéneo.")
            else:
                st.error("Comportamiento totalmente dominado por el forzado y/o autovalores con parte real positiva.")
        except Exception as e:
            st.error(f"Error: {e}")


# ==========================================
# MÓDULO 10: EDO ORDEN SUPERIOR → SISTEMA 1ER ORDEN
# ==========================================
elif categoria == "EDO Orden Superior → Sistema":
    st.header("🔄 Conversión EDO orden n a sistema de 1er orden")
    st.info("Para y⁽ⁿ⁾ + aₙ₋₁ y⁽ⁿ⁻¹⁾ + … + a₀ y = f(t): genera variables x₁=y, x₂=y', … y la forma matricial.")
    edo_in = st.text_input("EDO (ej: y'' + 3*y' + 2*y - cos(t))", value="y'' + 3*y' + 2*y - 0")
    st.caption("Usar y, y', y'', y''' como notación. RHS = 0 implícito; meter forzado f(t) con signo opuesto al pasarlo al lado izquierdo.")
    c1, c2 = st.columns(2)
    f_forz_in = c1.text_input("Forzado f(t) explícito (opcional, sobrescribe parseo)", value="0")
    n_in = c2.number_input("Orden n (si parseo automático falla)", min_value=1, max_value=5, value=2)

    if st.button("Convertir"):
        try:
            t = sp.Symbol('t', real=True)
            # Parseo manual de la EDO en forma y^(n) + a_{n-1} y^(n-1) + ... + a_0 y = f(t)
            edo_str = edo_in.replace("y'''", "D3").replace("y''", "D2").replace("y'", "D1").replace("y", "Y")
            # Solicitar coeficientes manualmente
            st.subheader("Coeficientes (orden ascendente: a₀, a₁, …)")
            n = int(n_in)
            cols = st.columns(n+1)
            coefs = []
            for i in range(n):
                v = cols[i].text_input(f"a_{i}", value=("2" if i==0 else "3" if i==1 else "0"), key=f"coef_{i}")
                coefs.append(parse_math(v))
            f_forz = parse_math(f_forz_in)
            # Sistema: x1 = y, x2 = y', ..., xn = y^(n-1)
            # x1' = x2; x2' = x3; ...; xn' = -a_{n-1} x_n - ... - a_0 x_1 + f(t)
            A = np.zeros((n, n))
            for i in range(n-1):
                A[i, i+1] = 1
            for j in range(n):
                A[n-1, j] = -coefs[j]
            b_vec = np.zeros(n); b_vec[-1] = f_forz if isinstance(f_forz, (int,float)) else 0
            st.write("**Matriz A** =")
            st.write(pd.DataFrame(A, index=[f"x{i+1}'" for i in range(n)],
                                  columns=[f"x{j+1}" for j in range(n)]))
            st.write(f"**Vector forzado b** = {b_vec.tolist()}")
            st.markdown("**Forma matricial:** $\\dot{X} = A\\,X + b(t)$")
            st.write("Variables: $x_1 = y$, $x_2 = y'$, ..., $x_n = y^{(n-1)}$")
            if n == 2:
                tipo, evals, _, (tr, det, disc) = clasificar_2d_lineal(A)
                st.success(f"Sistema 2D clasificado: **{tipo}** (autovalores: {evals})")
        except Exception as e:
            st.error(f"Error: {e}")


# ==========================================
# MÓDULO 11: SISTEMAS NO LINEALES 2D
# ==========================================
elif categoria == "Sistemas No Lineales 2D":
    st.header("🌀 Sistemas No Lineales 2D — ẋ = f(x,y), ẏ = g(x,y)")
    st.info("Encuentra equilibrios, calcula jacobiano local, clasifica cada punto y grafica campo + nullclines + trayectorias.")
    c1, c2 = st.columns(2)
    fx_in = c1.text_input("f(x,y) = ẋ", value="x*(2 - x - y)", key="funcion_input")
    fy_in = c2.text_input("g(x,y) = ẏ", value="y*(3 - 2*x - y)", key="funcion_input_mc")
    teclado_matematico()
    c3, c4, c5, c6 = st.columns(4)
    xlim_in = c3.text_input("x rango", value="-1, 4")
    ylim_in = c4.text_input("y rango", value="-1, 4")
    n_traj_in = c5.number_input("Nº trayectorias", min_value=0, value=12, step=1)
    t_max_in = c6.text_input("t máx", value="15")

    if st.button("Analizar No Lineal"):
        try:
            x, y = sp.symbols('x y', real=True)
            fx_sym = sp.sympify(fx_in); fy_sym = sp.sympify(fy_in)
            # Equilibrios
            eqs = sp.solve([sp.Eq(fx_sym, 0), sp.Eq(fy_sym, 0)], [x, y], dict=True)
            eqs_reales = []
            for e in eqs:
                try:
                    xv = complex(e[x]); yv = complex(e[y])
                    if abs(xv.imag) < 1e-7 and abs(yv.imag) < 1e-7:
                        eqs_reales.append((xv.real, yv.real))
                except Exception: continue
            st.subheader("Equilibrios y clasificación (jacobiano local)")
            J = sp.Matrix([[sp.diff(fx_sym, x), sp.diff(fx_sym, y)],
                           [sp.diff(fy_sym, x), sp.diff(fy_sym, y)]])
            st.latex(f"J(x,y) = {sp.latex(J)}")
            rows = []
            for (xe, ye) in eqs_reales:
                J_num = np.array(J.subs({x: xe, y: ye}).evalf().tolist(), dtype=float)
                tipo, (l1, l2), _, _ = clasificar_2d_lineal(J_num)
                rows.append({"x*": xe, "y*": ye, "λ₁": str(l1), "λ₂": str(l2), "Tipo (linealización)": tipo})
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

            xlims = parse_lista_numeros(xlim_in); ylims = parse_lista_numeros(ylim_in)
            fx_lamb = sp.lambdify((x, y), fx_sym, 'numpy')
            fy_lamb = sp.lambdify((x, y), fy_sym, 'numpy')
            fig, ax = plt.subplots(figsize=(8, 8))
            dibujar_campo_2d(ax, fx_lamb, fy_lamb, xlims, ylims)
            # Nullclines numéricas
            xs_n = np.linspace(xlims[0], xlims[1], 400)
            ys_n = np.linspace(ylims[0], ylims[1], 400)
            X, Y = np.meshgrid(xs_n, ys_n)
            with np.errstate(all='ignore'):
                Fx = fx_lamb(X, Y); Fy = fy_lamb(X, Y)
                if np.isscalar(Fx): Fx = np.full_like(X, Fx)
                if np.isscalar(Fy): Fy = np.full_like(Y, Fy)
                ax.contour(X, Y, Fx, levels=[0], colors='blue', linestyles='--', alpha=0.6)
                ax.contour(X, Y, Fy, levels=[0], colors='red',  linestyles='--', alpha=0.6)
            # Trayectorias
            n_traj = int(n_traj_in); t_max = parse_math(t_max_in)
            for k in range(n_traj):
                ang = 2*np.pi*k/max(n_traj,1)
                x0 = (xlims[0]+xlims[1])/2 + (xlims[1]-xlims[0])*0.35*np.cos(ang)
                y0 = (ylims[0]+ylims[1])/2 + (ylims[1]-ylims[0])*0.35*np.sin(ang)
                xs_t, ys_t = integrar_trayectoria(fx_lamb, fy_lamb, x0, y0, t_max=t_max, dt=0.02)
                ax.plot(xs_t, ys_t, color='purple', lw=1, alpha=0.7)
            # Equilibrios
            for (xe, ye) in eqs_reales:
                ax.plot(xe, ye, 'ko', markersize=12, zorder=6)
                ax.annotate(f'({xe:.2f},{ye:.2f})', xy=(xe,ye), xytext=(8,8),
                            textcoords='offset points', fontsize=9)
            ax.set_xlim(xlims); ax.set_ylim(ylims)
            ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_title('Retrato de fase no lineal')
            ax.grid(alpha=0.3); ax.set_aspect('equal', adjustable='box')
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error: {e}")


# ==========================================
# MÓDULO 12: APLICACIONES (HAMILTON, VOLTERRA, COMBATE)
# ==========================================
elif categoria == "Aplicaciones (Hamilton/Volterra)":
    st.header("⚛️ Aplicaciones — Hamiltoniano, Lotka-Volterra, Combate, Romeo-Julieta")
    modo = st.selectbox("Aplicación", [
        "Test Hamiltoniano (¿conservativo?)",
        "Lotka-Volterra (presa-depredador)",
        "Modelo de competencia",
        "Combate Lanchester",
        "Romeo y Julieta"
    ])

    if modo == "Test Hamiltoniano (¿conservativo?)":
        st.info("Verifica ∂f/∂x + ∂g/∂y = 0 (criterio de Hamilton). Si se cumple, halla H tal que ẋ = ∂H/∂y, ẏ = -∂H/∂x.")
        c1, c2 = st.columns(2)
        fx_in = c1.text_input("f(x,y) = ẋ", value="y", key="funcion_input")
        fy_in = c2.text_input("g(x,y) = ẏ", value="-x - x**3", key="funcion_input_mc")
        teclado_matematico()
        if st.button("Analizar conservación"):
            try:
                x, y = sp.symbols('x y', real=True)
                fx_sym = sp.sympify(fx_in); fy_sym = sp.sympify(fy_in)
                div = sp.simplify(sp.diff(fx_sym, x) + sp.diff(fy_sym, y))
                st.latex(f"\\nabla \\cdot F = \\frac{{\\partial f}}{{\\partial x}} + \\frac{{\\partial g}}{{\\partial y}} = {sp.latex(div)}")
                if sp.simplify(div) == 0:
                    st.success("Sistema **conservativo** (divergencia nula). Integro H:")
                    # H tal que dH/dy = f, dH/dx = -g
                    H1 = sp.integrate(fx_sym, y)
                    H2 = sp.integrate(-fy_sym, x)
                    # Combinar
                    H = sp.simplify(H1 + (H2 - H1.subs(y, 0)))
                    # Validar
                    check_x = sp.simplify(sp.diff(H, y) - fx_sym)
                    check_y = sp.simplify(sp.diff(H, x) + fy_sym)
                    if check_x == 0 and check_y == 0:
                        st.latex(f"H(x,y) = {sp.latex(H)}")
                    else:
                        st.latex(f"H_1 = \\int f\\,dy = {sp.latex(H1)}")
                        st.latex(f"H_2 = \\int -g\\,dx = {sp.latex(H2)}")
                        st.warning("Integrales no combinaron directo, verificar manual.")
                else:
                    st.error("Sistema **NO conservativo**: divergencia ≠ 0 → gana/pierde energía.")
                    if sp.simplify(div).is_negative: st.write("Divergencia negativa: sistema **disipa energía** (atractor).")
                    if sp.simplify(div).is_positive: st.write("Divergencia positiva: sistema **gana energía** (fuente).")
                # Retrato de fase con curvas de nivel de H si hamiltoniano
                fx_lamb = sp.lambdify((x,y), fx_sym, 'numpy')
                fy_lamb = sp.lambdify((x,y), fy_sym, 'numpy')
                rng = 3
                fig, ax = plt.subplots(figsize=(7,7))
                dibujar_campo_2d(ax, fx_lamb, fy_lamb, (-rng, rng), (-rng, rng))
                if sp.simplify(div) == 0:
                    try:
                        H_lamb = sp.lambdify((x,y), H, 'numpy')
                        X, Y = np.meshgrid(np.linspace(-rng,rng,300), np.linspace(-rng,rng,300))
                        ax.contour(X, Y, H_lamb(X, Y), levels=15, cmap='plasma', alpha=0.7)
                    except Exception: pass
                # Trayectorias
                for k in range(8):
                    ang = 2*np.pi*k/8
                    x0, y0 = rng*0.7*np.cos(ang), rng*0.7*np.sin(ang)
                    xs_t, ys_t = integrar_trayectoria(fx_lamb, fy_lamb, x0, y0, t_max=20, dt=0.02)
                    ax.plot(xs_t, ys_t, 'purple', lw=1, alpha=0.7)
                ax.set_xlim(-rng, rng); ax.set_ylim(-rng, rng); ax.set_aspect('equal')
                ax.set_xlabel('x'); ax.set_ylabel('y'); ax.grid(alpha=0.3)
                ax.set_title('Retrato de fase con curvas de nivel de H')
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error: {e}")

    elif modo == "Lotka-Volterra (presa-depredador)":
        st.info("ẋ = (a - b·y)·x;   ẏ = (d·x - c)·y. Equilibrio interior: (c/d, a/b).")
        c1, c2, c3, c4 = st.columns(4)
        a_in = c1.text_input("a (tasa presa)", value="1.0")
        b_in = c2.text_input("b (predación)", value="0.5")
        c_in = c3.text_input("c (mortalidad depredador)", value="0.75")
        d_in = c4.text_input("d (conversión)", value="0.25")
        c5, c6, c7 = st.columns(3)
        x0_in = c5.text_input("x₀ (presa)", value="2"); y0_in = c6.text_input("y₀ (depredador)", value="1")
        t_max_in = c7.text_input("t máx", value="30")
        if st.button("Simular Lotka-Volterra"):
            try:
                a, b, c, d = parse_math(a_in), parse_math(b_in), parse_math(c_in), parse_math(d_in)
                x0, y0 = parse_math(x0_in), parse_math(y0_in)
                t_max = parse_math(t_max_in)
                fx = lambda X, Y: (a - b*Y)*X; fy = lambda X, Y: (d*X - c)*Y
                xs, ys = integrar_trayectoria(fx, fy, x0, y0, t_max=t_max, dt=0.01)
                ts = np.linspace(0, t_max, len(xs))
                eq = (c/d, a/b)
                st.success(f"Equilibrios: (0, 0) y interior **({eq[0]:.3f}, {eq[1]:.3f})**.")
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                ax1.plot(ts, xs, 'g-', label='Presa x(t)')
                ax1.plot(ts, ys, 'r-', label='Depredador y(t)')
                ax1.set_xlabel('t'); ax1.legend(); ax1.grid(alpha=0.3)
                ax1.set_title('Series temporales')
                ax2.plot(xs, ys, 'b-', lw=1.5)
                ax2.plot(eq[0], eq[1], 'k*', markersize=15, label=f'Eq ({eq[0]:.2f},{eq[1]:.2f})')
                ax2.set_xlabel('Presa x'); ax2.set_ylabel('Depredador y')
                ax2.set_title('Retrato de fase (ciclo)'); ax2.legend(); ax2.grid(alpha=0.3)
                st.pyplot(fig)
                # Invariante: H = d·x - c·ln(x) + b·y - a·ln(y)
                H_inicial = d*x0 - c*np.log(x0) + b*y0 - a*np.log(y0)
                st.info(f"Invariante Lotka-Volterra: H = d·x − c·ln(x) + b·y − a·ln(y). H(x₀, y₀) = **{H_inicial:.4f}**")
            except Exception as e:
                st.error(f"Error: {e}")

    elif modo == "Modelo de competencia":
        st.info("ẋ = x·(α - β·x - γ·y);   ẏ = y·(δ - ε·y - ζ·x). Coexistencia, exclusión, etc.")
        c1, c2, c3 = st.columns(3)
        al_in = c1.text_input("α", value="14"); be_in = c2.text_input("β", value="0.5"); ga_in = c3.text_input("γ", value="1")
        c4, c5, c6 = st.columns(3)
        de_in = c4.text_input("δ", value="16"); ep_in = c5.text_input("ε", value="0.5"); ze_in = c6.text_input("ζ", value="1")
        c7, c8 = st.columns(2)
        x0_in = c7.text_input("x₀", value="5"); y0_in = c8.text_input("y₀", value="5")
        if st.button("Simular competencia"):
            try:
                al,be,ga,de,ep,ze = [parse_math(v) for v in (al_in,be_in,ga_in,de_in,ep_in,ze_in)]
                x0, y0 = parse_math(x0_in), parse_math(y0_in)
                fx = lambda X, Y: X*(al - be*X - ga*Y)
                fy = lambda X, Y: Y*(de - ep*Y - ze*X)
                # Equilibrios
                x_sp, y_sp = sp.symbols('x y', real=True, positive=True)
                fx_sym = x_sp*(al - be*x_sp - ga*y_sp); fy_sym = y_sp*(de - ep*y_sp - ze*x_sp)
                eqs = sp.solve([fx_sym, fy_sym], [x_sp, y_sp], dict=True)
                st.write("**Equilibrios:**")
                for e in eqs: st.write(f"({e.get(x_sp, 0)}, {e.get(y_sp, 0)})")
                xs, ys = integrar_trayectoria(fx, fy, x0, y0, t_max=30, dt=0.02)
                fig, ax = plt.subplots(figsize=(8,8))
                rng = max(al/be, de/ep)*1.2
                dibujar_campo_2d(ax, fx, fy, (0, rng), (0, rng))
                ax.plot(xs, ys, 'b-', lw=2, label='Trayectoria')
                for e in eqs:
                    try: ax.plot(float(e.get(x_sp,0)), float(e.get(y_sp,0)), 'ko', markersize=10)
                    except: pass
                ax.set_xlim(0,rng); ax.set_ylim(0,rng); ax.set_xlabel('x'); ax.set_ylabel('y')
                ax.legend(); ax.grid(alpha=0.3); ax.set_aspect('equal')
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error: {e}")

    elif modo == "Combate Lanchester":
        st.info("ẋ = -a·y;   ẏ = -b·x.  Invariante: b·x² - a·y² = cte (hipérbolas en plano x-y).")
        c1, c2 = st.columns(2)
        a_in = c1.text_input("a", value="0.1"); b_in = c2.text_input("b", value="0.2")
        c3, c4, c5 = st.columns(3)
        x0_in = c3.text_input("Fuerza x₀", value="100"); y0_in = c4.text_input("Fuerza y₀", value="60")
        t_max_in = c5.text_input("t máx", value="20")
        if st.button("Simular Lanchester"):
            try:
                a, b = parse_math(a_in), parse_math(b_in)
                x0, y0 = parse_math(x0_in), parse_math(y0_in)
                t_max = parse_math(t_max_in)
                fx = lambda X, Y: -a*Y; fy = lambda X, Y: -b*X
                xs, ys = integrar_trayectoria(fx, fy, x0, y0, t_max=t_max, dt=0.01)
                ts = np.linspace(0, t_max, len(xs))
                invariante_0 = b*x0**2 - a*y0**2
                gana = "x" if invariante_0 > 0 else "y" if invariante_0 < 0 else "empate"
                st.success(f"Invariante b·x² − a·y² = {invariante_0:.2f}.  **Gana: {gana}** (regla cuadrática de Lanchester).")
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                ax1.plot(ts, xs, 'b-', label='x(t)'); ax1.plot(ts, ys, 'r-', label='y(t)')
                ax1.axhline(0, color='k', lw=0.5); ax1.legend(); ax1.grid(alpha=0.3)
                ax2.plot(xs, ys, 'g-', lw=2); ax2.plot(x0, y0, 'go', markersize=10, label='Inicio')
                ax2.set_xlabel('x'); ax2.set_ylabel('y'); ax2.set_title('Retrato de fase x-y')
                ax2.legend(); ax2.grid(alpha=0.3)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error: {e}")

    elif modo == "Romeo y Julieta":
        st.info("Ṙ = a·R + b·J;   J̇ = b·R + a·J.  a=cautela, b=responsividad.")
        c1, c2 = st.columns(2)
        a_in = c1.text_input("a (cautela)", value="-0.3")
        b_in = c2.text_input("b (responsividad)", value="0.5")
        c3, c4, c5 = st.columns(3)
        R0 = c3.text_input("R(0)", value="1"); J0 = c4.text_input("J(0)", value="0.5")
        t_max_in = c5.text_input("t máx", value="20")
        if st.button("Simular romance"):
            try:
                a, b = parse_math(a_in), parse_math(b_in)
                R0v, J0v = parse_math(R0), parse_math(J0); t_max = parse_math(t_max_in)
                A = np.array([[a, b], [b, a]], dtype=float)
                tipo, evals, _, _ = clasificar_2d_lineal(A)
                st.success(f"Sistema lineal 2D: **{tipo}**. Autovalores: {evals}")
                fx = lambda X, Y: a*X + b*Y; fy = lambda X, Y: b*X + a*Y
                xs, ys = integrar_trayectoria(fx, fy, R0v, J0v, t_max=t_max, dt=0.01)
                ts = np.linspace(0, t_max, len(xs))
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                ax1.plot(ts, xs, label='R(t)'); ax1.plot(ts, ys, label='J(t)')
                ax1.legend(); ax1.grid(alpha=0.3); ax1.set_xlabel('t')
                ax2.plot(xs, ys, 'b-'); ax2.plot(R0v, J0v, 'go', label='Inicio')
                ax2.set_xlabel('R'); ax2.set_ylabel('J'); ax2.grid(alpha=0.3); ax2.legend()
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error: {e}")
