"""
Laboratorio de Modelado y Simulacion - Metodos Numericos
Simulador interactivo con Tkinter y Matplotlib
Cubre Clases 1-7 del programa (hasta 1er Parcial)

Requisitos: pip install matplotlib numpy sympy
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sympy import sympify, lambdify, Symbol

x_sym = Symbol('x')
y_sym = Symbol('y')


def parse_fx(expr_str):
    return lambdify(x_sym, sympify(expr_str), modules=['numpy'])


def parse_fxy(expr_str):
    return lambdify((x_sym, y_sym), sympify(expr_str), modules=['numpy'])


def fmt(val, dec=8):
    if isinstance(val, (int, float, np.floating)):
        if not np.isfinite(val):
            return str(val)
        return f"{val:.{dec}f}"
    return str(val)


class SimuladorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Laboratorio de Metodos Numericos - Parcial 1")
        self.root.geometry("1250x780")
        self.root.minsize(1050, 650)

        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Nav.TButton', padding=(10, 6), font=('Segoe UI', 9))
        style.configure('Run.TButton', padding=(14, 6), font=('Segoe UI', 10, 'bold'))
        style.configure('Header.TLabel', font=('Segoe UI', 15, 'bold'))
        style.configure('Sub.TLabel', font=('Segoe UI', 9), foreground='#666')

        self.nav_frame = ttk.Frame(root, width=230)
        self.nav_frame.pack(side=tk.LEFT, fill=tk.Y)
        self.nav_frame.pack_propagate(False)

        self.content_frame = ttk.Frame(root)
        self.content_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._build_nav()
        self.panels = {}
        self._build_all_panels()
        self.show_panel('biseccion')

    def _build_nav(self):
        ttk.Label(self.nav_frame, text="Metodos Numericos", font=('Segoe UI', 11, 'bold'),
                  foreground='#1a73e8').pack(pady=(10, 0), padx=8, anchor='w')
        ttk.Label(self.nav_frame, text="Parcial 1", font=('Segoe UI', 8), foreground='#888').pack(padx=8, anchor='w')
        ttk.Separator(self.nav_frame, orient='horizontal').pack(fill=tk.X, pady=6)

        sections = [
            ("C1: Raices", ['biseccion', 'punto_fijo']),
            ("C2: Newton / Aitken", ['newton', 'aitken']),
            ("C3: Interp / Derivacion", ['lagrange', 'newton_interp', 'derivacion']),
            ("C5: Integracion", ['rectangulos', 'trapecio', 'simpson']),
            ("C6: Montecarlo", ['montecarlo']),
            ("C7: EDOs / EDPs", ['euler', 'euler_mod', 'taylor', 'runge_kutta', 'edp']),
            ("Sist. Lineales", ['gauss', 'gauss_seidel']),
        ]
        names = {
            'biseccion': 'Biseccion', 'punto_fijo': 'Punto Fijo', 'newton': 'Newton-Raphson',
            'aitken': 'Aitken (Delta^2)', 'lagrange': 'Lagrange', 'newton_interp': 'Newton Dif.Div.',
            'derivacion': 'Derivacion Num.', 'rectangulos': 'Rectangulos', 'trapecio': 'Trapecio',
            'simpson': 'Simpson 1/3', 'montecarlo': 'Montecarlo 1D/2D', 'euler': 'Euler',
            'euler_mod': 'Euler Mod. (Heun)', 'taylor': 'Taylor O(2)', 'runge_kutta': 'RK4',
            'edp': 'EDP Dif. Finitas', 'gauss': 'Gauss', 'gauss_seidel': 'Gauss-Seidel',
        }
        self.nav_buttons = {}
        for section, methods in sections:
            ttk.Label(self.nav_frame, text=section, font=('Segoe UI', 8, 'bold'),
                      foreground='#888').pack(padx=8, pady=(8, 2), anchor='w')
            for m in methods:
                btn = ttk.Button(self.nav_frame, text=names[m], style='Nav.TButton',
                                 command=lambda mid=m: self.show_panel(mid))
                btn.pack(fill=tk.X, padx=4, pady=1)
                self.nav_buttons[m] = btn

    def show_panel(self, pid):
        for f in self.panels.values():
            f.pack_forget()
        if pid in self.panels:
            self.panels[pid].pack(fill=tk.BOTH, expand=True)

    def _make_panel(self, pid):
        f = ttk.Frame(self.content_frame)
        self.panels[pid] = f
        return f

    def _header(self, parent, title, sub):
        ttk.Label(parent, text=title, style='Header.TLabel').pack(anchor='w', padx=14, pady=(10, 0))
        ttk.Label(parent, text=sub, style='Sub.TLabel').pack(anchor='w', padx=14, pady=(0, 8))

    def _inputs(self, parent, fields):
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, padx=14, pady=2)
        entries = {}
        for label, default, width in fields:
            frm = ttk.Frame(row)
            frm.pack(side=tk.LEFT, padx=(0, 10))
            ttk.Label(frm, text=label, font=('Segoe UI', 8)).pack(anchor='w')
            e = ttk.Entry(frm, width=width, font=('Consolas', 10))
            e.insert(0, str(default))
            e.pack()
            entries[label] = e
        return entries

    def _buttons(self, parent, run_cmd, clear_cmd=None):
        frm = ttk.Frame(parent)
        frm.pack(fill=tk.X, padx=14, pady=6)
        ttk.Button(frm, text="Ejecutar", style='Run.TButton', command=run_cmd).pack(side=tk.LEFT, padx=(0, 4))
        if clear_cmd:
            ttk.Button(frm, text="Limpiar", command=clear_cmd).pack(side=tk.LEFT)

    def _result(self, parent):
        r = tk.Text(parent, height=3, font=('Consolas', 10), wrap=tk.WORD, bg='#e6f4ea', relief=tk.FLAT, padx=6, pady=4)
        r.pack(fill=tk.X, padx=14, pady=(0, 4))
        return r

    def _table_chart(self, parent, chart_height=4):
        pw = ttk.PanedWindow(parent, orient=tk.HORIZONTAL)
        pw.pack(fill=tk.BOTH, expand=True, padx=14, pady=(0, 8))
        tf = ttk.Frame(pw)
        pw.add(tf, weight=1)
        tree = ttk.Treeview(tf, show='headings', height=14)
        vsb = ttk.Scrollbar(tf, orient='vertical', command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        cf = ttk.Frame(pw)
        pw.add(cf, weight=1)
        fig = Figure(figsize=(5, chart_height), dpi=100)
        fig.set_facecolor('#ffffff')
        ax = fig.add_subplot(111)
        canvas = FigureCanvasTkAgg(fig, master=cf)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        return tree, fig, ax, canvas

    def _fill_table(self, tree, cols, rows):
        tree.delete(*tree.get_children())
        tree['columns'] = cols
        for c in cols:
            tree.heading(c, text=c)
            tree.column(c, width=85, anchor='center')
        for r in rows:
            tree.insert('', tk.END, values=r)

    def _set_result(self, widget, text):
        widget.delete('1.0', tk.END)
        widget.insert('1.0', text)

    # ==================== BUILD ALL ====================
    def _build_all_panels(self):
        self._build_biseccion()
        self._build_punto_fijo()
        self._build_newton()
        self._build_aitken()
        self._build_lagrange()
        self._build_newton_interp()
        self._build_derivacion()
        self._build_rectangulos()
        self._build_trapecio()
        self._build_simpson()
        self._build_montecarlo()
        self._build_euler()
        self._build_euler_mod()
        self._build_taylor()
        self._build_runge_kutta()
        self._build_edp()
        self._build_gauss()
        self._build_gauss_seidel()

    # ==================== BISECCION ====================
    def _build_biseccion(self):
        p = self._make_panel('biseccion')
        self._header(p, "Biseccion", "Busqueda binaria de raices - Clase 1")
        self.bis_e = self._inputs(p, [("f(x)", "x**3 - x - 2", 25), ("a", "1", 7), ("b", "2", 7), ("Tol", "0.0001", 9), ("Max", "50", 5)])
        self.bis_r = self._result(p)
        self._buttons(p, self._run_biseccion)
        self.bis_t, self.bis_f, self.bis_a, self.bis_c = self._table_chart(p)

    def _run_biseccion(self):
        try:
            f = parse_fx(self.bis_e["f(x)"].get())
            a, b = float(self.bis_e["a"].get()), float(self.bis_e["b"].get())
            tol, mx = float(self.bis_e["Tol"].get()), int(self.bis_e["Max"].get())
        except Exception as e:
            messagebox.showerror("Error", str(e)); return
        if f(a) * f(b) > 0:
            messagebox.showerror("Error", "f(a)*f(b) debe ser < 0"); return
        oA, oB, rows = a, b, []
        for i in range(1, mx + 1):
            c = (a + b) / 2; fc = f(c); err = abs(b - a) / 2
            rows.append((i, fmt(a), fmt(b), fmt(c), fmt(f(a)), fmt(fc), fmt(err)))
            if abs(fc) < 1e-15 or err < tol: break
            if f(a) * fc < 0: b = c
            else: a = c
        lc = float(rows[-1][3])
        self._set_result(self.bis_r, f"Raiz: {fmt(lc, 10)} | f(x)={fmt(f(lc))} | Iter:{rows[-1][0]} | Err:{rows[-1][6]}")
        self._fill_table(self.bis_t, ('n', 'a', 'b', 'c', 'f(a)', 'f(c)', 'Error'), rows)
        self.bis_a.clear()
        xs = np.linspace(oA - 0.5, oB + 0.5, 300)
        self.bis_a.plot(xs, f(xs), 'b-', lw=2, label='f(x)')
        self.bis_a.axhline(0, color='gray', lw=0.5)
        cs = [float(r[3]) for r in rows]
        self.bis_a.plot(cs, [f(c) for c in cs], 'ro', ms=4, label='Iter')
        self.bis_a.axvline(lc, color='green', ls='--', alpha=0.7)
        self.bis_a.legend(fontsize=8); self.bis_a.grid(True, alpha=0.3); self.bis_c.draw()

    # ==================== PUNTO FIJO ====================
    def _build_punto_fijo(self):
        p = self._make_panel('punto_fijo')
        self._header(p, "Punto Fijo", "Contracciones x = g(x) - Clase 1")
        self.pf_e = self._inputs(p, [("g(x)", "(x + 2)**(1/3)", 25), ("x0", "1.5", 7), ("Tol", "0.0001", 9), ("Max", "50", 5)])
        self.pf_r = self._result(p)
        self._buttons(p, self._run_punto_fijo)
        self.pf_t, self.pf_f, self.pf_a, self.pf_c = self._table_chart(p)

    def _run_punto_fijo(self):
        try:
            g = parse_fx(self.pf_e["g(x)"].get())
            x = float(self.pf_e["x0"].get()); tol = float(self.pf_e["Tol"].get()); mx = int(self.pf_e["Max"].get())
        except Exception as e:
            messagebox.showerror("Error", str(e)); return
        rows = []
        for i in range(1, mx + 1):
            xn = float(g(x)); err = abs(xn - x)
            rows.append((i, fmt(x), fmt(xn), fmt(err)))
            if not np.isfinite(xn): messagebox.showerror("Error", "Diverge"); return
            x = xn
            if err < tol: break
        self._set_result(self.pf_r, f"Punto fijo: {fmt(x, 10)} | Iter:{rows[-1][0]} | Err:{rows[-1][3]}")
        self._fill_table(self.pf_t, ('n', 'x_n', 'g(x_n)', 'Error'), rows)
        self.pf_a.clear()
        aX = [float(r[1]) for r in rows] + [float(r[2]) for r in rows]
        mn, mx2 = min(aX) - 0.5, max(aX) + 0.5
        xs = np.linspace(mn, mx2, 300)
        self.pf_a.plot(xs, g(xs), 'b-', lw=2, label='g(x)')
        self.pf_a.plot(xs, xs, '--', color='gray', label='y=x')
        self.pf_a.plot([float(r[1]) for r in rows], [float(r[2]) for r in rows], 'ro', ms=4, label='Iter')
        self.pf_a.legend(fontsize=8); self.pf_a.grid(True, alpha=0.3); self.pf_c.draw()

    # ==================== NEWTON ====================
    def _build_newton(self):
        p = self._make_panel('newton')
        self._header(p, "Newton-Raphson", "Convergencia cuadratica - Clase 2")
        self.nr_e = self._inputs(p, [("f(x)", "x**3 - x - 2", 20), ("f'(x)", "3*x**2 - 1", 20), ("x0", "1.5", 7), ("Tol", "0.0001", 9), ("Max", "50", 5)])
        self.nr_r = self._result(p)
        self._buttons(p, self._run_newton)
        self.nr_t, self.nr_f, self.nr_a, self.nr_c = self._table_chart(p)

    def _run_newton(self):
        try:
            f = parse_fx(self.nr_e["f(x)"].get()); df = parse_fx(self.nr_e["f'(x)"].get())
            x = float(self.nr_e["x0"].get()); tol = float(self.nr_e["Tol"].get()); mx = int(self.nr_e["Max"].get())
        except Exception as e:
            messagebox.showerror("Error", str(e)); return
        rows = []
        for i in range(1, mx + 1):
            fv, dv = f(x), df(x)
            if abs(dv) < 1e-15: messagebox.showerror("Error", "f'~0"); return
            xn = x - fv / dv; err = abs(xn - x)
            rows.append((i, fmt(x), fmt(fv), fmt(dv), fmt(xn), fmt(err)))
            x = xn
            if err < tol: break
        self._set_result(self.nr_r, f"Raiz: {fmt(x, 10)} | f(x)={fmt(f(x))} | Iter:{rows[-1][0]}")
        self._fill_table(self.nr_t, ('n', 'x_n', 'f(x_n)', "f'(x_n)", 'x_{n+1}', 'Error'), rows)
        self.nr_a.clear()
        rng = max(abs(x - float(self.nr_e["x0"].get())) * 2, 2)
        xs = np.linspace(x - rng, x + rng, 300)
        self.nr_a.plot(xs, f(xs), 'b-', lw=2, label='f(x)')
        self.nr_a.axhline(0, color='gray', lw=0.5)
        self.nr_a.plot([float(r[1]) for r in rows], [float(r[2]) for r in rows], 'ro', ms=4, label='Iter')
        self.nr_a.legend(fontsize=8); self.nr_a.grid(True, alpha=0.3); self.nr_c.draw()

    # ==================== AITKEN ====================
    def _build_aitken(self):
        p = self._make_panel('aitken')
        self._header(p, "Aceleracion de Aitken", "Acelera convergencia lineal - Clase 2")
        self.ait_e = self._inputs(p, [("g(x)", "(x + 2)**(1/3)", 25), ("x0", "1.5", 7), ("Tol", "0.0001", 9), ("Max", "50", 5)])
        self.ait_r = self._result(p)
        self._buttons(p, self._run_aitken)
        self.ait_t, self.ait_f, self.ait_a, self.ait_c = self._table_chart(p)

    def _run_aitken(self):
        try:
            g = parse_fx(self.ait_e["g(x)"].get())
            x = float(self.ait_e["x0"].get()); tol = float(self.ait_e["Tol"].get()); mx = int(self.ait_e["Max"].get())
        except Exception as e:
            messagebox.showerror("Error", str(e)); return
        seq = [x]
        for _ in range(mx + 2):
            x = float(g(x))
            if not np.isfinite(x): break
            seq.append(x)
        aitken = []
        for n in range(len(seq) - 2):
            s0, s1, s2 = seq[n], seq[n + 1], seq[n + 2]
            d = s2 - 2 * s1 + s0
            if abs(d) < 1e-15: break
            aitken.append(s0 - (s1 - s0) ** 2 / d)
        rows, pfE, aitE = [], [], []
        lim = min(len(seq), len(aitken) + 2, mx)
        for n in range(lim):
            pf = seq[n]; ai = aitken[n] if n < len(aitken) else None
            ep = abs(seq[n] - seq[n - 1]) if n > 0 else None
            ea = abs(aitken[n] - aitken[n - 1]) if n > 0 and n < len(aitken) else None
            if ep is not None: pfE.append(ep)
            if ea is not None: aitE.append(ea)
            rows.append((n, fmt(pf), fmt(ai) if ai else '-', fmt(ep) if ep else '-', fmt(ea) if ea else '-'))
            if ea is not None and ea < tol: break
        self._set_result(self.ait_r, f"P.Fijo: {fmt(seq[min(len(seq)-1,lim)])} | Aitken: {fmt(aitken[-1] if aitken else 0, 10)}")
        self._fill_table(self.ait_t, ('n', 'x_n PF', 'x_n* Aitken', 'Err PF', 'Err Aitken'), rows)
        self.ait_a.clear()
        if pfE: self.ait_a.plot(range(1, len(pfE) + 1), pfE, 'r-o', ms=3, label='Err P.Fijo')
        if aitE: self.ait_a.plot(range(1, len(aitE) + 1), aitE, 'g-o', ms=3, label='Err Aitken')
        self.ait_a.legend(fontsize=8); self.ait_a.set_title("Convergencia", fontsize=9); self.ait_a.grid(True, alpha=0.3); self.ait_c.draw()

    # ==================== LAGRANGE ====================
    def _build_lagrange(self):
        p = self._make_panel('lagrange')
        self._header(p, "Interpolacion de Lagrange", "Clase 3")
        self.lag_e = self._inputs(p, [("x vals", "0, 1, 2, 3", 20), ("y vals", "1, 2.718, 7.389, 20.086", 20), ("Eval x=", "1.5", 7)])
        self.lag_r = self._result(p)
        self._buttons(p, self._run_lagrange)
        self.lag_t, self.lag_f, self.lag_a, self.lag_c = self._table_chart(p)

    def _run_lagrange(self):
        xv = [float(v) for v in self.lag_e["x vals"].get().split(',')]
        yv = [float(v) for v in self.lag_e["y vals"].get().split(',')]
        ex = float(self.lag_e["Eval x="].get()); n = len(xv)
        def lag(xp):
            s = 0
            for i in range(n):
                l = 1
                for j in range(n):
                    if i != j: l *= (xp - xv[j]) / (xv[i] - xv[j])
                s += yv[i] * l
            return s
        res = lag(ex)
        self._set_result(self.lag_r, f"P({ex}) = {fmt(res, 10)}")
        self._fill_table(self.lag_t, ('i', 'x_i', 'y_i'), [(i, fmt(xv[i], 4), fmt(yv[i], 4)) for i in range(n)])
        self.lag_a.clear()
        xs = np.linspace(min(xv) - 0.5, max(xv) + 0.5, 300)
        self.lag_a.plot(xs, [lag(xi) for xi in xs], 'b-', lw=2, label='P(x)')
        self.lag_a.plot(xv, yv, 'ro', ms=6, label='Puntos')
        self.lag_a.plot(ex, res, 'gs', ms=8, label=f'P({ex})')
        self.lag_a.legend(fontsize=8); self.lag_a.grid(True, alpha=0.3); self.lag_c.draw()

    # ==================== NEWTON INTERP ====================
    def _build_newton_interp(self):
        p = self._make_panel('newton_interp')
        self._header(p, "Newton Dif. Divididas", "Clase 3")
        self.ni_e = self._inputs(p, [("x vals", "1, 2, 4, 7", 20), ("y vals", "1, 4, 16, 49", 20), ("Eval x=", "3", 7)])
        self.ni_r = self._result(p)
        self._buttons(p, self._run_newton_interp)
        self.ni_t, self.ni_f, self.ni_a, self.ni_c = self._table_chart(p)

    def _run_newton_interp(self):
        xv = [float(v) for v in self.ni_e["x vals"].get().split(',')]
        yv = [float(v) for v in self.ni_e["y vals"].get().split(',')]
        ex = float(self.ni_e["Eval x="].get()); n = len(xv)
        dd = [[0.0] * n for _ in range(n)]
        for i in range(n): dd[i][0] = yv[i]
        for j in range(1, n):
            for i in range(n - j): dd[i][j] = (dd[i + 1][j - 1] - dd[i][j - 1]) / (xv[i + j] - xv[i])
        def np_(xp):
            r, p = dd[0][0], 1.0
            for j in range(1, n): p *= (xp - xv[j - 1]); r += dd[0][j] * p
            return r
        res = np_(ex)
        self._set_result(self.ni_r, f"P({ex}) = {fmt(res, 10)}")
        cols = ['x_i', 'f[x_i]'] + [f'Ord{j}' for j in range(1, n)]
        rows = []
        for i in range(n):
            row = [fmt(xv[i], 4)] + [fmt(dd[i][j], 6) for j in range(n - i)] + [''] * i
            rows.append(tuple(row))
        self._fill_table(self.ni_t, tuple(cols), rows)
        self.ni_a.clear()
        xs = np.linspace(min(xv) - 1, max(xv) + 1, 300)
        self.ni_a.plot(xs, [np_(xi) for xi in xs], 'b-', lw=2, label='P(x)')
        self.ni_a.plot(xv, yv, 'ro', ms=6, label='Puntos')
        self.ni_a.plot(ex, res, 'gs', ms=8)
        self.ni_a.legend(fontsize=8); self.ni_a.grid(True, alpha=0.3); self.ni_c.draw()

    # ==================== DERIVACION NUMERICA ====================
    def _build_derivacion(self):
        p = self._make_panel('derivacion')
        self._header(p, "Derivacion Numerica", "Forward/Backward/Central + paso optimo - Clase 3")
        self.der_e = self._inputs(p, [("f(x)", "sin(x)", 20), ("x0", "1", 7), ("h", "0.1", 7), ("f'(x) exacta", "cos(x)", 15)])
        self.der_r = self._result(p)
        self._buttons(p, self._run_derivacion)
        self.der_t, self.der_f, self.der_a, self.der_c = self._table_chart(p)

    def _run_derivacion(self):
        try:
            f = parse_fx(self.der_e["f(x)"].get())
            x0 = float(self.der_e["x0"].get()); h0 = float(self.der_e["h"].get())
            exact = parse_fx(self.der_e["f'(x) exacta"].get()) if self.der_e["f'(x) exacta"].get() else None
        except Exception as e:
            messagebox.showerror("Error", str(e)); return
        fwd = (f(x0 + h0) - f(x0)) / h0
        bwd = (f(x0) - f(x0 - h0)) / h0
        ctr = (f(x0 + h0) - f(x0 - h0)) / (2 * h0)
        d2 = (f(x0 + h0) - 2 * f(x0) + f(x0 - h0)) / (h0 ** 2)
        ev = float(exact(x0)) if exact else None
        txt = f"Fwd: {fmt(fwd,10)} | Bwd: {fmt(bwd,10)} | Ctr: {fmt(ctr,10)} | f'': {fmt(d2,10)}"
        if ev is not None: txt += f"\nExacto: {fmt(ev,10)} | Err ctr: {fmt(abs(ctr-ev))}"
        self._set_result(self.der_r, txt)
        rows, hs, eF, eB, eC = [], [], [], [], []
        for k in range(12):
            h = 1 / 2 ** k; hs.append(h)
            fw = (f(x0 + h) - f(x0)) / h; bw = (f(x0) - f(x0 - h)) / h; ct = (f(x0 + h) - f(x0 - h)) / (2 * h)
            dd2 = (f(x0 + h) - 2 * f(x0) + f(x0 - h)) / (h ** 2)
            ef = abs(fw - ev) if ev else 0; eb = abs(bw - ev) if ev else 0; ec = abs(ct - ev) if ev else 0
            eF.append(ef); eB.append(eb); eC.append(ec)
            row = [fmt(h, 6), fmt(fw), fmt(bw), fmt(ct), fmt(dd2)]
            if ev: row += [fmt(ef), fmt(eb), fmt(ec)]
            rows.append(tuple(row))
        cols = ['h', 'Fwd', 'Bwd', 'Central', "f''"]
        if ev: cols += ['Err F', 'Err B', 'Err C']
        self._fill_table(self.der_t, tuple(cols), rows)
        self.der_a.clear()
        if ev:
            self.der_a.loglog(hs, eF, 'r-o', ms=3, label='Err Fwd O(h)')
            self.der_a.loglog(hs, eB, 'orange', marker='o', ms=3, label='Err Bwd O(h)')
            self.der_a.loglog(hs, eC, 'g-o', ms=3, label='Err Ctr O(h^2)')
            self.der_a.legend(fontsize=7); self.der_a.set_xlabel('h'); self.der_a.set_title('Error vs h (log-log)', fontsize=9)
        self.der_a.grid(True, alpha=0.3); self.der_c.draw()

    # ==================== RECTANGULOS ====================
    def _build_rectangulos(self):
        p = self._make_panel('rectangulos')
        self._header(p, "Regla de Rectangulos", "Izq/Der/Medio - Clase 5")
        self.rect_e = self._inputs(p, [("f(x)", "sin(x)", 20), ("a", "0", 7), ("b", "3.14159", 7), ("n", "10", 5), ("Tipo(L/R/M)", "M", 4)])
        self.rect_r = self._result(p)
        self._buttons(p, self._run_rectangulos)
        self.rect_t, self.rect_f, self.rect_a, self.rect_c = self._table_chart(p)

    def _run_rectangulos(self):
        try:
            f = parse_fx(self.rect_e["f(x)"].get())
            a, b = float(self.rect_e["a"].get()), float(self.rect_e["b"].get())
            n = int(self.rect_e["n"].get()); tipo = self.rect_e["Tipo(L/R/M)"].get().upper()
        except Exception as e:
            messagebox.showerror("Error", str(e)); return
        h = (b - a) / n; xs, ys = [], []
        for i in range(n):
            if tipo == 'L': xp = a + i * h
            elif tipo == 'R': xp = a + (i + 1) * h
            else: xp = a + (i + 0.5) * h
            xs.append(xp); ys.append(float(f(xp)))
        integral = h * sum(ys)
        nm = {'L': 'Izquierda', 'R': 'Derecha', 'M': 'Punto Medio'}
        self._set_result(self.rect_r, f"Integral ({nm.get(tipo, tipo)}) = {fmt(integral, 10)} | h={fmt(h)} | n={n}")
        self._fill_table(self.rect_t, ('i', 'x_i', 'f(x_i)', 'Area'), [(i, fmt(xs[i]), fmt(ys[i]), fmt(h * ys[i])) for i in range(n)])
        self.rect_a.clear()
        xf = np.linspace(a, b, 300)
        self.rect_a.plot(xf, f(xf), 'b-', lw=2, label='f(x)')
        for i in range(n):
            x0, x1 = a + i * h, a + (i + 1) * h
            self.rect_a.fill([x0, x1, x1, x0], [0, 0, ys[i], ys[i]], alpha=0.15, color='blue')
        self.rect_a.plot(xs, ys, 'ro', ms=3)
        self.rect_a.legend(fontsize=8); self.rect_a.grid(True, alpha=0.3); self.rect_c.draw()

    # ==================== TRAPECIO ====================
    def _build_trapecio(self):
        p = self._make_panel('trapecio')
        self._header(p, "Trapecio", "Clase 5")
        self.trap_e = self._inputs(p, [("f(x)", "sin(x)", 20), ("a", "0", 7), ("b", "3.14159", 7), ("n", "10", 5)])
        self.trap_r = self._result(p)
        self._buttons(p, self._run_trapecio)
        self.trap_t, self.trap_f, self.trap_a, self.trap_c = self._table_chart(p)

    def _run_trapecio(self):
        f = parse_fx(self.trap_e["f(x)"].get())
        a, b, n = float(self.trap_e["a"].get()), float(self.trap_e["b"].get()), int(self.trap_e["n"].get())
        h = (b - a) / n; xs = [a + i * h for i in range(n + 1)]; ys = [float(f(x)) for x in xs]
        s = ys[0] + ys[-1] + sum(2 * ys[i] for i in range(1, n))
        integral = (h / 2) * s
        self._set_result(self.trap_r, f"Integral = {fmt(integral, 10)} | h={fmt(h)} | n={n}")
        self._fill_table(self.trap_t, ('i', 'x_i', 'f(x_i)', 'Coef', 'Contrib'),
                         [(i, fmt(xs[i]), fmt(ys[i]), 1 if (i == 0 or i == n) else 2, fmt((1 if (i == 0 or i == n) else 2) * ys[i])) for i in range(n + 1)])
        self.trap_a.clear()
        xf = np.linspace(a, b, 300)
        self.trap_a.plot(xf, f(xf), 'b-', lw=2, label='f(x)')
        for i in range(n):
            self.trap_a.fill([xs[i], xs[i], xs[i + 1], xs[i + 1]], [0, ys[i], ys[i + 1], 0], alpha=0.12, color='blue')
        self.trap_a.plot(xs, ys, 'ro', ms=3)
        self.trap_a.legend(fontsize=8); self.trap_a.grid(True, alpha=0.3); self.trap_c.draw()

    # ==================== SIMPSON ====================
    def _build_simpson(self):
        p = self._make_panel('simpson')
        self._header(p, "Simpson 1/3", "Clase 5")
        self.simp_e = self._inputs(p, [("f(x)", "exp(-x**2)", 20), ("a", "0", 7), ("b", "1", 7), ("n(par)", "10", 5)])
        self.simp_r = self._result(p)
        self._buttons(p, self._run_simpson)
        self.simp_t, self.simp_f, self.simp_a, self.simp_c = self._table_chart(p)

    def _run_simpson(self):
        f = parse_fx(self.simp_e["f(x)"].get())
        a, b = float(self.simp_e["a"].get()), float(self.simp_e["b"].get())
        n = int(self.simp_e["n(par)"].get())
        if n % 2: n += 1
        h = (b - a) / n; xs = [a + i * h for i in range(n + 1)]; ys = [float(f(x)) for x in xs]
        s = ys[0] + ys[-1] + sum((4 if i % 2 else 2) * ys[i] for i in range(1, n))
        integral = (h / 3) * s
        self._set_result(self.simp_r, f"Integral = {fmt(integral, 10)} | h={fmt(h)} | n={n}")
        self._fill_table(self.simp_t, ('i', 'x_i', 'f(x_i)', 'Coef'),
                         [(i, fmt(xs[i]), fmt(ys[i]), 1 if (i == 0 or i == n) else (4 if i % 2 else 2)) for i in range(n + 1)])
        self.simp_a.clear()
        xf = np.linspace(a, b, 300)
        self.simp_a.plot(xf, f(xf), 'b-', lw=2, label='f(x)')
        self.simp_a.fill_between(xf, 0, f(xf), alpha=0.12, color='blue')
        self.simp_a.plot(xs, ys, 'ro', ms=4)
        self.simp_a.legend(fontsize=8); self.simp_a.grid(True, alpha=0.3); self.simp_c.draw()

    # ==================== MONTECARLO ====================
    def _build_montecarlo(self):
        p = self._make_panel('montecarlo')
        self._header(p, "Montecarlo", "1D + 2D, IC 95%, varianza - Clase 6")
        self.mc_e = self._inputs(p, [("f(x)", "sin(x)", 18), ("a", "0", 7), ("b", "3.14159", 7), ("N", "10000", 8)])
        self.mc_r = self._result(p)
        ttk.Label(p, text="Integral 2D:", font=('Segoe UI', 9, 'bold')).pack(anchor='w', padx=14, pady=(4, 0))
        self.mc2_e = self._inputs(p, [("f(x,y)", "x**2+y**2", 18), ("ax", "0", 5), ("bx", "1", 5), ("ay", "0", 5), ("by", "1", 5), ("N2", "50000", 7)])
        self.mc2_r = self._result(p)
        self._buttons(p, self._run_mc)
        self.mc_t, self.mc_f, self.mc_a, self.mc_c = self._table_chart(p)

    def _run_mc(self):
        # 1D
        f = parse_fx(self.mc_e["f(x)"].get())
        a, b, N = float(self.mc_e["a"].get()), float(self.mc_e["b"].get()), int(self.mc_e["N"].get())
        xs = np.random.uniform(a, b, N); vals = np.array([float(f(x)) for x in xs])
        mean = np.mean(vals); var = np.var(vals, ddof=1); se = np.sqrt(var / N)
        integral = (b - a) * mean; err = (b - a) * se; ci = 1.96 * err
        self._set_result(self.mc_r, f"MC 1D = {fmt(integral, 10)}\nVar={fmt(var)} | IC95%: [{fmt(integral-ci)}, {fmt(integral+ci)}]")
        # convergence
        cum = np.cumsum(vals); ns = np.arange(1, N + 1); conv = (b - a) * cum / ns
        step = max(1, N // 200)
        self.mc_a.clear()
        self.mc_a.plot(ns[::step], conv[::step], 'b-', lw=1, label='MC 1D')
        self.mc_a.set_xlabel('N'); self.mc_a.set_title('Convergencia', fontsize=9)
        self.mc_a.legend(fontsize=8); self.mc_a.grid(True, alpha=0.3); self.mc_c.draw()
        # 2D
        try:
            f2 = parse_fxy(self.mc2_e["f(x,y)"].get())
            ax, bx = float(self.mc2_e["ax"].get()), float(self.mc2_e["bx"].get())
            ay, by = float(self.mc2_e["ay"].get()), float(self.mc2_e["by"].get())
            N2 = int(self.mc2_e["N2"].get())
            vol = (bx - ax) * (by - ay)
            xr = np.random.uniform(ax, bx, N2); yr = np.random.uniform(ay, by, N2)
            v2 = np.array([float(f2(xr[i], yr[i])) for i in range(N2)])
            m2 = np.mean(v2); var2 = np.var(v2, ddof=1); i2 = vol * m2; se2 = vol * np.sqrt(var2 / N2)
            self._set_result(self.mc2_r, f"MC 2D = {fmt(i2, 10)} | IC95%: [{fmt(i2-1.96*se2)}, {fmt(i2+1.96*se2)}]")
        except Exception as e:
            self._set_result(self.mc2_r, f"Error 2D: {e}")

    # ==================== EULER ====================
    def _build_euler(self):
        p = self._make_panel('euler')
        self._header(p, "Euler", "EDO dy/dx = f(x,y) - Clase 7")
        self.eu_e = self._inputs(p, [("f(x,y)", "x + y", 20), ("x0", "0", 6), ("y0", "1", 6), ("xf", "2", 6), ("h", "0.2", 6)])
        self.eu_r = self._result(p)
        self._buttons(p, self._run_euler)
        self.eu_t, self.eu_f, self.eu_a, self.eu_c = self._table_chart(p)

    def _run_euler(self):
        f = parse_fxy(self.eu_e["f(x,y)"].get())
        x, y = float(self.eu_e["x0"].get()), float(self.eu_e["y0"].get())
        xf, h = float(self.eu_e["xf"].get()), float(self.eu_e["h"].get())
        rows = [(0, fmt(x, 6), fmt(y), fmt(float(f(x, y))))]; i = 1
        while x + h / 2 < xf:
            y += h * float(f(x, y)); x = round(x + h, 10)
            rows.append((i, fmt(x, 6), fmt(y), fmt(float(f(x, y))))); i += 1
            if i > 10000: break
        self._set_result(self.eu_r, f"y({fmt(xf,4)}) = {fmt(float(rows[-1][2]),10)} | Pasos:{len(rows)-1}")
        self._fill_table(self.eu_t, ('i', 'x', 'y', 'f(x,y)'), rows)
        self.eu_a.clear()
        self.eu_a.plot([float(r[1]) for r in rows], [float(r[2]) for r in rows], 'b-o', ms=4, label='Euler')
        self.eu_a.legend(fontsize=8); self.eu_a.grid(True, alpha=0.3); self.eu_c.draw()

    # ==================== EULER MODIFICADO ====================
    def _build_euler_mod(self):
        p = self._make_panel('euler_mod')
        self._header(p, "Euler Modificado (Heun)", "Predictor-corrector O(2) - Clase 7")
        self.em_e = self._inputs(p, [("f(x,y)", "x + y", 20), ("x0", "0", 6), ("y0", "1", 6), ("xf", "2", 6), ("h", "0.2", 6)])
        self.em_r = self._result(p)
        self._buttons(p, self._run_euler_mod)
        self.em_t, self.em_f, self.em_a, self.em_c = self._table_chart(p)

    def _run_euler_mod(self):
        f = parse_fxy(self.em_e["f(x,y)"].get())
        x, y = float(self.em_e["x0"].get()), float(self.em_e["y0"].get())
        xf, h = float(self.em_e["xf"].get()), float(self.em_e["h"].get())
        rows = [(0, fmt(x, 6), fmt(y), '-', '-', '-')]; i = 1
        while x + h / 2 < xf:
            f0 = float(f(x, y)); yP = y + h * f0; x1 = round(x + h, 10)
            f1 = float(f(x1, yP)); y = y + h / 2 * (f0 + f1); x = x1
            rows.append((i, fmt(x, 6), fmt(y), fmt(yP), fmt(f0), fmt(f1))); i += 1
            if i > 10000: break
        self._set_result(self.em_r, f"y({fmt(xf,4)}) = {fmt(float(rows[-1][2]),10)} | Pasos:{len(rows)-1}")
        self._fill_table(self.em_t, ('i', 'x', 'y', 'y*(pred)', 'f0', 'f1'), rows)
        self.em_a.clear()
        self.em_a.plot([float(r[1]) for r in rows], [float(r[2]) for r in rows], 'b-o', ms=4, label='Euler Mod.')
        self.em_a.legend(fontsize=8); self.em_a.grid(True, alpha=0.3); self.em_c.draw()

    # ==================== TAYLOR ====================
    def _build_taylor(self):
        p = self._make_panel('taylor')
        self._header(p, "Taylor Orden 2", "Clase 7")
        self.tay_e = self._inputs(p, [("f(x,y)", "x + y", 18), ("f'(x,y)", "1 + x + y", 18), ("x0", "0", 6), ("y0", "1", 6), ("xf", "2", 6), ("h", "0.2", 6)])
        self.tay_r = self._result(p)
        self._buttons(p, self._run_taylor)
        self.tay_t, self.tay_f, self.tay_a, self.tay_c = self._table_chart(p)

    def _run_taylor(self):
        f = parse_fxy(self.tay_e["f(x,y)"].get()); df = parse_fxy(self.tay_e["f'(x,y)"].get())
        x, y = float(self.tay_e["x0"].get()), float(self.tay_e["y0"].get())
        xf, h = float(self.tay_e["xf"].get()), float(self.tay_e["h"].get())
        rows = [(0, fmt(x, 6), fmt(y), fmt(float(f(x, y))), fmt(float(df(x, y))))]; i = 1
        while x + h / 2 < xf:
            fv, dfv = float(f(x, y)), float(df(x, y))
            y += h * fv + (h ** 2 / 2) * dfv; x = round(x + h, 10)
            rows.append((i, fmt(x, 6), fmt(y), fmt(float(f(x, y))), fmt(float(df(x, y))))); i += 1
            if i > 10000: break
        self._set_result(self.tay_r, f"y({fmt(xf,4)}) = {fmt(float(rows[-1][2]),10)} | Pasos:{len(rows)-1}")
        self._fill_table(self.tay_t, ('i', 'x', 'y', 'f', "f'"), rows)
        self.tay_a.clear()
        self.tay_a.plot([float(r[1]) for r in rows], [float(r[2]) for r in rows], 'b-o', ms=4, label='Taylor O(2)')
        self.tay_a.legend(fontsize=8); self.tay_a.grid(True, alpha=0.3); self.tay_c.draw()

    # ==================== RUNGE-KUTTA ====================
    def _build_runge_kutta(self):
        p = self._make_panel('runge_kutta')
        self._header(p, "Runge-Kutta (RK4)", "Clase 7")
        self.rk_e = self._inputs(p, [("f(x,y)", "x + y", 20), ("x0", "0", 6), ("y0", "1", 6), ("xf", "2", 6), ("h", "0.2", 6)])
        self.rk_r = self._result(p)
        self._buttons(p, self._run_rk4)
        self.rk_t, self.rk_f, self.rk_a, self.rk_c = self._table_chart(p)

    def _run_rk4(self):
        f = parse_fxy(self.rk_e["f(x,y)"].get())
        x, y = float(self.rk_e["x0"].get()), float(self.rk_e["y0"].get())
        xf, h = float(self.rk_e["xf"].get()), float(self.rk_e["h"].get())
        rows = [(0, fmt(x, 6), fmt(y), '-', '-', '-', '-')]; i = 1
        while x + h / 2 < xf:
            k1 = h * float(f(x, y)); k2 = h * float(f(x + h / 2, y + k1 / 2))
            k3 = h * float(f(x + h / 2, y + k2 / 2)); k4 = h * float(f(x + h, y + k3))
            y += (k1 + 2 * k2 + 2 * k3 + k4) / 6; x = round(x + h, 10)
            rows.append((i, fmt(x, 6), fmt(y), fmt(k1), fmt(k2), fmt(k3), fmt(k4))); i += 1
            if i > 10000: break
        self._set_result(self.rk_r, f"y({fmt(xf,4)}) = {fmt(float(rows[-1][2]),10)} | Pasos:{len(rows)-1}")
        self._fill_table(self.rk_t, ('i', 'x', 'y', 'k1', 'k2', 'k3', 'k4'), rows)
        self.rk_a.clear()
        self.rk_a.plot([float(r[1]) for r in rows], [float(r[2]) for r in rows], 'b-o', ms=4, label='RK4')
        self.rk_a.legend(fontsize=8); self.rk_a.grid(True, alpha=0.3); self.rk_c.draw()

    # ==================== EDP ====================
    def _build_edp(self):
        p = self._make_panel('edp')
        self._header(p, "EDP - Diferencias Finitas", "Laplace/Poisson en [0,1]x[0,1] - Clase 7")
        self.edp_e = self._inputs(p, [("f(x,y)", "0", 18), ("N", "10", 5)])
        self.edp_b = self._inputs(p, [("u(x,0) inf", "sin(pi*x)", 14), ("u(x,1) sup", "0", 14), ("u(0,y) izq", "0", 14), ("u(1,y) der", "0", 14)])
        self.edp_r = self._result(p)
        self._buttons(p, self._run_edp)
        # Solo grafica (sin tabla)
        cf = ttk.Frame(p)
        cf.pack(fill=tk.BOTH, expand=True, padx=14, pady=(0, 8))
        self.edp_fig = Figure(figsize=(6, 5), dpi=100); self.edp_fig.set_facecolor('#fff')
        self.edp_ax = self.edp_fig.add_subplot(111)
        self.edp_canvas = FigureCanvasTkAgg(self.edp_fig, master=cf)
        self.edp_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _run_edp(self):
        from sympy import pi as sym_pi
        fExpr = self.edp_e["f(x,y)"].get()
        N = int(self.edp_e["N"].get()); h = 1 / (N + 1)
        def ev_boundary(expr, val):
            return float(sympify(expr).subs({x_sym: val, Symbol('pi'): float(sym_pi)}))
        u = np.zeros((N + 2, N + 2))
        for i in range(N + 2):
            xi = i * h
            u[i, 0] = ev_boundary(self.edp_b["u(x,0) inf"].get(), xi)
            u[i, N + 1] = ev_boundary(self.edp_b["u(x,1) sup"].get(), xi)
            u[0, i] = ev_boundary(self.edp_b["u(0,y) izq"].get(), i * h)
            u[N + 1, i] = ev_boundary(self.edp_b["u(1,y) der"].get(), i * h)
        f_src = parse_fxy(fExpr) if fExpr.strip() != '0' else None
        h2 = h * h
        for _ in range(5000):
            mx = 0
            for i in range(1, N + 1):
                for j in range(1, N + 1):
                    fv = float(f_src(i * h, j * h)) if f_src else 0
                    nv = (u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1] + h2 * fv) / 4
                    mx = max(mx, abs(nv - u[i, j])); u[i, j] = nv
            if mx < 1e-6: break
        mid = N // 2 + 1
        self._set_result(self.edp_r, f"Resuelto {N}x{N} pts. u(0.5,0.5) ~ {fmt(u[mid, mid])}")
        self.edp_ax.clear()
        im = self.edp_ax.imshow(u.T, origin='lower', extent=[0, 1, 0, 1], cmap='RdYlBu_r', aspect='equal')
        self.edp_ax.set_xlabel('x'); self.edp_ax.set_ylabel('y'); self.edp_ax.set_title('u(x,y)', fontsize=10)
        if hasattr(self, '_edp_cb'):
            self._edp_cb.remove()
        self._edp_cb = self.edp_fig.colorbar(im, ax=self.edp_ax)
        self.edp_canvas.draw()

    # ==================== GAUSS ====================
    def _build_gauss(self):
        p = self._make_panel('gauss')
        self._header(p, "Eliminacion Gaussiana", "Pivoteo parcial")
        ctrl = ttk.Frame(p); ctrl.pack(fill=tk.X, padx=14, pady=4)
        ttk.Label(ctrl, text="n:").pack(side=tk.LEFT)
        self.gauss_n = ttk.Combobox(ctrl, values=['2', '3', '4', '5'], width=3, state='readonly')
        self.gauss_n.set('3'); self.gauss_n.pack(side=tk.LEFT, padx=4)
        self.gauss_n.bind('<<ComboboxSelected>>', lambda e: self._build_gauss_m())
        self.gauss_mf = ttk.Frame(p); self.gauss_mf.pack(fill=tk.X, padx=14, pady=2)
        self.gauss_em = []; self.gauss_r = self._result(p)
        self._buttons(p, self._run_gauss)
        self.gauss_st = scrolledtext.ScrolledText(p, height=10, font=('Consolas', 9), wrap=tk.WORD)
        self.gauss_st.pack(fill=tk.BOTH, expand=True, padx=14, pady=(0, 8))
        self._build_gauss_m()

    def _build_gauss_m(self):
        for w in self.gauss_mf.winfo_children(): w.destroy()
        n = int(self.gauss_n.get())
        d = {3: [[2, 1, -1, 8], [-3, -1, 2, -11], [-2, 1, 2, -3]]}
        self.gauss_em = []
        for i in range(n):
            rf = ttk.Frame(self.gauss_mf); rf.pack(pady=1); row = []
            for j in range(n + 1):
                if j == n: ttk.Label(rf, text="|", font=('Consolas', 10, 'bold'), foreground='#1a73e8').pack(side=tk.LEFT)
                v = d.get(n, [[1 if i == j else 0 for j in range(n + 1)] for i in range(n)])[i][j]
                e = ttk.Entry(rf, width=6, font=('Consolas', 10), justify='center'); e.insert(0, str(v)); e.pack(side=tk.LEFT, padx=1); row.append(e)
            self.gauss_em.append(row)

    def _run_gauss(self):
        n = int(self.gauss_n.get()); A = [[float(self.gauss_em[i][j].get()) for j in range(n + 1)] for i in range(n)]
        self.gauss_st.delete('1.0', tk.END)
        def ms(m, l):
            s = f"\n{l}\n"
            for r in m: s += "  ".join(f"{v:10.4f}" for v in r) + "\n"
            return s
        self.gauss_st.insert(tk.END, ms(A, "Original:"))
        for k in range(n - 1):
            mv, mr = abs(A[k][k]), k
            for i in range(k + 1, n):
                if abs(A[i][k]) > mv: mv, mr = abs(A[i][k]), i
            if mr != k: A[k], A[mr] = A[mr], A[k]; self.gauss_st.insert(tk.END, ms(A, f"Swap {k+1}<->{mr+1}"))
            if abs(A[k][k]) < 1e-12: messagebox.showerror("Error", "Singular"); return
            for i in range(k + 1, n):
                fac = A[i][k] / A[k][k]
                for j in range(k, n + 1): A[i][j] -= fac * A[k][j]
            self.gauss_st.insert(tk.END, ms(A, f"Elim col {k+1}:"))
        x = [0.0] * n
        for i in range(n - 1, -1, -1):
            x[i] = (A[i][n] - sum(A[i][j] * x[j] for j in range(i + 1, n))) / A[i][i]
        self._set_result(self.gauss_r, "  ".join(f"x{i+1}={fmt(x[i], 10)}" for i in range(n)))

    # ==================== GAUSS-SEIDEL ====================
    def _build_gauss_seidel(self):
        p = self._make_panel('gauss_seidel')
        self._header(p, "Gauss-Seidel", "Iterativo")
        ctrl = ttk.Frame(p); ctrl.pack(fill=tk.X, padx=14, pady=4)
        ttk.Label(ctrl, text="n:").pack(side=tk.LEFT)
        self.gs_n = ttk.Combobox(ctrl, values=['2', '3', '4'], width=3, state='readonly')
        self.gs_n.set('3'); self.gs_n.pack(side=tk.LEFT, padx=4)
        self.gs_n.bind('<<ComboboxSelected>>', lambda e: self._build_gs_m())
        ttk.Label(ctrl, text="Tol:").pack(side=tk.LEFT, padx=(8, 0))
        self.gs_tol = ttk.Entry(ctrl, width=8, font=('Consolas', 9)); self.gs_tol.insert(0, "0.0001"); self.gs_tol.pack(side=tk.LEFT, padx=2)
        ttk.Label(ctrl, text="Max:").pack(side=tk.LEFT, padx=(8, 0))
        self.gs_mx = ttk.Entry(ctrl, width=4, font=('Consolas', 9)); self.gs_mx.insert(0, "50"); self.gs_mx.pack(side=tk.LEFT, padx=2)
        self.gs_mf = ttk.Frame(p); self.gs_mf.pack(fill=tk.X, padx=14, pady=2)
        self.gs_em = []; self.gs_r = self._result(p)
        self._buttons(p, self._run_gs)
        self.gs_t, self.gs_f, self.gs_a, self.gs_c = self._table_chart(p)
        self._build_gs_m()

    def _build_gs_m(self):
        for w in self.gs_mf.winfo_children(): w.destroy()
        n = int(self.gs_n.get())
        d = {3: [[4, 1, -1, 5], [2, 7, 1, 15], [1, -3, 12, 36]]}
        self.gs_em = []
        for i in range(n):
            rf = ttk.Frame(self.gs_mf); rf.pack(pady=1); row = []
            for j in range(n + 1):
                if j == n: ttk.Label(rf, text="|", font=('Consolas', 10, 'bold'), foreground='#1a73e8').pack(side=tk.LEFT)
                v = d.get(n, [[10 if i == j else 1 for j in range(n + 1)] for i in range(n)])[i][j]
                e = ttk.Entry(rf, width=6, font=('Consolas', 10), justify='center'); e.insert(0, str(v)); e.pack(side=tk.LEFT, padx=1); row.append(e)
            self.gs_em.append(row)

    def _run_gs(self):
        n = int(self.gs_n.get()); tol = float(self.gs_tol.get()); mxi = int(self.gs_mx.get())
        A = [[float(self.gs_em[i][j].get()) for j in range(n)] for i in range(n)]
        b = [float(self.gs_em[i][n].get()) for i in range(n)]
        x = [0.0] * n; rows = []
        for it in range(1, mxi + 1):
            xo = x[:]
            for i in range(n):
                s = b[i] - sum(A[i][j] * x[j] for j in range(n) if j != i); x[i] = s / A[i][i]
            err = max(abs(x[i] - xo[i]) for i in range(n))
            rows.append((it,) + tuple(fmt(x[i]) for i in range(n)) + (fmt(err),))
            if err < tol: break
        self._set_result(self.gs_r, "  ".join(f"x{i+1}={fmt(x[i], 10)}" for i in range(n)) + f" | Iter:{rows[-1][0]}")
        cols = ('Iter',) + tuple(f'x{i+1}' for i in range(n)) + ('Error',)
        self._fill_table(self.gs_t, cols, rows)
        self.gs_a.clear()
        for i in range(n):
            self.gs_a.plot(range(1, len(rows) + 1), [float(r[i + 1]) for r in rows], '-o', ms=3, label=f'x{i+1}')
        self.gs_a.legend(fontsize=8); self.gs_a.grid(True, alpha=0.3); self.gs_a.set_title('Convergencia', fontsize=9); self.gs_c.draw()


if __name__ == '__main__':
    root = tk.Tk()
    app = SimuladorApp(root)
    root.mainloop()
