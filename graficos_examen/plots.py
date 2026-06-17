import numpy as np
import matplotlib.pyplot as plt

OUT = "/home/user/simulador-modelado-simulacion/graficos_examen/"

# ---------- Integrador RK4 (sin dependencias extra) ----------
def integrate(f, p0, t0, t1, dt):
    ts = np.arange(t0, t1, dt)
    xs = []
    x = np.array(p0, dtype=float)
    for t in ts:
        xs.append(x.copy())
        k1 = f(x)
        k2 = f(x + 0.5*dt*k1)
        k3 = f(x + 0.5*dt*k2)
        k4 = f(x + dt*k3)
        x = x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        if np.any(np.abs(x) > 60):
            break
    return np.array(xs)

def vector_field(ax, f, xr, yr, n=22, color='0.55'):
    X, Y = np.meshgrid(np.linspace(*xr, n), np.linspace(*yr, n))
    U = np.zeros_like(X); V = np.zeros_like(Y)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            d = f(np.array([X[i,j], Y[i,j]]))
            U[i,j], V[i,j] = d
    M = np.hypot(U, V); M[M==0]=1
    ax.quiver(X, Y, U/M, V/M, color=color, alpha=0.6,
              width=0.0035, scale=42, pivot='mid')

def style(ax, title, lim=4):
    ax.axhline(0, color='k', lw=0.8); ax.axvline(0, color='k', lw=0.8)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect('equal'); ax.grid(alpha=0.25)
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_title(title)

# ============================================================
# EJERCICIO 1 :  x' = x - 2y ,  y' = -2x + y   (punto silla)
# ============================================================
def ej1():
    f = lambda v: np.array([v[0]-2*v[1], -2*v[0]+v[1]])
    fig, ax = plt.subplots(figsize=(7,7))
    vector_field(ax, f, (-4,4), (-4,4))
    xs = np.linspace(-4,4,100)
    # Nulclinas
    ax.plot(xs, xs/2, 'b--', lw=2, label="Nulclina  x'=0 :  y=x/2")
    ax.plot(xs, 2*xs, 'g--', lw=2, label="Nulclina  y'=0 :  y=2x")
    # Direcciones invariantes (autovectores)
    ax.plot(xs, xs, color='purple', lw=2.5, label="Estable  λ=-1 : v=(1,1)")
    ax.plot(xs, -xs, color='red', lw=2.5, label="Inestable λ=3 : v=(1,-1)")
    # Trayectorias
    seeds = [(0.4,0.5),( -0.4,-0.5),(0.5,0.3),(-0.5,-0.3),
             (0.2,-0.2),(-0.2,0.2),(2,2.2),(-2,-2.2),(2.2,2),(-2.2,-2)]
    for s in seeds:
        tr = integrate(f, s, 0, 4, 0.01)
        ax.plot(tr[:,0], tr[:,1], 'k', lw=1.1)
        if len(tr) > 5:
            ax.annotate('', xy=tr[len(tr)//3+1], xytext=tr[len(tr)//3],
                        arrowprops=dict(arrowstyle='-|>', color='k'))
    ax.plot(0,0,'ko', ms=9)
    style(ax, "Ej.1  Punto silla en (0,0)  —  λ₁=-1, λ₂=3")
    ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
    fig.tight_layout(); fig.savefig(OUT+"ej1_silla.png", dpi=140); plt.close(fig)

# ============================================================
# EJERCICIO 2 :  x' = x^2 - y^2 - 1 ,  y' = 2y
# ============================================================
def ej2():
    f = lambda v: np.array([v[0]**2 - v[1]**2 - 1, 2*v[1]])
    fig, ax = plt.subplots(figsize=(7,7))
    vector_field(ax, f, (-3.5,3.5), (-3.5,3.5))
    # Nulclina y'=0 : y=0
    ax.axhline(0, color='g', lw=2, ls='--', label="Nulclina y'=0 : y=0")
    # Nulclina x'=0 : x^2 - y^2 = 1 (hiperbola)
    yy = np.linspace(-3.5,3.5,400)
    xr = np.sqrt(1+yy**2); xl = -np.sqrt(1+yy**2)
    ax.plot(xr, yy, 'b--', lw=2, label="Nulclina x'=0 : x²-y²=1")
    ax.plot(xl, yy, 'b--', lw=2)
    # Equilibrios
    ax.plot(1,0,'ro', ms=10); ax.annotate("(1,0) nodo inest.", (1.05,0.15))
    ax.plot(-1,0,'ks', ms=10); ax.annotate("(-1,0) silla", (-1.95,0.18))
    # Trayectorias
    seeds = [(-1,0.05),(-1,-0.05),(0,0.4),(0,-0.4),(1.2,0.2),(0.8,0.2),
             (-2,0.3),(-2,-0.3),(2,0.3),(2,-0.3),(0.5,1.5),(-0.5,-1.5),
             (-1.5,0.6),(-1.5,-0.6),(1.5,0.6)]
    for s in seeds:
        tr = integrate(f, s, 0, 3, 0.005)
        ax.plot(tr[:,0], tr[:,1], 'k', lw=1.0)
        if len(tr) > 8:
            k = len(tr)//4
            ax.annotate('', xy=tr[k+1], xytext=tr[k],
                        arrowprops=dict(arrowstyle='-|>', color='k'))
    style(ax, "Ej.2  No lineal — equilibrios (1,0) y (-1,0)", lim=3.5)
    ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
    fig.tight_layout(); fig.savefig(OUT+"ej2_nolineal.png", dpi=140); plt.close(fig)

# ============================================================
# EJERCICIO 3 :  a = -2  ->  espiral estable
#   A = [[-2,-2],[2,1]]
# ============================================================
def ej3():
    A = np.array([[-2,-2],[2,1]])
    f = lambda v: A.dot(v)
    fig, ax = plt.subplots(figsize=(7,7))
    vector_field(ax, f, (-4,4), (-4,4))
    seeds = [(3,3),(-3,-3),(3,-3),(-3,3),(1,3),(-1,-3),(3,1),(-3,-1)]
    for s in seeds:
        tr = integrate(f, s, 0, 12, 0.01)
        ax.plot(tr[:,0], tr[:,1], 'b', lw=1.2)
        k = max(1,len(tr)//12)
        ax.annotate('', xy=tr[k+1], xytext=tr[k],
                    arrowprops=dict(arrowstyle='-|>', color='b'))
    ax.plot(0,0,'ko', ms=9)
    style(ax, "Ej.3  a=-2 : espiral ESTABLE  (λ=-1/2 ± (√7/2)i)")
    fig.tight_layout(); fig.savefig(OUT+"ej3_espiral.png", dpi=140); plt.close(fig)

# ============================================================
# EJERCICIO 4 : Romeo & Julieta  R'=aR+bJ , J'=bR+aJ
#   caso representativo a=-1, b=2  ->  silla (a+b=1>0, a-b=-3<0)
# ============================================================
def ej4():
    a, b = -1, 2
    f = lambda v: np.array([a*v[0]+b*v[1], b*v[0]+a*v[1]])
    fig, ax = plt.subplots(figsize=(7,7))
    vector_field(ax, f, (-4,4), (-4,4))
    rs = np.linspace(-4,4,100)
    ax.plot(rs, rs, color='red', lw=2.5, label="Inestable λ=a+b=1 : (1,1)")
    ax.plot(rs, -rs, color='purple', lw=2.5, label="Estable λ=a-b=-3 : (1,-1)")
    seeds = [(0.5,0.3),(-0.5,-0.3),(0.3,0.5),(-0.3,-0.5),
             (0.5,-0.3),(-0.5,0.3),(2,-2.1),(-2,2.1),(3,2.6),(-3,-2.6)]
    for s in seeds:
        tr = integrate(f, s, 0, 4, 0.01)
        ax.plot(tr[:,0], tr[:,1], 'k', lw=1.1)
        if len(tr) > 5:
            k = len(tr)//3
            ax.annotate('', xy=tr[k+1], xytext=tr[k],
                        arrowprops=dict(arrowstyle='-|>', color='k'))
    ax.plot(0,0,'ko', ms=9)
    style(ax, "Ej.4  Romeo-Julieta (a=-1,b=2): punto silla")
    ax.set_xlabel('R (Romeo)'); ax.set_ylabel('J (Julieta)')
    ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
    fig.tight_layout(); fig.savefig(OUT+"ej4_romeo.png", dpi=140); plt.close(fig)

ej1(); ej2(); ej3(); ej4()
print("Listo: 4 graficos generados.")
