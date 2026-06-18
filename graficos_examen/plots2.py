"""Graficos para la resolucion del segundo examen (5 ejercicios).
Integrador RK4 propio (vectorizado) para no depender de scipy."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/home/user/simulador-modelado-simulacion/graficos_examen/"

def rk4(f, X, Y, dt, n):
    """Integra (x,y) (escalares o arrays) n pasos. Devuelve trayectoria si escalar."""
    xs = [np.array(X, dtype=float)]; ys = [np.array(Y, dtype=float)]
    x = np.array(X, dtype=float); y = np.array(Y, dtype=float)
    for _ in range(n):
        k1x, k1y = f(x, y)
        k2x, k2y = f(x + .5*dt*k1x, y + .5*dt*k1y)
        k3x, k3y = f(x + .5*dt*k2x, y + .5*dt*k2y)
        k4x, k4y = f(x + dt*k3x, y + dt*k3y)
        x = x + dt/6*(k1x + 2*k2x + 2*k3x + k4x)
        y = y + dt/6*(k1y + 2*k2y + 2*k3y + k4y)
        x = np.clip(x, -1e6, 1e6); y = np.clip(y, -1e6, 1e6)
        xs.append(x.copy()); ys.append(y.copy())
    return np.array(xs), np.array(ys)

def vector_field(ax, f, xr, yr, dens=22, color="0.55"):
    xx, yy = np.meshgrid(np.linspace(*xr, dens), np.linspace(*yr, dens))
    u, v = f(xx, yy)
    n = np.hypot(u, v); n[n == 0] = 1
    ax.quiver(xx, yy, u/n, v/n, color=color, alpha=.55,
              width=.0028, scale=42, pivot="mid")

# ===================================================================
# EJERCICIO 1 - Logistico
# ===================================================================
k, r, x0 = 20000.0, 0.15, 1500.0
A = (k - x0)/x0
ts = (1/r)*np.log(A)

# (1a) temporal
fig, ax = plt.subplots(figsize=(6.4, 4.0))
t = np.linspace(0, 60, 600)
x = k/(1 + A*np.exp(-r*t))
ax.plot(t, x, color="#1f5fb0", lw=2.4, label=r"$x(t)=k/(1+Ae^{-rt})$")
ax.axhline(k, ls="--", color="crimson", lw=1.3, label=f"capacidad $k$={k:.0f}")
ax.axhline(k/2, ls=":", color="gray", lw=1.1)
ax.axvline(ts, ls=":", color="green", lw=1.3)
ax.plot([ts], [k/2], "o", color="green", ms=7)
ax.annotate(f"$t_s$={ts:.2f} min\n($x=k/2$=10000)", (ts, k/2),
            xytext=(ts+6, k/2-3500), fontsize=10,
            arrowprops=dict(arrowstyle="->", color="green"))
ax.plot([0], [x0], "o", color="black", ms=6)
ax.annotate(f"$x_0$={x0:.0f}", (0, x0), xytext=(3, x0+1200), fontsize=10)
ax.set_xlabel("t  [min]"); ax.set_ylabel("x  [rps]")
ax.set_title("Ej.1  Comportamiento temporal (logistica)")
ax.legend(loc="center right", fontsize=9); ax.grid(alpha=.3)
ax.set_ylim(0, k*1.08)
fig.tight_layout(); fig.savefig(OUT+"ej1_temporal.png", dpi=150); plt.close(fig)

# (1b) linea de fase  y  (1c) bifurcacion
fig, (axp, axb) = plt.subplots(1, 2, figsize=(9.6, 4.0))
# linea de fase: xdot vs x
xv = np.linspace(-3000, 23000, 400)
xdot = r*xv*(1 - xv/k)
axp.plot(xv, xdot, color="#1f5fb0", lw=2.2)
axp.axhline(0, color="k", lw=.8); axp.axvline(0, color="k", lw=.6)
# flechas sobre el eje x (signo de xdot)
for xa in [-2000, 4000, 10000, 16000, 22000]:
    s = r*xa*(1 - xa/k)
    axp.annotate("", xy=(xa + (1600 if s>0 else -1600), 0), xytext=(xa, 0),
                 arrowprops=dict(arrowstyle="-|>", color="darkorange", lw=2))
axp.plot([0], [0], "o", mfc="white", mec="k", ms=11, mew=1.6, zorder=5)   # inestable
axp.plot([k], [0], "o", color="k", ms=11, zorder=5)                        # estable
axp.text(0, 1700, "0\n(inestable)", ha="center", fontsize=9)
axp.text(k, -2600, "k=20000\n(estable)", ha="center", fontsize=9)
axp.set_xlabel("x"); axp.set_ylabel(r"$\dot{x}=rx(1-x/k)$")
axp.set_title("Ej.1  Diagrama de fase (1D)"); axp.grid(alpha=.3)
# bifurcacion: r como parametro, x* = 0 y x* = k, intercambio en r=0 (transcritica)
rr = np.linspace(-0.3, 0.3, 200)
axb.plot(rr[rr>=0], 0*rr[rr>=0], "--", color="crimson", lw=2)      # x*=0 inestable (r>0)
axb.plot(rr[rr<0], 0*rr[rr<0], "-", color="green", lw=2.4)         # x*=0 estable  (r<0)
axb.plot(rr[rr>=0], k+0*rr[rr>=0], "-", color="green", lw=2.4)     # x*=k estable  (r>0)
axb.plot(rr[rr<0], k+0*rr[rr<0], "--", color="crimson", lw=2)      # x*=k inestable(r<0)
axb.axvline(0, color="k", lw=.7, ls=":")
axb.plot([0,0],[0,k], "o", color="purple", ms=7)
axb.text(0.02, k*0.55, "intercambio\nde estabilidad\nen r=0", fontsize=8.5)
axb.set_xlabel("r (parametro de control)"); axb.set_ylabel("x*  (equilibrio)")
axb.set_title("Ej.1  Diagrama de bifurcacion")
axb.plot([], [], "-", color="green", lw=2.4, label="estable")
axb.plot([], [], "--", color="crimson", lw=2, label="inestable")
axb.legend(fontsize=9, loc="center left"); axb.grid(alpha=.3)
axb.set_ylim(-3000, 23000)
fig.tight_layout(); fig.savefig(OUT+"ej1_fase_bif.png", dpi=150); plt.close(fig)

# ===================================================================
# EJERCICIO 2 - lineal silla  xdot=x+2y, ydot=4x+3y
# ===================================================================
f2 = lambda x, y: (x + 2*y, 4*x + 3*y)
fig, ax = plt.subplots(figsize=(6.0, 6.0))
L = 4
vector_field(ax, f2, (-L, L), (-L, L))
xs = np.linspace(-L, L, 100)
ax.plot(xs, -xs/2, "--", color="royalblue", lw=1.8, label=r"Nulclina $\dot x=0:\ y=-x/2$")
ax.plot(xs, -4*xs/3, "-.", color="seagreen", lw=1.8, label=r"Nulclina $\dot y=0:\ y=-4x/3$")
ax.plot(xs, 2*xs, color="red", lw=2, label=r"$v_1=(1,2)$, $\lambda_1=5$ (inestable)")
ax.plot(xs, -xs, color="purple", lw=2, label=r"$v_2=(1,-1)$, $\lambda_2=-1$ (estable)")
for c1 in [-1.5,-.5,.5,1.5]:
    for c2 in [-1.5,-.5,.5,1.5]:
        X, Y = rk4(lambda x,y: np.array(f2(x,y)), c1+c2, 2*c1-c2, 0.01, 90)
        m = (np.abs(X[:,...])<L) & (np.abs(Y[:,...])<L)
        ax.plot(X, Y, color="k", lw=.8, alpha=.5)
ax.plot([0],[0],"ko",ms=7)
ax.set_xlim(-L,L); ax.set_ylim(-L,L); ax.set_aspect("equal")
ax.axhline(0,color="k",lw=.6); ax.axvline(0,color="k",lw=.6)
ax.set_xlabel("x"); ax.set_ylabel("y")
ax.set_title(r"Ej.2  Punto silla en (0,0) — $\lambda_1=5,\ \lambda_2=-1$")
ax.legend(fontsize=7.5, loc="upper left"); ax.grid(alpha=.25)
fig.tight_layout(); fig.savefig(OUT+"ej2_silla.png", dpi=150); plt.close(fig)

# ===================================================================
# EJERCICIO 3 - no lineal  xdot=y, ydot=x^2+y^2-1
# Integral primera exacta: H=(x^2+y^2+x-1/2) e^{-2x}  (curvas de nivel = orbitas)
# ===================================================================
f3 = lambda x, y: (y, x**2 + y**2 - 1)
fig, ax = plt.subplots(figsize=(6.0, 6.0))
L = 2.3
vector_field(ax, f3, (-L, L), (-L, L), dens=24)
gx, gy = np.meshgrid(np.linspace(-L, L, 600), np.linspace(-L, L, 600))
H = (gx**2 + gy**2 + gx - 0.5)*np.exp(-2*gx)
Hsep = 1.5*np.exp(-2.0)        # valor de H en la silla (1,0): separatrices
lv = sorted([-3.5,-3,-2.5,-2,-1.5,-1,-0.5,0,0.1,0.3,0.6,1.0])
ax.contour(gx, gy, H, levels=lv, colors="#1f5fb0", linewidths=1.0, alpha=.9)
ax.contour(gx, gy, H, levels=[Hsep], colors="crimson", linewidths=2.0)
ax.plot([], [], color="#1f5fb0", lw=1.0, label="orbitas (curvas de nivel de H)")
ax.plot([], [], color="crimson", lw=2.0, label="separatriz (H de la silla)")
ax.plot([1],[0],"s",color="crimson",ms=9,label="(1,0) silla")
ax.plot([-1],[0],"o",color="purple",ms=9,label="(-1,0) centro")
ax.set_xlim(-L,L); ax.set_ylim(-L,L); ax.set_aspect("equal")
ax.axhline(0,color="k",lw=.6); ax.axvline(0,color="k",lw=.6)
ax.set_xlabel("x"); ax.set_ylabel("y")
ax.set_title(r"Ej.3  $\dot x=y,\ \dot y=x^2+y^2-1$  (div $=2y\neq0$: NO conservativo)")
ax.legend(fontsize=8, loc="upper right"); ax.grid(alpha=.25)
fig.tight_layout(); fig.savefig(OUT+"ej3_noconserv.png", dpi=150); plt.close(fig)

# ===================================================================
# EJERCICIO 4 - Lotka-Volterra  xdot=x(2-x-y), ydot=y(3-2x-y)
# ===================================================================
f4 = lambda x, y: (x*(2 - x - y), y*(3 - 2*x - y))
fig, ax = plt.subplots(figsize=(6.4, 6.0))
L = 3.4
# clasificacion de cuencas: integrar grilla y ver destino
g = 240
gx, gy = np.meshgrid(np.linspace(0.001, L, g), np.linspace(0.001, L, g))
X, Y = rk4(lambda x,y: np.array(f4(x,y)), gx, gy, 0.02, 1400)
xf, yf = X[-1], Y[-1]
to_rabbit = (np.abs(xf-2) + np.abs(yf-0)) < (np.abs(xf-0) + np.abs(yf-3))
ax.contourf(gx, gy, to_rabbit.astype(float), levels=[-.5,.5,1.5],
            colors=["#d9ead3", "#fde9d9"], alpha=.8)
vector_field(ax, f4, (0, L), (0, L), dens=20, color="0.5")
# separatriz = variedad estable de la silla (1,1), autovector (1,sqrt2)
ev = np.array([1, np.sqrt(2)]); ev = ev/np.linalg.norm(ev)
for s in (+1, -1):
    p = np.array([1.0,1.0]) + s*1e-3*ev
    Xs, Ys = rk4(lambda x,y: np.array(f4(x,y)), p[0], p[1], -0.01, 1600)  # tiempo atras
    ax.plot(Xs, Ys, color="red", lw=2.4)
ax.plot([],[],color="red",lw=2.4,label="Separatriz (var. estable silla)")
# trayectorias de muestra
for x0v in np.linspace(0.2, 3.2, 7):
    for y0v in np.linspace(0.2, 3.2, 7):
        Xt, Yt = rk4(lambda x,y: np.array(f4(x,y)), x0v, y0v, 0.01, 800)
        ax.plot(Xt, Yt, color="k", lw=.6, alpha=.35)
# equilibrios
eq = {"(0,0) fuente":(0,0,"white"), "(2,0) nodo est.":(2,0,"black"),
      "(0,3) nodo est.":(0,3,"black"), "(1,1) silla":(1,1,"red")}
for name,(ex,ey,c) in eq.items():
    ax.plot([ex],[ey],"o",mfc=c,mec="k",ms=10,mew=1.4,zorder=6)
ax.text(2.55, 0.95, "Ganan\nCONEJOS\n-> (2,0)", fontsize=9, color="#7a4")
ax.text(0.15, 2.6, "Ganan\nOVEJAS\n-> (0,3)", fontsize=9, color="#c70")
ax.set_xlim(0,L); ax.set_ylim(0,L); ax.set_aspect("equal")
ax.set_xlabel("x  (conejos)"); ax.set_ylabel("y  (ovejas)")
ax.set_title("Ej.4  Competencia Lotka-Volterra (exclusion competitiva)")
ax.legend(fontsize=8.5, loc="upper right"); ax.grid(alpha=.2)
fig.tight_layout(); fig.savefig(OUT+"ej4_lotka.png", dpi=150); plt.close(fig)

# ===================================================================
# EJERCICIO 5 - foco estable  xdot=-y-x r^2, ydot=x-y r^2
# ===================================================================
f5 = lambda x, y: (-y - x*(x**2+y**2), x - y*(x**2+y**2))
fig, ax = plt.subplots(figsize=(6.0, 6.0))
L = 1.25
vector_field(ax, f5, (-L, L), (-L, L), dens=22)
for ang in np.linspace(0, 2*np.pi, 7, endpoint=False):
    x0v, y0v = 1.1*np.cos(ang), 1.1*np.sin(ang)
    X, Y = rk4(lambda x,y: np.array(f5(x,y)), x0v, y0v, 0.01, 1400)
    ax.plot(X, Y, color="#1f5fb0", lw=1.2, alpha=.9)
ax.plot([0],[0],"o",color="crimson",ms=9,label="(0,0) foco estable")
ax.set_xlim(-L,L); ax.set_ylim(-L,L); ax.set_aspect("equal")
ax.axhline(0,color="k",lw=.6); ax.axvline(0,color="k",lw=.6)
ax.set_xlabel("x"); ax.set_ylabel("y")
ax.set_title(r"Ej.5  Foco estable: $\dot r=-r^3,\ \dot\theta=1,\ r(t)\to0$")
ax.legend(fontsize=9, loc="upper right"); ax.grid(alpha=.25)
fig.tight_layout(); fig.savefig(OUT+"ej5_espiral.png", dpi=150); plt.close(fig)

print("OK  ts=%.4f min  A=%.4f" % (ts, A))
print("listos: ej1_temporal, ej1_fase_bif, ej2_silla, ej3_noconserv, ej4_lotka, ej5_espiral")
