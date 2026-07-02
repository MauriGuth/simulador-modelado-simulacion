"""Graficos - tercera tanda del examen.
Ej2: nulclinas/competencia con parametro mu.  Ej3: Van der Pol.  Ej4: SIR."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/home/user/simulador-modelado-simulacion/graficos_examen/"

def rk4(f, X, Y, dt, n):
    x = np.array(X, dtype=float); y = np.array(Y, dtype=float)
    xs = [x.copy()]; ys = [y.copy()]
    for _ in range(n):
        k1x, k1y = f(x, y)
        k2x, k2y = f(x + .5*dt*k1x, y + .5*dt*k1y)
        k3x, k3y = f(x + .5*dt*k2x, y + .5*dt*k2y)
        k4x, k4y = f(x + dt*k3x, y + dt*k3y)
        x = x + dt/6*(k1x + 2*k2x + 2*k3x + k4x)
        y = y + dt/6*(k1y + 2*k2y + 2*k3y + k4y)
        x = np.clip(x, -1e4, 1e4); y = np.clip(y, -1e4, 1e4)
        xs.append(x.copy()); ys.append(y.copy())
    return np.array(xs), np.array(ys)

def vfield(ax, f, xr, yr, dens=20, color="0.55"):
    xx, yy = np.meshgrid(np.linspace(*xr, dens), np.linspace(*yr, dens))
    u, v = f(xx, yy)
    n = np.hypot(u, v); n[n == 0] = 1
    ax.quiver(xx, yy, u/n, v/n, color=color, alpha=.5, width=.003, scale=40, pivot="mid")

# ===================================================================
# EJ.2  xdot = x(2-x-y),  ydot = y(1+mu-x-y)   nulclinas y equilibrios
# ===================================================================
def panel(ax, mu):
    f = lambda x, y: (x*(2 - x - y), y*(1 + mu - x - y))
    L = 3.4
    vfield(ax, f, (0, L), (0, L), dens=19)
    xs = np.linspace(0, L, 100)
    ax.plot(xs, 2 - xs, color="royalblue", lw=2, label=r"$\dot x=0:\ x+y=2$")
    ax.plot(xs, (1+mu) - xs, color="seagreen", lw=2, label=r"$\dot y=0:\ x+y=1+\mu$")
    ax.axvline(0, color="royalblue", lw=2, alpha=.5)
    ax.axhline(0, color="seagreen", lw=2, alpha=.5)
    # trayectorias
    for x0 in np.linspace(0.2, 3.2, 6):
        for y0 in np.linspace(0.2, 3.2, 6):
            X, Y = rk4(lambda x,y: np.array(f(x,y)), x0, y0, 0.01, 700)
            ax.plot(X, Y, color="k", lw=.5, alpha=.3)
    # equilibrios
    eqs = [(0,0,"white","(0,0) fuente"), (2,0,"black","(2,0)"),
           (0,1+mu,"black",f"(0,{1+mu:g})")]
    for ex,ey,c,_ in eqs:
        ax.plot([ex],[ey],"o",mfc=c,mec="k",ms=10,mew=1.4,zorder=6)
    ax.set_xlim(0,L); ax.set_ylim(0,L); ax.set_aspect("equal")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title(rf"$\mu={mu:g}$")
    ax.legend(fontsize=8, loc="upper right"); ax.grid(alpha=.2)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 5.3))
panel(a1, 0.5); a1.text(2.05,0.15,"ESTABLE\n(gana x)",fontsize=8,color="navy")
a1.text(0.05,1.55,"silla",fontsize=8,color="darkred")
panel(a2, 2.0); a2.text(2.05,0.15,"silla",fontsize=8,color="darkred")
a2.text(0.05,3.05,"ESTABLE\n(gana y)",fontsize=8,color="darkgreen")
fig.suptitle(r"Ej.2  Competencia: $\mu<1$ gana x  |  $\mu>1$ gana y  (bifurcacion en $\mu=1$)",
             fontsize=12)
fig.tight_layout(); fig.savefig(OUT+"ej2b_nulclinas.png", dpi=150); plt.close(fig)

# ===================================================================
# EJ.3  Van der Pol:  x1dot=x2,  x2dot=mu(1-x1^2)x2 - x1
# ===================================================================
def vdp_panel(ax, mu):
    f = lambda x, y: (y, mu*(1 - x**2)*y - x)
    L = 4.2 if mu <= 1.5 else 4.6
    vfield(ax, f, (-L, L), (-L, L), dens=21)
    # ciclo limite: integrar largo desde un punto lejano
    X, Y = rk4(lambda x,y: np.array(f(x,y)), 3.5, 0.0, 0.005, 8000)
    ax.plot(X[3000:], Y[3000:], color="crimson", lw=2.4, label="ciclo límite")
    # trayectoria desde afuera y desde adentro
    X2, Y2 = rk4(lambda x,y: np.array(f(x,y)), 0.05, 0.05, 0.005, 4000)
    ax.plot(X2, Y2, color="#1f5fb0", lw=1.0, alpha=.85, label="desde el origen (sale)")
    X3, Y3 = rk4(lambda x,y: np.array(f(x,y)), -4.0, 3.5, 0.005, 4000)
    ax.plot(X3, Y3, color="darkorange", lw=1.0, alpha=.85, label="desde afuera (entra)")
    ax.plot([0],[0],"o",mfc="white",mec="k",ms=9,mew=1.5,zorder=6,label="(0,0) inestable")
    ax.set_xlim(-L,L); ax.set_ylim(-L,L); ax.set_aspect("equal")
    ax.axhline(0,color="k",lw=.5); ax.axvline(0,color="k",lw=.5)
    ax.set_xlabel(r"$x_1=x$"); ax.set_ylabel(r"$x_2=\dot x$")
    ax.set_title(rf"Van der Pol  $\mu={mu:g}$")
    ax.legend(fontsize=8, loc="upper right"); ax.grid(alpha=.2)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 5.4))
vdp_panel(a1, 1.0); vdp_panel(a2, 3.0)
fig.suptitle("Ej.3  Van der Pol: el origen es inestable y todo tiende a un CICLO LÍMITE estable",
             fontsize=12)
fig.tight_layout(); fig.savefig(OUT+"ej3b_vanderpol.png", dpi=150); plt.close(fig)

# ===================================================================
# EJ.4  SIR   Sdot=-a S I, Idot=a S I - b I, Rdot=b I
# ===================================================================
al, be = 0.2, 5.0
S0, I0 = 25.0, 3.0
Sc = be/al
# integrar en el tiempo (RK4 3D via 2D S,I y R=N-S-I)
def sir(S, I): return (-al*S*I, al*S*I - be*I)
dt = 0.001; N = 12000
S, I = S0, I0
tS=[0]; SS=[S0]; II=[I0]; RR=[0.0]; t=0
for _ in range(N):
    k1=sir(S,I); k2=sir(S+.5*dt*k1[0],I+.5*dt*k1[1])
    k3=sir(S+.5*dt*k2[0],I+.5*dt*k2[1]); k4=sir(S+dt*k3[0],I+dt*k3[1])
    S+=dt/6*(k1[0]+2*k2[0]+2*k3[0]+k4[0]); I+=dt/6*(k1[1]+2*k2[1]+2*k3[1]+k4[1])
    t+=dt; tS.append(t); SS.append(S); II.append(I); RR.append(S0+I0-S-I)
tS=np.array(tS); SS=np.array(SS); II=np.array(II); RR=np.array(RR)

fig, (axt, axp) = plt.subplots(1, 2, figsize=(11, 4.8))
# temporal
axt.plot(tS, SS, color="#1f5fb0", lw=2.2, label="S(t) susceptibles")
axt.plot(tS, II, color="crimson", lw=2.2, label="I(t) infectados")
axt.plot(tS, RR, color="seagreen", lw=2.2, label="R(t) recuperados")
axt.axhline(SS[-1], ls=":", color="#1f5fb0", lw=1)
axt.annotate(f"$S_\\infty\\approx{SS[-1]:.2f}$", (tS[-1], SS[-1]),
             xytext=(tS[-1]*0.55, SS[-1]+2.5), fontsize=10, color="navy")
axt.set_xlabel("t"); axt.set_ylabel("personas")
axt.set_title("Ej.4  Evolución temporal (SIR)")
axt.legend(fontsize=9); axt.grid(alpha=.3); axt.set_xlim(0, tS[-1])
# plano de fase S-I con umbral
Sg = np.linspace(0.5, 28, 300)
Icurve = I0 + (S0 - Sg) + (be/al)*np.log(Sg/S0)
axp.plot(Sg, Icurve, color="purple", lw=2, label=r"trayectoria $I(S)$")
axp.plot(SS, II, color="crimson", lw=1.4, ls="--", label="RK4")
axp.axvline(Sc, color="black", ls="--", lw=1.5, label=r"umbral $S_c=\beta/\alpha=25$")
axp.plot([S0],[I0],"go",ms=8,label="inicio (25,3)")
axp.plot([SS[-1]],[0],"o",mfc="white",mec="k",ms=8, label=f"fin ({SS[-1]:.2f},0)")
axp.fill_betweenx([0,3.4], 0, Sc, color="#e8f4e8", alpha=.6)
axp.fill_betweenx([0,3.4], Sc, 28, color="#fdeaea", alpha=.6)
axp.text(4,2.55,"$S<S_c$\n$\\dot I<0$ (decae)",fontsize=9,color="green")
axp.text(25.4,2.4,"$S>S_c$\n$\\dot I>0$",fontsize=9,color="crimson")
axp.set_xlim(0,28); axp.set_ylim(0,3.4)
axp.set_xlabel("S"); axp.set_ylabel("I")
axp.set_title("Ej.4  Plano de fase S-I y umbral")
axp.legend(fontsize=8, loc="lower left"); axp.grid(alpha=.3)
fig.tight_layout(); fig.savefig(OUT+"ej4b_sir.png", dpi=150); plt.close(fig)

print("S_inf(RK4)=%.4f  I_max=%.4f" % (SS[-1], II.max()))
print("listos: ej2b_nulclinas, ej3b_vanderpol, ej4b_sir")
