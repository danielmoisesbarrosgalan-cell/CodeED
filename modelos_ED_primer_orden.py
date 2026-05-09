"""
=============================================================
TALLER EVALUATIVO: MODELADO MATEMÁTICO Y SIMULACIÓN
Universidad de La Guajira - Ingeniería de Sistemas
Ecuaciones Diferenciales - Unidad III
=============================================================
Temas:
  1. Ley de Enfriamiento de Newton
  2. Crecimiento y Decaimiento Exponencial
  3. Mezclas y Dilución
=============================================================
Requisitos: pip install numpy matplotlib scipy
=============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import sys

# ─────────────────────────────────────────────────────────
# CONFIGURACIÓN VISUAL
# ─────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0f1117",
    "axes.facecolor":   "#1a1d2e",
    "axes.edgecolor":   "#444466",
    "axes.labelcolor":  "#c8cce8",
    "text.color":       "#c8cce8",
    "xtick.color":      "#888aaa",
    "ytick.color":      "#888aaa",
    "grid.color":       "#2a2d40",
    "grid.linestyle":   "--",
    "grid.alpha":       0.6,
    "lines.linewidth":  2.2,
    "font.family":      "monospace",
})

COLORES = ["#7ee8fa", "#ff6b9d", "#a8edea", "#fed6e3", "#fccb90"]

# ─────────────────────────────────────────────────────────
# UTILIDADES
# ─────────────────────────────────────────────────────────
def separador(titulo: str) -> None:
    ancho = 60
    print("\n" + "═" * ancho)
    print(f"  {titulo}")
    print("═" * ancho)

def input_float(mensaje: str, default: float = None) -> float:
    while True:
        try:
            texto = input(mensaje)
            if texto == "" and default is not None:
                return default
            return float(texto)
        except ValueError:
            print("  ⚠  Ingrese un número válido.")

def input_int(mensaje: str, default: int = None) -> int:
    while True:
        try:
            texto = input(mensaje)
            if texto == "" and default is not None:
                return default
            return int(texto)
        except ValueError:
            print("  ⚠  Ingrese un entero válido.")

# ─────────────────────────────────────────────────────────
# MÓDULO 1: LEY DE ENFRIAMIENTO DE NEWTON
# ─────────────────────────────────────────────────────────
def enfriamiento_newton() -> None:
    """
    Modelo:  dT/dt = -k(T - T_amb)
    Solución analítica: T(t) = T_amb + (T0 - T_amb) * e^(-k*t)

    Aplicación en Ing. de Sistemas:
      Gestión térmica de servidores y Data Centers.
      Un servidor a alta temperatura se enfría al detener procesos;
      esta curva determina cuándo puede volver a operar de forma segura.
    """
    separador("1 · LEY DE ENFRIAMIENTO DE NEWTON")
    print("""
  Modelo matemático:
    dT/dt = -k(T - T_amb)
    T(t)  = T_amb + (T₀ - T_amb)·e^(-k·t)

  Aplicación: Enfriamiento de CPUs / Servidores en Data Centers
    """)

    # ── Entrada de datos ──────────────────────────────────
    T0    = input_float("  Temperatura inicial del objeto T₀ (°C) [default 90]: ", 90)
    T_amb = input_float("  Temperatura ambiente T_amb (°C)        [default 22]: ", 22)
    k     = input_float("  Constante de enfriamiento k (>0)       [default 0.08]: ", 0.08)
    t_fin = input_float("  Tiempo de simulación (minutos)         [default 60]: ", 60)

    if k <= 0:
        print("  ⚠  k debe ser mayor que 0. Se usará 0.08.")
        k = 0.08

    # ── Solución analítica ────────────────────────────────
    t     = np.linspace(0, t_fin, 500)
    T_sol = T_amb + (T0 - T_amb) * np.exp(-k * t)

    # ── Solución numérica (odeint) ────────────────────────
    def modelo(T, t):
        return -k * (T - T_amb)

    T_num = odeint(modelo, T0, t).flatten()

    # ── Cálculos clave ────────────────────────────────────
    t_mitad_idx = np.argmin(np.abs(T_sol - (T_amb + (T0 - T_amb) / 2)))
    t_mitad     = t[t_mitad_idx]
    vida_media  = np.log(2) / k          # semivida teórica

    print(f"""
  ── Resultados ──────────────────────────────────────
  T inicial        : {T0:.2f} °C
  T ambiente       : {T_amb:.2f} °C
  k                : {k}
  T a t={t_fin/4:.1f} min   : {T_amb + (T0-T_amb)*np.exp(-k*t_fin/4):.2f} °C
  T a t={t_fin/2:.1f} min  : {T_amb + (T0-T_amb)*np.exp(-k*t_fin/2):.2f} °C
  T a t={t_fin:.1f} min  : {T_amb + (T0-T_amb)*np.exp(-k*t_fin):.2f} °C
  Semivida (t½)    : {vida_media:.2f} min  (temp. media alcanzada en ≈{t_mitad:.1f} min)
    """)

    # ── Gráfica ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t, T_sol, color=COLORES[0], label="Solución analítica")
    ax.plot(t, T_num, color=COLORES[1], linestyle="--", lw=1.4, label="Solución numérica (odeint)")
    ax.axhline(T_amb, color="#888aaa", linestyle=":", label=f"T_amb = {T_amb}°C")
    ax.axvline(t_mitad, color=COLORES[3], linestyle=":", alpha=0.7,
               label=f"t semivida ≈ {t_mitad:.1f} min")
    ax.scatter([0], [T0], color="white", zorder=5)

    ax.fill_between(t, T_sol, T_amb, alpha=0.08, color=COLORES[0])
    ax.set_title("Ley de Enfriamiento de Newton\n(Gestión Térmica de Servidores)",
                 fontsize=13, pad=12)
    ax.set_xlabel("Tiempo (min)")
    ax.set_ylabel("Temperatura (°C)")
    ax.legend(fontsize=9)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig("grafica_enfriamiento_newton.png", dpi=150, bbox_inches="tight")
    print("   Gráfica guardada: grafica_enfriamiento_newton.png")
    plt.show()


# ─────────────────────────────────────────────────────────
# MÓDULO 2: CRECIMIENTO Y DECAIMIENTO EXPONENCIAL
# ─────────────────────────────────────────────────────────
def crecimiento_decaimiento() -> None:
    """
    Modelo:  dP/dt = k·P
    Solución analítica: P(t) = P₀·e^(k·t)
      k > 0 → crecimiento   (usuarios activos, tráfico de red)
      k < 0 → decaimiento   (obsolescencia de hardware, datos en caché)

    Aplicación en Ing. de Sistemas:
      Modelar el crecimiento de usuarios en una plataforma digital
      o la tasa de obsolescencia de componentes de hardware.
    """
    separador("2 · CRECIMIENTO Y DECAIMIENTO EXPONENCIAL")
    print("""
  Modelo matemático:
    dP/dt = k·P
    P(t)  = P₀·e^(k·t)

  k > 0 → Crecimiento  (ej: usuarios en plataforma, datos almacenados)
  k < 0 → Decaimiento  (ej: obsolescencia de hardware, caché)
    """)

    modo = input("  Modo: [1] Crecimiento  [2] Decaimiento  [default 1]: ").strip() or "1"
    P0   = input_float("  Valor inicial P₀                          [default 1000]: ", 1000)

    if modo == "2":
        k     = input_float("  Tasa de decaimiento k (valor positivo)    [default 0.05]: ", 0.05)
        k     = -abs(k)
        t_fin = input_float("  Tiempo de simulación (años)               [default 20]: ", 20)
        unidad_y, unidad_t = "Unidades / Equipos", "Años"
        titulo = "Decaimiento Exponencial\n(Obsolescencia de Hardware)"
    else:
        k     = input_float("  Tasa de crecimiento k (valor positivo)    [default 0.12]: ", 0.12)
        k     = abs(k)
        t_fin = input_float("  Tiempo de simulación (meses)              [default 24]: ", 24)
        unidad_y, unidad_t = "Usuarios / Unidades", "Meses"
        titulo = "Crecimiento Exponencial\n(Usuarios en Plataforma Digital)"

    # ── Solución ──────────────────────────────────────────
    t     = np.linspace(0, t_fin, 500)
    P_sol = P0 * np.exp(k * t)

    def modelo(P, t):
        return k * P

    P_num = odeint(modelo, P0, t).flatten()

    # ── Tiempo de duplicación / semivida ──────────────────
    t_doble = np.log(2) / abs(k)
    etiqueta = "Semivida" if k < 0 else "T. duplicación"

    print(f"""
  ── Resultados ──────────────────────────────────────
  P₀               : {P0:.2f}
  k                : {k:.4f}
  {etiqueta:<17}: {t_doble:.2f} {unidad_t.split('/')[0].strip()}
  P a t={t_fin/4:.1f}        : {P0*np.exp(k*t_fin/4):.2f}
  P a t={t_fin/2:.1f}        : {P0*np.exp(k*t_fin/2):.2f}
  P a t={t_fin:.1f}        : {P0*np.exp(k*t_fin):.2f}
    """)

    # ── Gráfica ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t, P_sol, color=COLORES[2], label="Solución analítica")
    ax.plot(t, P_num, color=COLORES[4], linestyle="--", lw=1.4, label="Solución numérica (odeint)")
    ax.axvline(t_doble, color=COLORES[1], linestyle=":", alpha=0.8,
               label=f"{etiqueta} ≈ {t_doble:.2f} {unidad_t.split('/')[0].strip()}")
    ax.scatter([0], [P0], color="white", zorder=5)

    color_fill = COLORES[2] if k > 0 else COLORES[1]
    ax.fill_between(t, P_sol, alpha=0.07, color=color_fill)
    ax.set_title(titulo, fontsize=13, pad=12)
    ax.set_xlabel(f"Tiempo ({unidad_t})")
    ax.set_ylabel(unidad_y)
    ax.legend(fontsize=9)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig("grafica_crecimiento_decaimiento.png", dpi=150, bbox_inches="tight")
    print("   Gráfica guardada: grafica_crecimiento_decaimiento.png")
    plt.show()


# ─────────────────────────────────────────────────────────
# MÓDULO 3: MEZCLAS Y DILUCIÓN
# ─────────────────────────────────────────────────────────
def mezclas_dilucion() -> None:
    """
    Modelo (tanque con entrada y salida):
      dQ/dt = (c_ent · f_ent) - (Q/V) · f_sal
      donde V(t) = V0 + (f_ent - f_sal)·t  (volumen variable)

    Si f_ent == f_sal → V constante:
      dQ/dt = c_ent·f - (f/V)·Q
      Solución: Q(t) = V·c_ent + (Q0 - V·c_ent)·e^(-f·t/V)

    Aplicación en Ing. de Sistemas:
      Modelar el flujo de paquetes en una red (buffer de router),
      donde llegan datos con cierta concentración y salen con otra.
    """
    separador("3 · MEZCLAS Y DILUCIÓN")
    print("""
  Modelo matemático (tanque bien mezclado):
    dQ/dt = c_ent·f_ent - (Q/V(t))·f_sal
    V(t)  = V₀ + (f_ent - f_sal)·t

  Aplicación: Buffer de paquetes en un router de red
    """)

    V0    = input_float("  Volumen inicial del tanque V₀ (L)           [default 100]: ", 100)
    Q0    = input_float("  Cantidad inicial de sustancia Q₀ (g o kg)   [default 5]: ", 5)
    c_ent = input_float("  Concentración de entrada c_ent (g/L)        [default 0.5]: ", 0.5)
    f_ent = input_float("  Flujo de entrada f_ent (L/min)              [default 4]: ", 4)
    f_sal = input_float("  Flujo de salida  f_sal (L/min)              [default 4]: ", 4)
    t_fin = input_float("  Tiempo de simulación (min)                  [default 60]: ", 60)

    # ── Solución numérica odeint ──────────────────────────
    t = np.linspace(0, t_fin, 500)

    def modelo(Q, t):
        V = V0 + (f_ent - f_sal) * t
        if V <= 0:
            return 0.0
        return c_ent * f_ent - (f_sal / V) * Q

    Q_num = odeint(modelo, Q0, t).flatten()
    V_t   = V0 + (f_ent - f_sal) * t
    V_t   = np.maximum(V_t, 1e-9)           # evitar división por 0
    C_t   = Q_num / V_t                      # concentración en el tiempo

    # ── Solución analítica (sólo si f_ent == f_sal) ───────
    tiene_analitica = np.isclose(f_ent, f_sal, rtol=1e-4)
    if tiene_analitica:
        V   = V0                             # volumen constante
        f   = f_ent
        Q_eq  = V * c_ent
        Q_sol = Q_eq + (Q0 - Q_eq) * np.exp(-f * t / V)
        C_sol = Q_sol / V
    else:
        Q_sol = None
        C_sol = None

    print(f"""
  ── Resultados ──────────────────────────────────────
  V₀               : {V0:.2f} L
  Q₀               : {Q0:.2f}  (conc. inicial {Q0/V0:.4f} g/L)
  Flujo neto       : {f_ent - f_sal:+.2f} L/min
  Q en t={t_fin:.0f} min  : {Q_num[-1]:.4f}
  Conc. en t={t_fin:.0f}  : {C_t[-1]:.4f} g/L
  Conc. entrada    : {c_ent:.4f} g/L
  ¿Equilibrio posible? : {"Sí (flujos iguales)" if tiene_analitica else "No (flujos distintos, V varía)"}
    """)

    # ── Gráfica doble ─────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Cantidad Q(t)
    ax1.plot(t, Q_num, color=COLORES[0], label="Q(t) numérico")
    if tiene_analitica:
        ax1.plot(t, Q_sol, color=COLORES[1], linestyle="--", lw=1.4, label="Q(t) analítico")
        ax1.axhline(V0 * c_ent, color="#888aaa", linestyle=":", label=f"Q_eq = {V0*c_ent:.2f}")
    ax1.scatter([0], [Q0], color="white", zorder=5)
    ax1.fill_between(t, Q_num, alpha=0.07, color=COLORES[0])
    ax1.set_title("Cantidad de Sustancia Q(t)", fontsize=12)
    ax1.set_xlabel("Tiempo (min)")
    ax1.set_ylabel("Cantidad (g)")
    ax1.legend(fontsize=9)
    ax1.grid(True)

    # Concentración C(t)
    ax2.plot(t, C_t, color=COLORES[2], label="C(t) numérica")
    if tiene_analitica:
        ax2.plot(t, C_sol, color=COLORES[4], linestyle="--", lw=1.4, label="C(t) analítica")
    ax2.axhline(c_ent, color="#888aaa", linestyle=":", label=f"c_ent = {c_ent} g/L")
    ax2.fill_between(t, C_t, alpha=0.07, color=COLORES[2])
    ax2.set_title("Concentración C(t) en el Tanque", fontsize=12)
    ax2.set_xlabel("Tiempo (min)")
    ax2.set_ylabel("Concentración (g/L)")
    ax2.legend(fontsize=9)
    ax2.grid(True)

    fig.suptitle("Modelo de Mezclas y Dilución\n(Buffer de Paquetes en Router de Red)",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig("grafica_mezclas_dilucion.png", dpi=150, bbox_inches="tight")
    print(" Gráfica guardada: grafica_mezclas_dilucion.png")
    plt.show()


# ─────────────────────────────────────────────────────────
# MENÚ PRINCIPAL
# ─────────────────────────────────────────────────────────
def menu() -> None:
    print("""
╔══════════════════════════════════════════════════════╗
║  SIMULADOR DE ECUACIONES DIFERENCIALES - UNIDAD III  ║
║  Universidad de La Guajira · Ingeniería de Sistemas  ║
╚══════════════════════════════════════════════════════╝

  [1]  Ley de Enfriamiento de Newton
  [2]  Crecimiento y Decaimiento Exponencial
  [3]  Mezclas y Dilución
  [4]  Ejecutar los 3 modelos seguidos
  [0]  Salir
    """)

    opciones = {
        "1": enfriamiento_newton,
        "2": crecimiento_decaimiento,
        "3": mezclas_dilucion,
    }

    while True:
        opcion = input("  Seleccione una opción: ").strip()

        if opcion == "0":
            print("\n  Hasta luego. ¡Éxitos en el taller! \n")
            sys.exit(0)
        elif opcion == "4":
            enfriamiento_newton()
            crecimiento_decaimiento()
            mezclas_dilucion()
            break
        elif opcion in opciones:
            opciones[opcion]()
            print("\n  ✔  Módulo completado. Volviendo al menú...")
            menu()
            break
        else:
            print("  ⚠  Opción no válida. Intente de nuevo.")


# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    menu()
