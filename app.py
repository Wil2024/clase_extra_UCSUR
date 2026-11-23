# -*- coding: utf-8 -*-
"""Actividad: EL ANALISTA DE NEGOCIOS — Laboratorio de Business Intelligence (Streamlit)
Ejecutar:  streamlit run app.py
Clave docente: variable INSTRUCTOR_PASSWORD en st.secrets (por defecto: analista2026)"""
import sqlite3
from datetime import datetime
import numpy as np, pandas as pd, plotly.express as px, plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

st.set_page_config(page_title="El Analista de Negocios", page_icon="📊", layout="wide")
try: _v = tuple(int(x) for x in st.__version__.split(".")[:2])
except Exception: _v = (1, 0)
ANCHO = {"width": "stretch"} if _v >= (1, 50) else {"use_container_width": True}
TEAL, GOLD, NAVY, BLUE, RED = "#14b8a6", "#f5b942", "#0b1f3a", "#3b82f6", "#f87171"
PALETA = [TEAL, GOLD, BLUE, "#a78bfa", RED, "#34d399"]
st.markdown(f"""<style>
.stApp{{background:linear-gradient(160deg,#f8fafc 0%,#e8f0f8 100%);color:{NAVY}}}
h1,h2,h3,h4{{color:{NAVY}!important}}
[data-testid="stMetric"]{{background:#ffffff;border-left:5px solid {TEAL};padding:12px 16px;border-radius:8px;box-shadow:0 1px 4px rgba(11,31,58,.15)}}
[data-testid="stMetricValue"]{{color:#0f766e}} [data-testid="stMetricLabel"]{{color:{NAVY}}}
.caja{{background:#ffffff;color:{NAVY};border:2px solid {GOLD};border-radius:10px;padding:14px 18px;margin:8px 0}}
.badge{{display:inline-block;background:{GOLD};color:{NAVY};font-weight:700;padding:4px 12px;border-radius:20px;margin:3px}}
</style>""", unsafe_allow_html=True)

def estilo(fig, h=380):
    fig.update_layout(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      height=h, margin=dict(l=10, r=10, t=45, b=10), colorway=PALETA)
    return fig
usd = lambda x: f"US$ {x:,.0f}"

# ------------------------------------------------------------------ DATOS
@st.cache_data
def generar_datos(n=1000, seed=2026):
    rng = np.random.default_rng(seed)
    paises = {"Perú": (.26, "Andina", 0), "México": (.24, "Norte y Centroamérica", .035), "Colombia": (.18, "Andina", .01),
              "Chile": (.14, "Cono Sur", 0), "Argentina": (.10, "Cono Sur", .02), "Ecuador": (.08, "Andina", .005)}
    canales = {"Tienda física": (.20, .04, .06), "E-commerce": (.25, .07, .09), "Distribuidores": (.20, .12, .05),
               "Marketplace": (.27, .11, .17), "WhatsApp Business": (.08, .03, .04)}  # peso, desc. medio, costo de servir
    prods = [("Alimentos y Bebidas", "Café Orgánico 1kg", 45, .62, 1.6), ("Alimentos y Bebidas", "Quinua Premium 1kg", 28, .55, 1.2),
             ("Alimentos y Bebidas", "Chocolate Cacao 70%", 18, .58, 1.4), ("Tecnología", "Audífonos Bluetooth", 120, .74, 2.2),
             ("Tecnología", "Smartwatch Fit", 260, .70, 1.0), ("Tecnología", "Power Bank 20k mAh", 75, .68, 1.8),
             ("Hogar", "Licuadora Pro", 210, .68, .8), ("Hogar", "Set de Ollas", 180, .66, .7), ("Hogar", "Lámpara LED Smart", 60, .60, 1.0),
             ("Salud y Cuidado Personal", "Suplemento Colágeno", 85, .35, 1.0), ("Salud y Cuidado Personal", "Kit Cuidado Facial", 110, .38, .9),
             ("Salud y Cuidado Personal", "Oxímetro Digital", 70, .52, .8), ("Oficina", "Silla Ergonómica", 480, .70, .5),
             ("Oficina", "Escritorio Ajustable", 900, .68, .3), ("Oficina", "Organizador Pack", 35, .50, .9)]
    P = pd.DataFrame(prods, columns=["categoria", "producto", "precio", "ratio_costo", "w"])
    # clientes por país (concentración tipo Pareto)
    cli = {}
    k = 1
    for p, (w, _, _) in paises.items():
        m = max(int(round(140 * w)), 8)
        ids = [f"CLI-{k+i:03d}" for i in range(m)]; k += m
        pw = rng.pareto(1.3, m) + .3
        cli[p] = (ids, pw / pw.sum())
    d0, d1 = pd.Timestamp("2025-01-01"), pd.Timestamp("2026-10-31")
    dias = pd.date_range(d0, d1)
    wd = 1 + .5 * dias.month.isin([11, 12]) + .3 * (dias.month == 5)
    fechas = rng.choice(dias, n, p=wd / wd.sum())
    pais = rng.choice(list(paises), n, p=[v[0] for v in paises.values()])
    canal = rng.choice(list(canales), n, p=[v[0] for v in canales.values()])
    pr = P.iloc[rng.choice(len(P), n, p=P.w / P.w.sum())].reset_index(drop=True)
    cliente = [rng.choice(cli[p][0], p=cli[p][1]) for p in pais]
    base = np.where(pr.precio < 100, 5, np.where(pr.precio < 300, 2, 1))
    qty = np.clip(rng.poisson(base), 1, None) * np.where(canal == "Distribuidores", 3, 1)
    dmed = np.array([canales[c][1] for c in canal]) + np.array([paises[p][2] for p in pais])
    desc = np.clip(rng.normal(dmed, .025), 0, .30)
    cts = np.array([canales[c][2] for c in canal])
    bruto = qty * pr.precio.values
    d_monto = bruto * desc
    neto = bruto - d_monto
    c_prod = bruto * pr.ratio_costo.values * rng.uniform(.97, 1.03, n)
    c_canal = neto * cts
    df = pd.DataFrame({"id_transaccion": [f"TX-{i:04d}" for i in range(1, n + 1)], "fecha": fechas, "cliente_id": cliente,
                       "pais": pais, "region": [paises[p][1] for p in pais], "categoria": pr.categoria, "producto": pr.producto,
                       "canal": canal, "cantidad": qty, "precio_lista": pr.precio, "descuento_pct": desc.round(4),
                       "ingreso_bruto": bruto.round(2), "descuento_monto": d_monto.round(2), "ingreso_neto": neto.round(2),
                       "costo_producto": c_prod.round(2), "costo_canal": c_canal.round(2)})
    df["margen"] = (df.ingreso_neto - df.costo_producto - df.costo_canal).round(2)
    df["margen_pct"] = (df.margen / df.ingreso_neto).round(4)
    return df.sort_values("fecha").reset_index(drop=True)

DF = generar_datos()
FASES = [(10, "1️⃣ Explicación"), (20, "2️⃣ Exploración"), (20, "3️⃣ Retos"), (10, "4️⃣ Puesta en común")]

# ------------------------------------------------------------------ BASE DE DATOS (retos y ranking)
def db():
    con = sqlite3.connect("laboratorio.db")
    con.execute("create table if not exists respuestas(equipo text, reto int, respuesta text, interpretacion text,"
                " p_dato int, p_interp int, intentos int, ts text, primary key(equipo,reto))")
    con.execute("create table if not exists config(k text primary key, v text)")
    return con
def get_cfg(k):
    r = db().execute("select v from config where k=?", (k,)).fetchone(); return r[0] if r else None
def set_cfg(k, v):
    with db() as c: c.execute("insert or replace into config values(?,?)", (k, v))

@st.fragment(run_every=10)
def reloj():
    ini = get_cfg("inicio")
    if not ini:
        st.info("⏳ El docente aún no inicia el laboratorio."); return
    t = (datetime.now() - datetime.fromisoformat(ini)).total_seconds() / 60
    acum = 0
    for dur, nombre in FASES:
        if t < acum + dur:
            st.markdown(f"**{nombre}**"); st.progress(min((t - acum) / dur, 1.0))
            st.caption(f"⏱️ Quedan {int(acum + dur - t) + 1} min en esta fase · total {int(t)}/60 min"); return
        acum += dur
    st.success("✅ Laboratorio finalizado")

# ------------------------------------------------------------------ FILTROS
st.sidebar.markdown("## 📊 El Analista de Negocios")
with st.sidebar.form("f_equipo"):
    _e = st.text_input("👥 Nombre de tu equipo", value=st.session_state.get("equipo", ""), placeholder="Ej.: Los Quipus Analíticos")
    if st.form_submit_button("Registrar equipo"): st.session_state["equipo"] = _e.strip()
equipo = st.session_state.get("equipo", "")
if equipo: st.sidebar.success(f"Equipo registrado: {equipo}")
with st.sidebar: reloj()
st.sidebar.divider()
modulo = st.sidebar.radio("Módulo", ["🏠 Inicio y dataset", "📈 Dashboard ejecutivo", "🌎 Análisis comercial", "👥 Análisis de clientes",
                                     "💰 Análisis de rentabilidad", "🎯 Retos y ranking", "🔐 Panel docente"])
st.sidebar.markdown("### 🎛️ Filtros dinámicos")
rango = st.sidebar.date_input("Periodo", (DF.fecha.min().date(), DF.fecha.max().date()),
                              min_value=DF.fecha.min().date(), max_value=DF.fecha.max().date())
if len(rango) < 2: rango = (rango[0], rango[0])
regs = st.sidebar.multiselect("Región", sorted(DF.region.unique()), default=sorted(DF.region.unique()))
pais_op = sorted(DF[DF.region.isin(regs)].pais.unique())
paises_sel = st.sidebar.multiselect("País", pais_op, default=pais_op)
cats = st.sidebar.multiselect("Categoría", sorted(DF.categoria.unique()), default=sorted(DF.categoria.unique()))
cans = st.sidebar.multiselect("Canal", sorted(DF.canal.unique()), default=sorted(DF.canal.unique()))
f = DF[(DF.fecha.dt.date >= rango[0]) & (DF.fecha.dt.date <= rango[1]) & DF.pais.isin(paises_sel)
       & DF.categoria.isin(cats) & DF.canal.isin(cans)]
st.sidebar.caption(f"{len(f):,} de {len(DF):,} transacciones")
st.session_state.setdefault("visitados", set()).add(modulo)

def kpis(d):
    ing = d.ingreso_neto.sum(); mg = d.margen.sum()
    return ing, len(d), mg, (mg / ing if ing else 0), (ing / len(d) if len(d) else 0)

if f.empty and modulo not in ("🏠 Inicio y dataset", "🎯 Retos y ranking", "🔐 Panel docente"):
    st.warning("Los filtros no devuelven datos. Amplía el periodo o las selecciones."); st.stop()

# ------------------------------------------------------------------ MÓDULOS
if modulo.startswith("🏠"):
    st.title("📊 Actividad: “El analista de negocios”")
    st.markdown(f"""<div class="caja"><b>Escenario.</b> Eres analista de <b>AndinaTrade S.A.C.</b>, distribuidora con sede en Lima que vende en
    6 países por 5 canales. El gerente comercial te escribe: <i>“Cerramos 2025 con crecimiento, pero el margen no acompaña. Antes del comité
    del lunes necesito saber dónde ganamos plata de verdad y dónde solo movemos volumen. No me traigas gráficos: tráeme decisiones.”</i></div>""",
                unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    c1.markdown("### 🧭 Ruta del laboratorio\n- **10 min** · Explicación de la app y el dataset\n- **20 min** · Exploración individual o en parejas\n"
                "- **20 min** · Resolución de los 4 retos\n- **10 min** · Puesta en común de hallazgos")
    c2.markdown("### 🧠 Regla de oro\nNo basta con leer un gráfico. En cada reto responde: **¿Qué pasa? → ¿Por qué importa? → ¿Qué decisión tomarías?**\n\n"
                "*Ejemplo:* “Marketplace tiene un margen de solo 11 %: entre descuentos y comisión se queda con muy poco. "
                "Recomiendo renegociar la comisión antes de invertir más en ese canal.”")
    st.markdown("### 📖 Diccionario de datos")
    st.dataframe(pd.DataFrame({"Campo": ["ingreso_bruto", "descuento_monto", "ingreso_neto", "costo_producto", "costo_canal", "margen", "margen_pct"],
        "Significado": ["Cantidad × precio de lista", "Descuento comercial otorgado", "Bruto − descuento (lo que realmente se factura)",
                        "Costo de adquirir/producir lo vendido", "Costo de servir el canal (comisión, logística)",
                        "Margen de contribución = neto − costo producto − costo canal", "Margen / ingreso neto"]}), hide_index=True, **ANCHO)
    st.markdown("### 🔎 Muestra del dataset (1,000 transacciones, 2025-ene → 2026-oct, cifras en US$)")
    st.dataframe(DF.head(15), **ANCHO, hide_index=True)

elif modulo.startswith("📈"):
    st.title("📈 Dashboard ejecutivo")
    ing, ped, mg, mgp, tk = kpis(f)
    c = st.columns(5)
    c[0].metric("Ingreso neto", usd(ing)); c[1].metric("Pedidos", f"{ped:,}"); c[2].metric("Margen contribución", usd(mg))
    c[3].metric("Margen %", f"{mgp:.1%}"); c[4].metric("Ticket promedio", usd(tk))
    m = f.groupby(f.fecha.dt.to_period("M").dt.to_timestamp()).agg(ing=("ingreso_neto", "sum"), mg=("margen", "sum"), ped=("id_transaccion", "count")).reset_index()
    m["mgp"] = m.mg / m.ing
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_bar(x=m.fecha, y=m.ing, name="Ingreso neto", marker_color=TEAL)
    fig.add_scatter(x=m.fecha, y=m.mgp, name="Margen %", mode="lines+markers", line=dict(color=GOLD, width=3), secondary_y=True)
    fig.update_yaxes(tickformat=".0%", secondary_y=True); fig.update_layout(title="Ventas mensuales vs margen %")
    st.plotly_chart(estilo(fig, 400), **ANCHO)
    a, b = st.columns(2)
    a.plotly_chart(estilo(px.pie(f, names="categoria", values="ingreso_neto", hole=.55, title="Mix de ingresos por categoría")), **ANCHO)
    pm = f.groupby("pais")[["ingreso_neto", "margen"]].sum().reset_index().sort_values("ingreso_neto")
    b.plotly_chart(estilo(px.bar(pm, y="pais", x=["ingreso_neto", "margen"], barmode="group", orientation="h", title="Ingreso y margen por país")), **ANCHO)
    peor = f.groupby("canal").apply(lambda d: d.margen.sum() / d.ingreso_neto.sum()).idxmin()
    st.markdown(f"<div class='caja'>💡 <b>Pregunta para el comité:</b> con los filtros actuales, el canal de menor margen % es <b>{peor}</b>. ¿Lo estamos impulsando por volumen o por rentabilidad?</div>", unsafe_allow_html=True)

elif modulo.startswith("🌎"):
    st.title("🌎 Análisis comercial")
    dim = st.radio("Comparar ventas por…", ["pais", "canal", "categoria", "producto"], horizontal=True, format_func=str.capitalize)
    g = f.groupby(dim).agg(ingreso=("ingreso_neto", "sum"), pedidos=("id_transaccion", "count"), unidades=("cantidad", "sum")).reset_index().sort_values("ingreso", ascending=False)
    a, b = st.columns(2)
    a.plotly_chart(estilo(px.bar(g, x=dim, y="ingreso", color="ingreso", color_continuous_scale=["#e6f4f1", TEAL, NAVY], title=f"Ingreso neto por {dim}")), **ANCHO)
    b.plotly_chart(estilo(px.bar(g, x=dim, y="unidades", title=f"Unidades vendidas por {dim}", color_discrete_sequence=[GOLD])), **ANCHO)
    st.plotly_chart(estilo(px.treemap(f, path=["categoria", "producto"], values="ingreso_neto", color="margen_pct", color_continuous_scale="RdYlGn", title="Mapa de ingresos (color = margen %)"), 420), **ANCHO)
    pv = f.pivot_table(index="pais", columns="canal", values="ingreso_neto", aggfunc="sum", fill_value=0)
    st.plotly_chart(estilo(px.imshow(pv, text_auto=".2s", color_continuous_scale=["#e6f4f1", TEAL, NAVY], title="Ingreso neto: país × canal"), 400), **ANCHO)

elif modulo.startswith("👥"):
    st.title("👥 Análisis de clientes")
    c1, c2 = st.columns(2)
    hi = c1.slider("Cliente FRECUENTE desde (pedidos)", 6, 25, 12)
    lo = c2.slider("Cliente OCASIONAL hasta (pedidos)", 1, 8, 4)
    cl = f.groupby("cliente_id").agg(pedidos=("id_transaccion", "count"), ingreso=("ingreso_neto", "sum"), margen=("margen", "sum"), pais=("pais", "first")).reset_index()
    cl["segmento"] = np.where(cl.pedidos >= hi, "Frecuente", np.where(cl.pedidos <= lo, "Ocasional", "Regular"))
    cl["ticket"] = cl.ingreso / cl.pedidos
    s = cl.groupby("segmento").agg(clientes=("cliente_id", "count"), ingreso=("ingreso", "sum"), margen=("margen", "sum")).reset_index()
    s["% ingresos"] = (s.ingreso / s.ingreso.sum()).map("{:.1%}".format)
    a, b = st.columns([1, 2])
    a.dataframe(s, hide_index=True, **ANCHO)
    b.plotly_chart(estilo(px.scatter(cl, x="pedidos", y="ticket", size="ingreso", color="segmento", hover_name="cliente_id", title="Frecuencia vs ticket promedio",
                                     color_discrete_map={"Frecuente": TEAL, "Regular": BLUE, "Ocasional": GOLD})), **ANCHO)
    p = cl.sort_values("ingreso", ascending=False).reset_index(drop=True)
    p["acum"] = p.ingreso.cumsum() / p.ingreso.sum(); p["pct_clientes"] = (p.index + 1) / len(p)
    fig = px.area(p, x="pct_clientes", y="acum", title="Curva de Pareto: % de clientes vs % de ingresos", color_discrete_sequence=[TEAL])
    fig.update_xaxes(tickformat=".0%"); fig.update_yaxes(tickformat=".0%")
    top20 = p[p.pct_clientes <= .2].acum.max()
    st.plotly_chart(estilo(fig, 340), **ANCHO)
    st.markdown(f"<div class='caja'>💡 El <b>20 % de los clientes</b> concentra el <b>{top20:.0%}</b> de los ingresos. ¿Qué riesgo implica perder a uno de ellos?</div>", unsafe_allow_html=True)
    st.subheader("🏆 Top 15 clientes"); st.dataframe(p.head(15)[["cliente_id", "pais", "pedidos", "ingreso", "margen", "segmento"]].round(0), hide_index=True, **ANCHO)

elif modulo.startswith("💰"):
    st.title("💰 Análisis de rentabilidad")
    t = f[["ingreso_bruto", "descuento_monto", "costo_producto", "costo_canal"]].sum()
    wf = go.Figure(go.Waterfall(x=["Ingreso bruto", "Descuentos", "Costo producto", "Costo canal", "Margen contribución"],
        measure=["absolute", "relative", "relative", "relative", "total"],
        y=[t.ingreso_bruto, -t.descuento_monto, -t.costo_producto, -t.costo_canal, 0], increasing_marker_color=TEAL, decreasing_marker_color=RED, totals_marker_color=GOLD))
    wf.update_layout(title="De ingreso bruto a margen: ¿dónde se va el dinero?")
    st.plotly_chart(estilo(wf, 380), **ANCHO)
    dim = st.radio("Rentabilidad por…", ["categoria", "canal", "pais", "producto"], horizontal=True, format_func=str.capitalize)
    g = f.groupby(dim).agg(bruto=("ingreso_bruto", "sum"), desc=("descuento_monto", "sum"), neto=("ingreso_neto", "sum"), unidades=("cantidad", "sum"),
                           margen=("margen", "sum")).reset_index()
    g["margen_pct"] = g.margen / g.neto; g["desc_pct"] = g.desc / g.bruto
    a, b = st.columns(2)
    a.plotly_chart(estilo(px.bar(g.sort_values("margen_pct"), x="margen_pct", y=dim, orientation="h", color="margen_pct", color_continuous_scale="RdYlGn", title="Margen % (mayor = mejor)")), **ANCHO)
    b.plotly_chart(estilo(px.scatter(g, x="unidades", y="margen_pct", size="neto", color=dim, text=dim, title="Volumen vs rentabilidad")), **ANCHO)
    hm = f.pivot_table(index="pais", columns="canal", values="descuento_pct", aggfunc="mean")
    st.plotly_chart(estilo(px.imshow(hm, text_auto=".1%", color_continuous_scale="YlOrRd", title="Descuento promedio: país × canal"), 380), **ANCHO)
    st.subheader("🧪 Simulador de decisión: ¿y si reducimos descuentos?")
    c1, c2, c3 = st.columns(3)
    can = c1.selectbox("Canal", sorted(DF.canal.unique()), index=3)
    red = c2.slider("Reducir descuento (puntos %)", 0.0, 8.0, 3.0, .5)
    el = c3.slider("Caída de volumen por cada punto reducido (%)", 0.0, 5.0, 1.5, .5)
    d = f[f.canal == can]
    if not d.empty:
        vol = max(1 - el * red / 100, 0)
        nd = np.maximum(d.descuento_pct - red / 100, 0)
        neto2 = d.ingreso_bruto * (1 - nd) * vol
        mg2 = (neto2 - d.costo_producto * vol - neto2 * (d.costo_canal / d.ingreso_neto)).sum()
        st.metric(f"Margen de {can}", usd(mg2), f"{mg2 - d.margen.sum():+,.0f} US$ vs actual")
        st.caption("Supuesto didáctico: el volumen cae linealmente al subir el precio neto. ¿Qué pasaría con otra elasticidad?")

elif modulo.startswith("🎯"):
    st.title("🎯 Retos y ranking")
    if not equipo: st.warning("Escribe el nombre de tu equipo en la barra lateral para registrar tus respuestas."); 
    full = DF
    cat = full.groupby("categoria").agg(m=("margen", "sum"), n=("ingreso_neto", "sum")); cat["p"] = cat.m / cat.n
    can = full.groupby("canal").agg(m=("margen", "sum"), n=("ingreso_neto", "sum")); can["p"] = can.m / can.n
    pro = full.groupby("producto").agg(u=("cantidad", "sum"), m=("margen", "sum"), n=("ingreso_neto", "sum")); pro["p"] = pro.m / pro.n
    top_u, top_m = set(pro.u.nlargest(3).index), set(pro.p.nlargest(3).index)
    CLAVE = {1: {full.groupby("pais").ingreso_neto.sum().idxmax()}, 2: {cat.m.idxmax(), cat.p.idxmax()},
             3: {can.n.idxmax() + " | " + can.p.idxmax()}, 4: {"No" if top_u != top_m else "Sí"}}
    PREG = {1: "¿Qué país genera mayores ingresos?", 2: "¿Qué categoría tiene mayor margen de contribución?",
            3: "¿Qué canal vende más y cuál es más rentable?", 4: "¿Los productos más vendidos son necesariamente los más rentables?"}
    EJ = {1: "Ejemplo: “Chile aporta US$ 310 mil, pero …, por lo tanto el comité debería …”",
          2: "Ejemplo: “Hogar tiene alto margen en soles pero …, lo que significa que …”",
          3: "Ejemplo: “El canal que más factura no es el que más deja: entre descuentos y comisión …, recomiendo …”",
          4: "Ejemplo: “Vender más unidades no garantiza ganar más: el producto estrella deja solo … de margen, así que …”"}
    CLAVES = ["porque", "por qué", "decisi", "recomiend", "margen", "descuento", "riesgo", "estrategia", "cliente", "costo", "precio", "rentab", "impacto", "oportunidad"]
    def puntuar(t):
        t = t.lower(); p = min(len(t.split()) / 40, 1) * 5 + min(sum(c in t for c in CLAVES) / 5, 1) * 5
        return round(p)
    st.caption("Los retos se corrigen con el dataset completo; usa los filtros para explorar. Puntos: dato correcto 10 + interpretación hasta 10 "
               "(rúbrica: extensión ≥ 40 palabras, conceptos de negocio y una decisión concreta). Cada reto se responde una sola vez: "
               "la respuesta queda registrada, sea correcta o no, y se comenta en la puesta en común.")
    con = db()
    for r in range(1, 5):
        prev = con.execute("select respuesta, p_dato, p_interp from respuestas where equipo=? and reto=?", (equipo, r)).fetchone() if equipo else None
        with st.expander(f"Reto {r}: {PREG[r]}" + (" · ✔ registrado" if prev else ""), expanded=(r == 1 and not prev)):
            if prev:
                st.info(f"Respuesta registrada: **{prev[0]}** · {'✅ dato correcto' if prev[1] > 0 else '❌ dato no coincide'} · {prev[1] + prev[2]} pts")
                continue
            with st.form(f"form{r}"):
                resp = ""
                if r == 1: resp = st.selectbox("Tu respuesta", sorted(DF.pais.unique()), key="r1")
                if r == 2: resp = st.selectbox("Tu respuesta", sorted(DF.categoria.unique()), key="r2")
                if r == 3:
                    c1, c2 = st.columns(2); a = c1.selectbox("Canal que MÁS vende", sorted(DF.canal.unique()), key="r3a")
                    b = c2.selectbox("Canal MÁS rentable (margen %)", sorted(DF.canal.unique()), key="r3b"); resp = f"{a} | {b}"
                if r == 4: resp = st.radio("Tu respuesta", ["Sí", "No"], horizontal=True, key="r4")
                txt = st.text_area("¿Qué significa para el negocio? ¿Qué decisión tomarías?", key=f"t{r}", placeholder=EJ[r], height=120)
                enviado = st.form_submit_button("Registrar respuesta")
            if enviado:
                if not equipo: st.error("Primero escribe tu equipo en la barra lateral y pulsa «Registrar equipo».")
                elif len(txt.split()) < 8: st.error(f"Tu explicación tiene {len(txt.split())} palabras; escribe al menos 8.")
                else:
                    ok = resp in CLAVE[r]; pd_ = 10 if ok else 0; pi = puntuar(txt)
                    with con: con.execute("insert or replace into respuestas values(?,?,?,?,?,?,?,?)", (equipo, r, resp, txt, pd_, pi, 1, datetime.now().isoformat(timespec="seconds")))
                    if ok: st.success(f"✅ Respuesta registrada: dato correcto. +{pd_ + pi} pts")
                    else: st.warning(f"Respuesta registrada: el dato no coincide con el dataset completo. +{pi} pts por tu interpretación. Lo revisaremos en la puesta en común.")
    st.divider(); st.subheader("🏅 Ranking del laboratorio")
    lb = pd.read_sql("select equipo, sum(p_dato+p_interp) as puntos, count(*) as retos from respuestas where reto<=4 group by equipo order by puntos desc", db())
    if lb.empty: st.info("Aún no hay respuestas.")
    else:
        lb.insert(0, "🏆", ["🥇", "🥈", "🥉"] + [""] * max(len(lb) - 3, 0) if len(lb) > 3 else ["🥇", "🥈", "🥉"][:len(lb)])
        st.dataframe(lb, hide_index=True, **ANCHO)
    if equipo:
        mine = pd.read_sql("select * from respuestas where equipo=? and reto<=4", db(), params=(equipo,))
        B = []
        if {"📈 Dashboard ejecutivo", "🌎 Análisis comercial", "👥 Análisis de clientes", "💰 Análisis de rentabilidad"} <= st.session_state.visitados: B.append("🧭 Explorador total")
        if (mine.p_dato > 0).sum() >= 3: B.append("🎯 Analista certero")
        if (mine.p_interp >= 8).any(): B.append("🧠 Pensamiento crítico")
        st.markdown("**Tus insignias:** " + (" ".join(f"<span class='badge'>{x}</span>" for x in B) or "aún ninguna — ¡explora y responde!"), unsafe_allow_html=True)

else:
    st.title("🔐 Panel docente")
    try: clave = st.secrets.get("INSTRUCTOR_PASSWORD", "analista2026")
    except Exception: clave = "analista2026"
    if st.text_input("Contraseña", type="password") != clave:
        st.info("Ingresa la contraseña del docente."); st.stop()
    c1, c2 = st.columns(2)
    if c1.button("▶️ Iniciar laboratorio (60 min)"): set_cfg("inicio", datetime.now().isoformat()); st.rerun()
    if c2.button("⏹️ Reiniciar reloj"):
        with db() as c: c.execute("delete from config where k='inicio'")
        st.rerun()
    res = pd.read_sql("select * from respuestas where reto<=4 order by ts", db())
    res["resultado"] = np.where(res.p_dato > 0, "Correcta", "Incorrecta")
    st.subheader("📝 Respuestas de los equipos"); st.dataframe(res, **ANCHO, hide_index=True)
    st.download_button("⬇️ Exportar respuestas (CSV)", res.to_csv(index=False).encode("utf-8-sig"), "respuestas_laboratorio.csv", "text/csv")
    st.download_button("⬇️ Exportar dataset (CSV)", DF.to_csv(index=False).encode("utf-8-sig"), "dataset_analista_negocios.csv", "text/csv")
    st.subheader("🗝️ Clave de respuestas (dataset completo)")
    k1 = DF.groupby("pais").ingreso_neto.sum().sort_values(ascending=False)
    k2 = DF.groupby("categoria").apply(lambda d: pd.Series({"margen_US$": d.margen.sum(), "margen_%": d.margen.sum() / d.ingreso_neto.sum()}))
    k3 = DF.groupby("canal").apply(lambda d: pd.Series({"ingreso_US$": d.ingreso_neto.sum(), "margen_%": d.margen.sum() / d.ingreso_neto.sum()}))
    k4 = DF.groupby("producto").apply(lambda d: pd.Series({"unidades": d.cantidad.sum(), "margen_%": d.margen.sum() / d.ingreso_neto.sum()})).sort_values("unidades", ascending=False)
    a, b = st.columns(2)
    a.write("**Reto 1 · Ingreso por país**"); a.dataframe(k1.round(0)); a.write("**Reto 3 · Canales**"); a.dataframe(k3.round(3))
    b.write("**Reto 2 · Categorías**"); b.dataframe(k2.round(3)); b.write("**Reto 4 · Productos (por unidades)**"); b.dataframe(k4.round(3))
    st.markdown("**Preguntas para la puesta en común:** ¿Qué canal recortarías primero y por qué? ¿Qué cliente protegerías? ¿Qué descuento renegociarías y qué riesgo asumirías? ¿Qué hallazgo cambiaría una decisión comercial?")
