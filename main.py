#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Веб-додаток: Модель Леслі (4 вікові класи) — Dash + Plotly
#
# Запуск:
#   pip install dash==2.* plotly numpy pandas
#   python app.py
# Потім відкрий у браузері адресу, яку покаже консоль (зазвичай http://127.0.0.1:8050)

import numpy as np
import pandas as pd
from numpy.linalg import eig

from dash import Dash, dcc, html, Input, Output, State, callback_context
import plotly.graph_objects as go

# ---------- Модель ----------

def build_leslie_matrix(f, s):
    f = np.asarray(f, dtype=float)
    s = np.asarray(s, dtype=float)
    if f.size != 4 or s.size != 4:
        raise ValueError("Очікується f і s довжини 4.")
    A = np.zeros((4, 4), dtype=float)
    A[0, :] = f
    A[1, 0] = s[0]
    A[2, 1] = s[1]
    A[3, 2] = s[2]
    # s[3] для переходу з останнього класу далі у цій постановці ігноруємо (зазвичай 0)
    return A

def simulate_leslie(A, n0, T):
    n0 = np.asarray(n0, dtype=float)
    traj = np.zeros((T + 1, 4), dtype=float)
    traj[0, :] = n0
    totals = np.zeros(T + 1, dtype=float)
    totals[0] = n0.sum()
    for t in range(T):
        traj[t + 1, :] = A @ traj[t, :]
        totals[t + 1] = traj[t + 1, :].sum()
    years = np.arange(T + 1)
    return years, traj, totals

def stability_metrics(A, totals):
    vals, vecs = eig(A)
    idx = np.argmax(np.abs(vals))
    lam = float(np.real(vals[idx]))
    v = np.real(vecs[:, idx])
    v = np.abs(v)
    stable = v / v.sum()
    ratios = totals[1:] / np.where(totals[:-1] == 0, np.nan, totals[:-1])
    mean_growth = float(np.nanmean(ratios))
    return lam, stable, mean_growth

# ---------- Dash UI ----------

app = Dash(__name__)
app.title = "Модель Леслі — Dash"

def number_input(id_, value, step=0.1, min_=0, max_=None):
    return dcc.Input(
        id=id_, type="number", value=value, step=step,
        min=min_, max=max_, debounce=True, style={"width":"100%","padding":"6px"}
    )

controls = html.Div([
    html.H2("Модель Леслі (4 вікові групи)"),
    html.P("Задай параметри та натисни «Обчислити». За замовчуванням — дані з Варіанта 1."),
    html.Div([
        html.Div([
            html.H4("Початкові чисельності n₀"),
            html.Div(["0→1 років", number_input("n0_0", 100, step=1)]),
            html.Div(["1→2 роки", number_input("n0_1", 65, step=1)]),
            html.Div(["2→3 роки", number_input("n0_2", 78, step=1)]),
            html.Div(["3→<4 роки", number_input("n0_3", 140, step=1)]),
        ], style={"flex":"1","gap":"6px","display":"grid"}),
        html.Div([
            html.H4("Плідність f"),
            html.Div(["0→1 років", number_input("f_0", 0.3, step=0.1)]),
            html.Div(["1→2 роки", number_input("f_1", 2.5, step=0.1)]),
            html.Div(["2→3 роки", number_input("f_2", 3.7, step=0.1)]),
            html.Div(["3→<4 роки", number_input("f_3", 0.3, step=0.1)]),
        ], style={"flex":"1","gap":"6px","display":"grid"}),
        html.Div([
            html.H4("Виживання s"),
            html.Div(["0→1 років", number_input("s_0", 0.5, step=0.05, max_=1)]),
            html.Div(["1→2 роки", number_input("s_1", 0.9, step=0.05, max_=1)]),
            html.Div(["2→3 роки", number_input("s_2", 0.75, step=0.05, max_=1)]),
            html.Div(["3→<4 роки", number_input("s_3", 0.0, step=0.05, max_=1)]),
        ], style={"flex":"1","gap":"6px","display":"grid"}),
        html.Div([
            html.H4("Налаштування моделювання"),
            html.Div(["Тривалість (років)", number_input("years", 50, step=1, min_=1)]),
            html.Button("Обчислити", id="run", n_clicks=0, style={
                "padding":"10px 16px","marginTop":"8px","borderRadius":"10px","cursor":"pointer"
            }),
            html.Button("Завантажити CSV", id="dl", n_clicks=0, style={
                "padding":"10px 16px","marginTop":"8px","borderRadius":"10px","cursor":"pointer","marginLeft":"8px"
            }),
            dcc.Download(id="download"),
        ], style={"flex":"1"}),
    ], style={"display":"flex","gap":"24px","alignItems":"flex-start","flexWrap":"wrap"}),

    html.Hr(),
    html.Div(id="metrics"),
    dcc.Graph(id="by_age"),
    dcc.Graph(id="total"),
], style={"maxWidth":"1100px","margin":"24px auto","padding":"16px"})

app.layout = html.Main(children=[controls])

# ---------- Callbacks ----------

@app.callback(
    Output("metrics","children"),
    Output("by_age","figure"),
    Output("total","figure"),
    Input("run","n_clicks"),
    State("n0_0","value"), State("n0_1","value"), State("n0_2","value"), State("n0_3","value"),
    State("f_0","value"),  State("f_1","value"),  State("f_2","value"),  State("f_3","value"),
    State("s_0","value"),  State("s_1","value"),  State("s_2","value"),  State("s_3","value"),
    State("years","value"),
    prevent_initial_call=True
)
def run_model(_, n00, n01, n02, n03, f0, f1, f2, f3, s0, s1, s2, s3, years):
    n0 = [n00 or 0, n01 or 0, n02 or 0, n03 or 0]
    f  = [f0 or 0, f1 or 0, f2 or 0, f3 or 0]
    s  = [s0 or 0, s1 or 0, s2 or 0, s3 or 0]
    T = int(years or 50)

    A = build_leslie_matrix(f, s)
    yy, traj, totals = simulate_leslie(A, n0, T)
    lam, stable, mean_g = stability_metrics(A, totals)

    # Метрики
    metrics = html.Div([
        html.H3("Показники стійкості"),
        html.P(f"Домінантне власне значення λ ≈ {lam:.6f} "
               "(λ>1 — зростання, λ≈1 — стаціонарність, λ<1 — спад)"),
        html.P(f"Емпіричний середній темп росту (геометричне N(t+1)/N(t)) ≈ {mean_g:.6f}"),
        html.P("Стабільний віковий розподіл (сума = 1): " +
               ", ".join([f"клас {i}: {stable[i]:.4f}" for i in range(4)])),
    ])

    # Графік по вікових групах
    fig1 = go.Figure()
    labels = ["0→1 р.", "1→2 р.", "2→3 р.", "3→<4 р."]
    for j in range(4):
        fig1.add_trace(go.Scatter(x=yy, y=traj[:, j], mode="lines", name=f"Вікова група {labels[j]}"))
    fig1.update_layout(
        title="Чисельності за віковими групами",
        xaxis_title="Рік моделювання", yaxis_title="Кількість особин",
        template="plotly_white", hovermode="x unified"
    )

    # Графік загальної чисельності
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=yy, y=totals, mode="lines", name="Загальна чисельність"))
    fig2.update_layout(
        title="Динаміка загальної чисельності",
        xaxis_title="Рік моделювання", yaxis_title="Кількість особин",
        template="plotly_white", hovermode="x unified"
    )

    return metrics, fig1, fig2

@app.callback(
    Output("download","data"),
    Input("dl","n_clicks"),
    State("n0_0","value"), State("n0_1","value"), State("n0_2","value"), State("n0_3","value"),
    State("f_0","value"),  State("f_1","value"),  State("f_2","value"),  State("f_3","value"),
    State("s_0","value"),  State("s_1","value"),  State("s_2","value"),  State("s_3","value"),
    State("years","value"),
    prevent_initial_call=True
)
def download_csv(_, n00, n01, n02, n03, f0, f1, f2, f3, s0, s1, s2, s3, years):
    n0 = [n00 or 0, n01 or 0, n02 or 0, n03 or 0]
    f  = [f0 or 0, f1 or 0, f2 or 0, f3 or 0]
    s  = [s0 or 0, s1 or 0, s2 or 0, s3 or 0]
    T = int(years or 50)

    A = build_leslie_matrix(f, s)
    yy, traj, totals = simulate_leslie(A, n0, T)

    df = pd.DataFrame(traj, columns=["Age_0_1", "Age_1_2", "Age_2_3", "Age_3_4minus"])
    df.insert(0, "Year", yy)
    df["Total"] = totals

    return dcc.send_data_frame(df.to_csv, "leslie_simulation.csv", index=False)

if __name__ == "__main__":
    app.run(debug=True)
