import dash
import uuid

import numpy as np
import dash_bootstrap_components as dbc
import plotly.graph_objs as go

from dash import dcc, html, dash_table, Input, Output, State
from scipy.integrate import solve_ivp

from waitress import serve
from multiprocessing import Process, Manager

from solve import EigSolver


DEFAULT_N = 25
DEFAULT_BETA_ARR = np.array([1.0, 0.1, 0.01])
DEFAULT_NUM_EIGEN = 5

DEFAULT_EQUATION_TYPE = "gamma2-nonlinear"
DEFAULT_GAMMA_START = 1e-5
DEFAULT_GAMMA_END = 10.0
DEFAULT_GAMMA_STEP = 0.1
DEFAULT_EIGEN_METHOD = "afev"  # adaptive_find_eigenvalues


def beta_key(beta):
    return f"{float(beta):.8f}"


processes = {}
results = {}

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY]
)

PRIMARY_COLOR = "#0077b6"
SECONDARY_COLOR = "#26A69A"
BACKGROUND_COLOR = "#F7F9FB"
CARD_BG = "#FFFFFF"
TEXT_COLOR = "#23272F"
ACCENT_COLOR = "#23272F"
WARNING_COLOR = "#FFA726"
ERROR_COLOR = "#EF5350"

CONTENT_STYLE = {
    "flex": "1 1 0%",
    "padding": "2rem 1rem",
    "color": TEXT_COLOR,
    "text-align": "center",
    "backgroundColor": BACKGROUND_COLOR,
    "minHeight": "100vh"
}

SIDEBAR_STYLE = {
    "position": "sticky",
    "top": "2rem",
    "alignSelf": "flex-start",
    "width": "15rem",
    "padding": "1rem 0.8rem",
    "background-color": CARD_BG,
    "borderRadius": "14px",
    "boxShadow": "0 2px 8px rgba(0,0,0,0.07)",
    "border": f"1px solid {PRIMARY_COLOR}",
    "margin": "2rem 2rem 2rem 1rem",
    "zIndex": 1000,
    "text-align": "center",
    "minWidth": "240px",
    "maxWidth": "300px",
    "color": TEXT_COLOR,
    "maxHeight": "90vh",
    "overflowY": "auto"
}

CARD_STYLE = {
    "borderRadius": "10px",
    "marginBottom": "0.7rem",
    "background": CARD_BG,
    "boxShadow": "0 1px 4px rgba(0,0,0,0.07)",
    "padding": "0.5rem",
    "text-align": "center",
    "color": TEXT_COLOR
}

INPUT_STYLE = {
    "borderRadius": "7px",
    "background": "#F7F9FB",
    "border": f"1px solid {PRIMARY_COLOR}",
    "fontSize": "1em",
    "padding": "0.35rem 0.7rem",
    "color": TEXT_COLOR
}

BUTTON_STYLE = {
    "width": "100%",
    "backgroundColor": PRIMARY_COLOR,
    "borderColor": PRIMARY_COLOR,
    "color": "#ffffff",
    "borderRadius": "8px",
    "fontSize": "1em",
    "padding": "0.6rem 0",
    "text-align": "center"
}

sidebar = html.Div(
    [
        html.H2(
            "Параметры",
            className="display-6",
            style={
                "fontSize": "1.15em",
                "marginBottom": "0.7em",
                "fontWeight": "bold",
                "textAlign": "center"
            }
        ),
        html.Hr(style={"margin": "0.5em 0", "borderColor": PRIMARY_COLOR}),
        dbc.CardGroup(
            [
                dbc.Label("Параметр n", style={"fontSize": "0.97em", "textAlign": "center", "width": "100%", "fontWeight": "bold"}),
                dbc.Input(
                    id="n-input",
                    type="number",
                    value=DEFAULT_N,
                    min=1,
                    style=INPUT_STYLE,
                    inputMode="decimal",
                    step="1",
                    placeholder="25"
                ),
            ],
            style=CARD_STYLE
        ),
        dbc.CardGroup(
            [
                dbc.Label("Значения бета", style={"fontSize": "0.97em", "textAlign": "center", "width": "100%", "fontWeight": "bold"}),
                dbc.Input(
                    id="beta-input",
                    value=", ".join(str(x) for x in DEFAULT_BETA_ARR),
                    style=INPUT_STYLE,
                    inputMode="decimal",
                    placeholder="1.0, 0.1, 0.01"
                ),
                html.Small(
                    "Дробные числа вводите через точку (например: 1.5, 0.25)",
                    style={"color": "#888", "display": "block", "marginTop": "0.3em"}
                ),
            ],
            style=CARD_STYLE
        ),
        dbc.CardGroup(
            [
                dbc.Label("Количество СЗ", style={"fontSize": "0.97em", "textAlign": "center", "width": "100%", "fontWeight": "bold"}),
                dbc.Input(
                    id="eigen-count-input",
                    type="number",
                    value=DEFAULT_NUM_EIGEN,
                    min=1,
                    style=INPUT_STYLE,
                    inputMode="decimal",
                    step="1",
                    placeholder="5"
                ),
            ],
            style=CARD_STYLE
        ),
        html.Div(
            [
                dbc.Button(
                    "Дополнительные параметры",
                    id="advanced-toggle",
                    color="secondary",
                    outline=True,
                    style={"margin": "0.5em 0", "width": "100%"}
                ),
                dbc.Collapse(
                    dbc.CardGroup(
                        [
                            html.Div([
                                dbc.Label("Тип уравнения", style={"fontSize": "0.97em", "textAlign": "center", "width": "100%", "fontWeight": "bold"}),
                                dcc.Dropdown(
                                    id="equation-type-dropdown",
                                    options=[
                                        {"label": "gamma2-nlr", "value": "gamma2-nonlinear"},
                                        {"label": "gamma-nlr", "value": "gamma-nonlinear"},
                                        {"label": "gamma2-lr", "value": "gamma2-linear"},
                                        {"label": "gamma-lr", "value": "gamma-linear"}
                                    ],
                                    value=DEFAULT_EQUATION_TYPE,
                                    clearable=False,
                                    style={"marginBottom": "0.5em", "color": "#222"}
                                )
                            ], style={"display": "flex", "flexDirection": "column", "width": "100%"}),

                            html.Div([
                                dbc.Label("Метод поиска СЗ", style={"fontSize": "0.97em", "textAlign": "center", "width": "100%", "fontWeight": "bold"}),
                                dcc.Dropdown(
                                    id="eigen-method-dropdown",
                                    options=[
                                        {"label": "Адаптивный (afev)", "value": "afev"},
                                        {"label": "Равномерный (fev)", "value": "fev"},
                                        {"label": "Быстрый (ffev)", "value": "ffev"}
                                    ],
                                    value=DEFAULT_EIGEN_METHOD,
                                    clearable=False,
                                    style={"marginBottom": "0.5em", "color": "#222"}
                                )
                            ], style={"display": "flex", "flexDirection": "column", "width": "100%"}),
                            # --- END NEW ---
                            dbc.Label("gamma_start", style={"fontSize": "0.97em", "textAlign": "center", "width": "100%", "fontWeight": "bold"}),
                            dbc.Input(
                                id="gamma-start-input",
                                type="number",
                                value=DEFAULT_GAMMA_START,
                                style=INPUT_STYLE,
                                inputMode="decimal",
                                step="any",
                                placeholder="1.0"
                            ),
                            dbc.Label("gamma_end", style={"fontSize": "0.97em", "textAlign": "center", "width": "100%", "fontWeight": "bold"}),
                            dbc.Input(
                                id="gamma-end-input",
                                type="number",
                                value=DEFAULT_GAMMA_END,
                                style=INPUT_STYLE,
                                inputMode="decimal",
                                step="any",
                                placeholder="10.0"
                            ),
                            dbc.Label("gamma_step", style={"fontSize": "0.97em", "textAlign": "center", "width": "100%", "fontWeight": "bold"}),
                            dbc.Input(
                                id="gamma-step-input",
                                type="number",
                                value=DEFAULT_GAMMA_STEP,
                                style=INPUT_STYLE,
                                inputMode="decimal",
                                step="any",
                                placeholder="1.0"
                            ),
                            html.Small(
                                "Дробные числа вводите через точку/запятую/нотацию (например: 0.01 или 0,01 или 1e-2)",
                                style={"color": "#888", "display": "block", "marginTop": "0.3em"}
                            ),
                        ],
                        style={**CARD_STYLE, "marginBottom": "0"}
                    ),
                    id="advanced-collapse",
                    is_open=False
                )
            ]
        ),
        dbc.Button(
            "Рассчитать",
            id="calculate-button",
            color="primary",
            style=BUTTON_STYLE
        ),
        dbc.Button(
            "Остановить вычисления",
            id="stop-calc-button",
            color="danger",
            style={**BUTTON_STYLE, "backgroundColor": SECONDARY_COLOR, "borderColor": SECONDARY_COLOR, "marginTop": "0.5em"}
        ),
    ],
    style=SIDEBAR_STYLE
)

content = html.Div(
    [
        dcc.Store(id="session-id", data=str(uuid.uuid4())),
        html.H1("Задача на собственные значения", style={"color": ACCENT_COLOR}),
        html.Hr(),
        html.Div(
            [
                html.H2("Определение задачи", style={"color": ACCENT_COLOR}),
                html.P(
                    dcc.Markdown(
                        r"""
                        ** Уравнение **

                        $$
                        u''(x) = -(\epsilon(x) - \gamma^2)u(x) - \alpha|u(x)|^p,
                        $$

                        **где**

                        $$
                        x \in [0, h], h = 5 + (-1)^n \frac{1}{n+3},
                        $$
                        $$
                        \alpha = \beta + \frac{1}{n^2 + 10},
                        $$
                        $$
                        \epsilon(x) = \left(1 + \frac{1}{n}\right)x + \frac{x^2}{n^3 + 1},
                        $$
                        $$
                        p = 3 + \frac{1}{n+1}.
                        $$
                        
                        **Начальные условия:**
                        
                        $$
                        u(0) = 0,
                        $$
                        $$
                        u'(0) = \frac{n+2}{n} + \frac{n}{n^2 + 1}.
                        $$
                        
                        **Граничные условия:**

                        $$
                        u(h) = 0.
                        $$
                        """,
                        mathjax=True,
                        style={"font-size": "1.1em"}
                    )
                )
            ],
            style={"margin-bottom": "2rem"}
        ),
        html.Hr(id="result-divider", style={"display": "none", "borderColor": PRIMARY_COLOR}),
        html.Div(
            id="tabs-section",
            style={"display": "none"}
        ),
        dcc.Store(id="beta-calc-params", data={}),
        dcc.Store(id="task-status", data="idle"),
        dcc.Store(id="task-result", data=None),
        dcc.Interval(id="beta-poll-interval", interval=1000, n_intervals=0, disabled=True)
    ],
    style=CONTENT_STYLE
)

app.layout = html.Div(
    [sidebar, content],
    style={
        "display": "flex",
        "flexDirection": "row",
        "backgroundColor": BACKGROUND_COLOR,
        "minHeight": "100vh"
    }
)

@app.callback(
    Output("advanced-collapse", "is_open"),
    Input("advanced-toggle", "n_clicks"),
    State("advanced-collapse", "is_open"),
    prevent_initial_call=True
)
def toggle_advanced(n, is_open):
    if n:
        return not is_open
    return is_open

def run_solver(params, return_dict):
    try:
        beta_arr = params["beta_arr"]
        n = params["n"]
        num_eigen = params["num_eigen"]
        equation_type = params["equation_type"]
        gamma_start = params["gamma_start"]
        gamma_end = params["gamma_end"]
        gamma_step = params["gamma_step"]
        eigen_method = params.get("eigen_method", "afev")

        beta_results = {}
        for beta in beta_arr:
            solver = EigSolver(n=n, beta=beta, equation_type=equation_type)
            if eigen_method == "afev":
                eigen_gammas = solver.adaptive_find_eigenvalues(
                    eigen_count=num_eigen,
                    gamma_start=gamma_start,
                    gamma_end=gamma_end,
                    gamma_step=gamma_step,
                    verbose=False
                )
            elif eigen_method == "fev":
                eigen_gammas = solver.find_eigenvalues(
                    eigen_count=num_eigen,
                    gamma_start=gamma_start,
                    gamma_end=gamma_end,
                    gamma_count=int((gamma_end - gamma_start) / gamma_step) + 1,
                    verbose=False
                )
            elif eigen_method == "ffev":
                eigen_gammas = solver.fast_find_eigenvalues(
                    eigen_count=num_eigen,
                    gamma_start=gamma_start,
                    gamma_end=gamma_end,
                    gamma_step=gamma_step,
                    verbose=False
                )
            
            h = solver.h
            x = np.linspace(0, h, 300)
            eigenfunctions = []
            table_rows = []
            for i, gamma in enumerate(eigen_gammas):
                if i > num_eigen:
                    break
                sol = solve_ivp(
                    lambda x_, y: solver.rhs_equation(x_, y, gamma),
                    [0, h],
                    [solver.u_0, solver.du_0],
                    t_eval=x,
                    dense_output=True
                )
                u = sol.y[0] if sol.success else np.full_like(x, np.nan)
                uh = u[-1] if sol.success else np.nan
                eigenfunctions.append((gamma, u))

                u0_val = u[0]
                du0_actual = sol.y[1][0] if sol.success else np.nan

                gamma_str = f"{gamma:.8f}" + (" ✅" if abs(gamma) < 1e-3 else "")
                residual_val = abs(uh)
                residual_str = f"{residual_val:.2e}" + (" ✅" if residual_val < 1e-3 else "")
                u0_str = f"{u0_val:.4f}" + (" ✅" if abs(u0_val) < 1e-3 else "")
                du0_str = f"{du0_actual:.4f}" + (" ✅" if abs(du0_actual - solver.du_0) < 1e-3 else "")
                uh_str = f"{uh:.4f}" + (" ✅" if abs(uh) < 1e-3 else "")

                table_rows.append({
                    "index": i + 1,
                    "gamma": gamma_str,
                    "residual": residual_str,
                    "u0": u0_str,
                    "du0": du0_str,
                    "uh": uh_str
                })

            fig = go.Figure()
            for i, (gamma, u) in enumerate(eigenfunctions):
                fig.add_trace(go.Scatter(
                    x=x, y=u, mode='lines',
                    name=f"γ={gamma:.5f}",
                    line=dict(width=2)
                ))
            fig.update_layout(
                title=f"Собственные функции для β={beta}",
                xaxis_title="x",
                yaxis_title="u(x)",
                template="plotly_white",
                legend_title="γ",
                plot_bgcolor=CARD_BG,
                paper_bgcolor=CARD_BG,
                font=dict(color=TEXT_COLOR)
            )

            warning = None
            if len(eigen_gammas) < num_eigen:
                warning = f"Внимание: найдено {len(eigen_gammas)} СЗ (ожидалось {num_eigen})."
            elif len(eigen_gammas) > num_eigen:
                warning = f"Внимание: найдено {len(eigen_gammas)} СЗ (ожидалось {num_eigen}) — отображаются {len(eigen_gammas)} СЗ"

            beta_results[beta_key(beta)] = {
                "figure": fig.to_dict(),
                "table": table_rows,
                "warning": warning
            }
        return_dict['result'] = beta_results
    except Exception as e:
        return_dict['error'] = str(e)

@app.callback(
    Output("beta-calc-params", "data"),
    Output("task-status", "data"),
    Output("task-result", "data", allow_duplicate=True),
    Output("beta-poll-interval", "disabled"),
    Input("calculate-button", "n_clicks"),
    State("n-input", "value"),
    State("beta-input", "value"),
    State("eigen-count-input", "value"),
    State("equation-type-dropdown", "value"),
    State("eigen-method-dropdown", "value"),
    State("gamma-start-input", "value"),
    State("gamma-end-input", "value"),
    State("gamma-step-input", "value"),
    State("session-id", "data"),
    prevent_initial_call=True
)
def start_task(n_clicks, n, beta_str, num_eigen, equation_type, eigen_method, gamma_start, gamma_end, gamma_step, session_id):
    if session_id in processes:
        p, _ = processes[session_id]
        if p.is_alive():
            p.terminate()
            p.join()
        del processes[session_id]
        if session_id in results:
            del results[session_id]
    try:
        beta_arr = [float(x.strip()) for x in beta_str.split(",")]
    except Exception:
        beta_arr = list(DEFAULT_BETA_ARR)
    params = dict(
        n=n,
        beta_arr=beta_arr,
        num_eigen=num_eigen,
        equation_type=equation_type,
        eigen_method=eigen_method,
        gamma_start=gamma_start,
        gamma_end=gamma_end,
        gamma_step=gamma_step
    )
    manager = Manager()
    return_dict = manager.dict()
    p = Process(target=run_solver, args=(params, return_dict))
    p.start()
    processes[session_id] = (p, return_dict)
    return params, "running", None, False

@app.callback(
    Output("task-status", "data", allow_duplicate=True),
    Output("task-result", "data", allow_duplicate=True),
    Output("beta-poll-interval", "disabled", allow_duplicate=True),
    Input("beta-poll-interval", "n_intervals"),
    State("session-id", "data"),
    State("task-status", "data"),
    prevent_initial_call=True
)
def poll_task(n_intervals, session_id, status):
    if status != "running":
        return dash.no_update, dash.no_update, True
    if session_id not in processes:
        return "idle", None, True
    p, return_dict = processes[session_id]
    if not p.is_alive():
        result = dict(return_dict)
        del processes[session_id]
        return "finished", result, True
    return dash.no_update, dash.no_update, False

@app.callback(
    Output("task-status", "data", allow_duplicate=True),
    Output("beta-poll-interval", "disabled", allow_duplicate=True),
    Input("stop-calc-button", "n_clicks"),
    State("session-id", "data"),
    State("task-status", "data"),
    prevent_initial_call=True
)
def stop_task(n_clicks, session_id, status):
    if status == "running" and session_id in processes:
        p, _ = processes[session_id]
        if p.is_alive():
            p.terminate()
            p.join()
        del processes[session_id]
    return "stopped", True

@app.callback(
    Output("result-divider", "style"),
    Output("tabs-section", "style"),
    Output("tabs-section", "children"),
    Input("task-status", "data"),
    Input("task-result", "data"),
    State("beta-calc-params", "data"),
    prevent_initial_call=True
)
def update_results(status, result, params):
    if status == "running":
        tabs = []
        for beta in params["beta_arr"]:
            key = beta_key(beta)
            tabs.append(
                dcc.Tab(
                    label=f"⏳ β = {beta}",
                    value=f"beta-{key}",
                    children=[
                        html.Div(
                            dbc.Spinner(
                                html.Div("Вычисление..."),
                                color="primary",
                                type="grow",
                                fullscreen=False,
                                size="lg"
                            ),
                            style={"padding": "2em"}
                        )
                    ]
                )
            )
        return {"display": "block"}, {"display": "block"}, dcc.Tabs(
            tabs,
            id="beta-tabs",
            value=f"beta-{beta_key(params['beta_arr'][0])}" if params["beta_arr"] else None
        )
    elif status == "finished":
        if result is None or "result" not in result:
            return {"display": "block"}, {"display": "block"}, html.Div("Нет результата.", style={"color": ERROR_COLOR})
        beta_results = result["result"]
        tabs = []
        for beta in params["beta_arr"]:
            key = beta_key(beta)
            if key not in beta_results:
                tabs.append(
                    dcc.Tab(
                        label=f"β = {beta}",
                        value=f"beta-{key}",
                        children=[
                            html.Div(
                                [
                                    html.H4("Ошибка вычисления"),
                                    html.P("Нет результата для этого β.", style={"color": ERROR_COLOR})
                                ],
                                style={"padding": "2em"}
                            )
                        ]
                    )
                )
            else:
                children = []
                if beta_results[key].get("warning"):
                    children.append(
                        html.Div(
                            beta_results[key]["warning"],
                            style={"color": WARNING_COLOR, "fontWeight": "bold", "marginBottom": "1em"}
                        )
                    )
                children.extend([
                    dcc.Graph(figure=beta_results[key]["figure"]),
                    html.H5("Таблица параметров для всех γ"),
                    dash_table.DataTable(
                        columns=[
                            {"name": "№", "id": "index"},
                            {"name": "γ", "id": "gamma"},
                            {"name": "Невязка", "id": "residual"},
                            {"name": "u(0)", "id": "u0"},
                            {"name": "u'(0)", "id": "du0"},
                            {"name": "u(h)", "id": "uh"}
                        ],
                        data=beta_results[key]["table"],
                        style_table={"overflowX": "auto", "backgroundColor": CARD_BG},
                        style_cell={"textAlign": "center", "backgroundColor": CARD_BG, "color": TEXT_COLOR},
                        style_header={"backgroundColor": PRIMARY_COLOR, "color": "#fff"}
                    )
                ])
                tabs.append(
                    dcc.Tab(
                        label=f"β = {beta}",
                        value=f"beta-{key}",
                        children=children
                    )
                )
        return {"display": "block"}, {"display": "block"}, dcc.Tabs(
            tabs,
            id="beta-tabs",
            value=f"beta-{beta_key(params['beta_arr'][0])}" if params["beta_arr"] else None
        )
    elif status == "stopped":
        tabs = []
        for beta in params["beta_arr"]:
            key = beta_key(beta)
            tabs.append(
                dcc.Tab(
                    label=f"β = {beta}",
                    value=f"beta-{key}",
                    children=[
                        html.Div(
                            [
                                html.H4("Вычисления остановлены"),
                                html.P("Вычисления были остановлены пользователем.", style={"color": WARNING_COLOR})
                            ],
                            style={"padding": "2em"}
                        )
                    ]
                )
            )
        return {"display": "block"}, {"display": "block"}, dcc.Tabs(
            tabs,
            id="beta-tabs",
            value=f"beta-{beta_key(params['beta_arr'][0])}" if params["beta_arr"] else None
        )
    else:
        return {"display": "none"}, {"display": "none"}, ""


if __name__ == '__main__':
    serve(app.server, host="0.0.0.0", port=30049, threads=100)
    # app.run(debug=True)
