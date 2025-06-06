import dash
import plotly.colors

import numpy as np
import plotly.graph_objects as go
import dash_bootstrap_components as dbc

from dash import dcc, html
from dash.dependencies import Input, Output

from scipy.optimize import root
from plotly.subplots import make_subplots


def check_boundary_conditions(u, tol=1e-5):
    """Проверяет выполнение граничных условий u(0)=0 и u(h)=0"""
    return abs(u[0]) < tol and abs(u[-1]) < tol

def check_derivative_condition(u, dx, n, tol=1e-2):
    """Проверяет дополнительное условие на производную"""
    computed_deriv = (u[1] - u[0]) / dx  # Односторонняя разность
    target_deriv = (n + 2) / n + n/(n ** 2 + 1)
    return abs(computed_deriv - target_deriv) < tol, computed_deriv

def finite_difference_matrix(N, dx):
    # --- Строим матрицу конечных разностей для второй производной с граничными условиями Дирихле ---

    main_diag = -2.0 * np.ones(N)
    off_diag = np.ones(N - 1)

    A = (
        np.diag(main_diag) +
        np.diag(off_diag, k=1) +
        np.diag(off_diag, k=-1)
    )

    return A / dx ** 2

def epsilon_func(x, n):
    # --- Функция эпсилон(x) по формуле задачи ---

    return (1 + 1.0 / n) * x + x ** 2 / (n ** 3 + 1)

def compute_residual(u, gamma, alpha, p, x, n):
    """Вычисление невязки для проверки решения"""
    dx = x[1] - x[0]
    N = len(u)
    A = finite_difference_matrix(N, dx)
    eps = epsilon_func(x, n)
    F = -(eps - gamma**2) * u - alpha * np.abs(u)**p
    return A @ u + F

def nonlinear_eigen_system(uv, alpha, p, x, n):
    u = uv[:-1]
    gamma = uv[-1]
    N = len(u)
    dx = x[1] - x[0]
    
    # Матрица второй производной
    A = finite_difference_matrix(N, dx)
    eps = epsilon_func(x, n)
    
    # Правая часть уравнения
    F = -(eps - gamma**2) * u - alpha * np.abs(u)**p
    
    # ОДУ для внутренних точек
    ode = A @ u + F
    
    # Условие нормировки
    norm = np.sum(u**2) * dx - 1.0
    
    # Условие на производную (учитываем явно)
    deriv_condition = (u[1] - u[0])/dx - ((n + 2)/n + n/(n**2 + 1))
    
    return np.concatenate([ode, [norm, deriv_condition]])

def prepare_initial_guess(mode, x, h, n):
    """Готовим начальное приближение, учитывающее условие на производную"""
    # Решение линеаризованной задачи
    u_lin = np.sin(np.pi * mode * x / h)
    
    # Корректируем наклон в x=0
    target_deriv = (n + 2)/n + n/(n**2 + 1)
    correction = target_deriv * x
    u = u_lin + 0.1 * correction
    
    # Нормировка
    u /= np.sqrt(np.sum(u**2) * (x[1] - x[0]))
    return u

def find_nonlinear_eigenpairs(n, alpha, p, h, num_points=100, num_eigen=5, max_attempts=15):
    x = np.linspace(0, h, num_points)
    x_interior = x[1:-1]
    
    eigenvalues = []
    eigenfunctions = []
    residuals = []
    
    for mode in range(1, num_eigen + 1):
        found = False
        attempt = 0
        
        while not found and attempt < max_attempts:
            # Подготовка начального приближения
            u0 = prepare_initial_guess(mode, x_interior, h, n)
            gamma0 = (np.pi*mode/h)**2 + np.mean(epsilon_func(x_interior, n))
            
            # Решаем систему
            sol = root(
                lambda uv: nonlinear_eigen_system(uv, alpha, p, x_interior, n),
                np.concatenate([u0, [gamma0]]),
                method='lm',  # Метод Левенберга-Марквардта лучше для нелинейных задач
                options={'maxiter': 10000, 'xtol': 1e-10}
            )
            
            if sol.success:
                u = sol.x[:-1]
                gamma = sol.x[-1]
                
                # Проверка производной
                deriv = (u[1] - u[0])/(x[1] - x[0])
                target_deriv = (n + 2)/n + n/(n**2 + 1)
                deriv_error = abs(deriv - target_deriv)
                
                if deriv_error < 1:  # Допустимая погрешность
                    u_full = np.zeros_like(x)
                    u_full[1:-1] = u
                    res = compute_residual(u, gamma, alpha, p, x_interior, n)
                    
                    eigenvalues.append(gamma)
                    eigenfunctions.append(u_full)
                    residuals.append(np.max(np.abs(res)))
                    print(f"Мода {mode}: gamma = {gamma:.6f}, невязка = {np.max(np.abs(res)):.2e}, производная = {deriv:.3f} (требуется {target_deriv:.3f})")\
                    
                    break
                else:
                    print(f"Мода {mode} не удовлетворяет условию на производную (ошибка = {deriv_error:.2e})")
                    eigenvalues.append(np.nan)
                    eigenfunctions.append(np.zeros_like(x))
                    residuals.append(np.nan)

            attempt += 1
    
    return x, eigenvalues, eigenfunctions, residuals

# --- Создаем Dash-приложение ---

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# --- Параметры по умолчанию ---

DEFAULT_N = 25
DEFAULT_BETA_ARR = np.array([1.0, 0.1, 0.01])
DEFAULT_NUM_EIGEN = 5

# --- Стили для layout ---

CONTENT_STYLE = {
    "margin-left": "2rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
    "@media (max-width: 768px)": {
        "margin-left": "1rem",
        "margin-right": "1rem",
        "padding": "1rem 0.5rem"
    }
}

SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "20rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

# --- Элементы управления ---

controls = dbc.Card(
    [
        dbc.CardGroup(
            [
                dbc.Label("Параметр n"),
                dbc.Input(id="n-input", type="number", value=DEFAULT_N, min=1),
            ]
        ),
        dbc.CardGroup(
            [
                dbc.Label("Значения beta (через запятую)"),
                dbc.Input(id="beta-input", value=", ".join(map(str, DEFAULT_BETA_ARR))),
            ]
        ),
        dbc.CardGroup(
            [
                dbc.Label("Количество мод"),
                dbc.Input(id="num-eigen-input", type="number", value=DEFAULT_NUM_EIGEN, min=1),
            ]
        ),
        dbc.Button(
            "Рассчитать",
            id="calculate-button",
            color="primary",
            className="mr-1",
            style={"margin-top": "10px"}
        )
    ],
    body=True,
)

# --- Layout приложения ---

app.layout = dbc.Container(
    [
        html.H1("Интерактивный анализ нелинейной задачи на собственные значения"),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(controls, md=3),
                dbc.Col(
                    [
                        dcc.Loading(
                            id="loading",
                            type="default",
                            children=[
                                html.Div(id="results-container"),
                                dcc.Store(id="results-store"),
                            ],
                        )
                    ],
                    md=9,
                ),
            ]
        ),
    ],
    fluid=True,
)

def validate_inputs(n, beta_input, num_eigen):
    errors = []
    if n < 1:
        errors.append("Параметр n должен быть ≥ 1")
    if num_eigen < 1:
        errors.append("Количество мод должно быть ≥ 1")
    try:
        beta_arr = np.array([float(b.strip()) for b in beta_input.split(",")])
        if len(beta_arr) == 0:
            errors.append("Введите хотя бы одно значение beta")
    except ValueError:
        errors.append("Некорректный формат значений beta")
    
    return errors if errors else None

@app.callback(
    Output("results-store", "data"),
    [Input("calculate-button", "n_clicks")],
    [
        dash.dependencies.State("n-input", "value"),
        dash.dependencies.State("beta-input", "value"),
        dash.dependencies.State("num-eigen-input", "value"),
    ],
)
def calculate_results(n_clicks, n, beta_input, num_eigen):
    if n_clicks is None:
        return dash.no_update
    
    errors = validate_inputs(n, beta_input, num_eigen)
    if errors:
        return {"error": "; ".join(errors)}
    
    try:
        n = int(n)
        beta_arr = np.array([float(b.strip()) for b in beta_input.split(",")])
        num_eigen = int(num_eigen)

        p = 3 + 1.0 / (n + 1)
        h = 5 + (-1) ** n * (1.0 / (n + 3))
        
        alpha_arr = beta_arr + 1.0 / (n ** 2 + 10)
        
        results = []
        for beta, alpha in zip(beta_arr, alpha_arr):
            x, gamma_vals, eigenfunctions, residuals = find_nonlinear_eigenpairs(
                n, alpha, p, h, num_points=250, num_eigen=num_eigen, max_attempts=15
            )
            
            # --- Вычисляем дисперсии и центры масс ---

            dx = x[1] - x[0]

            dispersions = []
            centers = []
            boundary_checks = []
            deriv_checks = []
            computed_derivs = []

            for u in eigenfunctions:
                norm = (u ** 2).sum() * dx
                if norm > 0:
                    prob_density = u ** 2 / norm
                    x_mean = (x * prob_density).sum() * dx
                    var = ((x - x_mean) ** 2 * prob_density).sum() * dx
                    dispersions.append(var)
                    centers.append(x_mean)
                else:
                    dispersions.append(float("nan"))
                    centers.append(float("nan"))
                
                # Проверка граничных условий
                bc_ok = check_boundary_conditions(u)
                boundary_checks.append(bc_ok)
                
                # Проверка условия на производную
                deriv_ok, deriv = check_derivative_condition(u, dx, n)
                deriv_checks.append(deriv_ok)
                computed_derivs.append(deriv)

            results.append({
                "beta": beta,
                "alpha": alpha,
                "gamma_vals": gamma_vals,
                "residuals": residuals,
                "dispersions": dispersions,
                "boundary_checks": boundary_checks,
                "deriv_checks": deriv_checks,
                "computed_derivs": computed_derivs,
                "centers": centers,
                "eigenfunctions": [u.tolist() for u in eigenfunctions],
                "x": x.tolist(),
                "num_eigen": num_eigen,
                "p": p,
                "h": h,
                "n": n
            })
        
        return results
    except Exception as e:
        return {"error": str(e)}

@app.callback(
    Output("results-container", "children"),
    [Input("results-store", "data")]
)
def update_results(data):
    if not data or "error" in data:
        return html.Div("Ошибка в расчетах. Проверьте входные параметры.", className="alert alert-danger")
    
    results = data
    children = []
    
    # --- Описание задачи с LaTeX (mathjax) ---

    children.append(
        dbc.Card(
            dbc.CardBody(
                [
                    html.H2("Описание задачи", className="card-title"),
                    html.Div(
                        [
                            dcc.Markdown(
                                r"""
                                **Уравнение:**

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
                                
                                **Граничные условия:**
                                
                                $$
                                u(0) = 0, u(h) = 0.
                                $$
                                
                                **Дополнительные условия:**

                                $$
                                u'(0) = \frac{n+2}{n} + \frac{n}{n^2 + 1}.
                                $$
                                """,
                                mathjax=True,
                                style={"font-size": "1.1em"}
                            ),
                        ]
                    ),
                ]
            ),
            className="mb-4"
        )
    )
    
    # --- Результаты для каждого набора параметров ---

    for i, run in enumerate(results):
        # --- Создаем графики для текущего набора параметров ---

        fig = create_dashboard_figure(
            np.array(run["x"]),
            run["gamma_vals"],
            [np.array(u) for u in run["eigenfunctions"]],
            run["residuals"],
            run["alpha"],
            run["beta"],
            run["p"],
            run["n"],
            run["h"]
        )
        
        # --- Таблица результатов ---

        table_rows = []
        for j in range(len(run["gamma_vals"])):
            table_rows.append(
                html.Tr([
                    html.Td(j+1),
                    html.Td(f"{run['gamma_vals'][j]:.5f}"),
                    html.Td(f"{run['residuals'][j]:.2e}"),
                    html.Td(f"{run['dispersions'][j]:.3f}"),
                    html.Td(f"{run['centers'][j]:.3f}"),
                ])
            )
        
        conditions_table = []
        for j in range(len(run["gamma_vals"])):
            bc_status = "✓" if run["boundary_checks"][j] else "✗"
            deriv_status = "✓" if run["deriv_checks"][j] else "✗"
            target_deriv = (run["n"] + 2)/run["n"] + run["n"]/(run["n"]**2 + 1)
            
            conditions_table.append(
                html.Tr([
                    html.Td(j+1),
                    html.Td(f"{run['gamma_vals'][j]:.5f}"),
                    html.Td(bc_status, style={"color": "green" if run["boundary_checks"][j] else "red"}),
                    html.Td(deriv_status, style={"color": "green" if run["deriv_checks"][j] else "red"}),
                    html.Td(f"{run['computed_derivs'][j]:.5f}"),
                    html.Td(f"{target_deriv:.5f}"),
                ])
            )
        
        children.append(
            dbc.Card(
                dbc.CardBody(
                    [
                        html.H2(f"Результаты для β = {run['beta']}, α = {run['alpha']:.5f}", className="card-title"),
                        dcc.Graph(figure=fig),
                        html.H3("Таблица результатов"),
                        dbc.Table(
                            [
                                html.Thead(
                                    html.Tr([
                                        html.Th("Мода"),
                                        html.Th("γ (gamma)"),
                                        html.Th("Невязка"),
                                        html.Th("Дисперсия"),
                                        html.Th("Центр масс"),
                                    ])
                                ),
                                html.Tbody(table_rows)
                            ],
                            bordered=True,
                            hover=True,
                            responsive=True,
                            className="mb-4"
                        ),
                        html.H3("Проверка условий"),
                        dbc.Table([
                            html.Thead(
                                html.Tr([
                                    html.Th("Мода"),
                                    html.Th("γ"),
                                    html.Th("Граничные условия"),
                                    html.Th("Производная u'(0)"),
                                    html.Th("Вычисленное значение u'(0)"),
                                    html.Th("Требуемое значение u'(0)"),
                                ])
                            ),
                            html.Tbody(conditions_table)
                        ], bordered=True, hover=True, className="mb-4"),
                        html.P("✓ - условие выполнено, ✗ - не выполнено", className="text-muted"),
                    ]
                ),
                className="mb-4"
            )
        )
    
    # --- Итоговая таблица невязок ---

    residuals_header = [html.Th("Параметры")]
    for j in range(results[0]["num_eigen"] if results else 0):
        residuals_header.append(html.Th(f"Мода {j+1}"))
    
    residuals_rows = []
    for run in results:
        row = [html.Td(f"β={run['beta']}, α={run['alpha']:.5f}")]
        for r in run["residuals"]:
            row.append(html.Td(f"{r:.2e}"))
        residuals_rows.append(html.Tr(row))
    
    children.append(
        dbc.Card(
            dbc.CardBody(
                [
                    html.H2("Итоговая таблица невязок", className="card-title"),
                    dbc.Table(
                        [
                            html.Thead(html.Tr(residuals_header)),
                            html.Tbody(residuals_rows)
                        ],
                        bordered=True,
                        hover=True,
                        responsive=True
                    ),
                ]
            )
        )
    )
    
    return children

def create_dashboard_figure(x, gamma_vals, eigenfunctions, residuals, alpha, beta, p, n, h):
    """Создает фигуру с графиками для Dash-приложения (аналогично plotly_dashboard из оригинального кода)"""
    num_modes = len(eigenfunctions)
    x_interior = x[1:-1]

    # --- Расчёт невязок по x для каждой моды ---

    residuals_x = []
    for i, u in enumerate(eigenfunctions):
        if not np.isnan(residuals[i]):
            u_interior = u[1:-1]
            gamma = gamma_vals[i]
            res_x = compute_residual(u_interior, gamma, alpha, p, x_interior, n)
            residuals_x.append(np.abs(res_x))
        else:
            residuals_x.append(np.full_like(x_interior, np.nan))

    # --- Дисперсии и центры масс ---

    dispersions = []
    centers = []
    for i, u in enumerate(eigenfunctions):
        norm = np.sum(u ** 2) * (x[1] - x[0])
        if norm > 0:
            prob_density = u ** 2 / norm
            x_mean = np.sum(x * prob_density) * (x[1] - x[0])
            var = np.sum((x - x_mean) ** 2 * prob_density) * (x[1] - x[0])
            dispersions.append(var)
            centers.append(x_mean)
        else:
            dispersions.append(np.nan)
            centers.append(np.nan)

    # --- Суммарная плотность вероятности ---

    total_density = np.zeros_like(x)
    for u in eigenfunctions:
        total_density += u ** 2
    dx = x[1] - x[0]
    total_density /= np.sum(total_density) * dx

    # --- Цветовая палитра для мод ---

    palette = plotly.colors.qualitative.Plotly
    while len(palette) < num_modes:
        palette += palette

    # --- Создание subplot с 6 графиками (3x2) ---

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            "Собственные функции",
            "Суммарная плотность вероятности",
            "Невязка по x",
            "Спектр собственных значений (гамма)",
            "Дисперсия по модам",
            "Центр масс по модам"
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.15
    )

    for annotation in fig["layout"]["annotations"]:
        annotation["font"] = dict(size=11)

    # --- Собственные функции ---

    for i, u in enumerate(eigenfunctions):
        fig.add_trace(
            go.Scatter(
                x=x, y=u, mode="lines",
                name=f"Мода {i+1}",
                line=dict(color=palette[i], width=1),
                legendgroup=f"group{i+1}",  # Та же группа, что и у невязки
                hovertemplate=f"x=%{{x:.2f}}<br>u=%{{y:.3f}}<extra>Мода {i+1}</extra>"
            ),
            row=1, col=1
        )

    # --- Суммарная плотность вероятности ---

    fig.add_trace(
        go.Scatter(
            x=x,
            y=total_density,
            mode="lines",
            line=dict(color="orange", width=1),
            name="Суммарная плотность",
            hovertemplate="x=%{x:.2f}<br>Плотность=%{y:.3f}"
        ),
        row=1, col=2
    )

    # --- Невязка по x ---

    for i, res_x in enumerate(residuals_x):
        fig.add_trace(
            go.Scatter(
                x=x_interior, y=res_x, mode="lines",
                name=f"Мода {i+1}",  # То же самое имя, что и у собственных функций
                line=dict(color=palette[i], width=1),
                showlegend=False,  # Разрешаем отображение в легенде
                legendgroup=f"group{i+1}",  # Группируем с соответствующей собственной функцией
                hovertemplate=f"x=%{{x:.2f}}<br>res=%{{y:.2e}}<extra>Мода {i+1}</extra>"
            ),
            row=2, col=1
        )

    # -- Спектр собственных значений (гамма) ---

    fig.add_trace(
        go.Scatter(
            x=np.arange(1, num_modes+1),
            y=gamma_vals,
            mode="lines+markers",
            marker=dict(color="blue", size=6, symbol="circle"),
            line=dict(color="blue", width=1),
            name="Гамма",
            hovertemplate="Мода=%{x}<br>Гамма=%{y:.5f}"
        ),
        row=2, col=2
    )

    # --- Дисперсия по модам ---

    fig.add_trace(
        go.Scatter(
            x=np.arange(1, num_modes+1),
            y=dispersions,
            mode="lines+markers",
            marker=dict(color="green", size=6, symbol="triangle-left"),
            line=dict(color="green", width=1),
            name="Дисперсия",
            hovertemplate="Мода=%{x}<br>Дисперсия=%{y:.3f}"
        ),
        row=3, col=1
    )

    # --- Центр масс по модам ---

    fig.add_trace(
        go.Scatter(
            x=np.arange(1, num_modes+1),
            y=centers,
            mode="lines+markers",
            marker=dict(color="purple", size=6, symbol="square"),
            line=dict(color="purple", width=1),
            name="Центр масс",
            hovertemplate="Мода=%{x}<br>Центр масс=%{y:.3f}"
        ),
        row=3, col=2
    )

    fig.update_layout(
        height=800, width=1000,
        template="plotly_white",
        title_text=f"Параметры: n = {n}, p = {p:.5f}, h = {h:.5f}",
        legend=dict(
            x=1.2, y=1,
            bgcolor="rgba(255,255,255,0.7)",
            bordercolor="black",
            borderwidth=0.5,
            font=dict(size=11)
        ),
        font=dict(family="Arial, sans-serif", size=9),
        margin=dict(l=60, r=20, t=80, b=60),
        plot_bgcolor="rgba(245,245,255,1)",
        paper_bgcolor="rgba(245,245,255,1)"
    )

    # --- Обновляем подписи осей ---

    fig.update_xaxes(title_text="x", row=1, col=1)
    fig.update_xaxes(title_text="x", row=1, col=2)
    fig.update_xaxes(title_text="x", row=2, col=1)
    fig.update_xaxes(title_text="Номер моды", row=2, col=2)
    fig.update_xaxes(title_text="Номер моды", row=3, col=1)
    fig.update_xaxes(title_text="Номер моды", row=3, col=2)
    
    fig.update_yaxes(title_text="u(x)", row=1, col=1)
    fig.update_yaxes(title_text="Суммарная плотность", row=1, col=2)
    fig.update_yaxes(title_text="abs(невязка)", row=2, col=1)
    fig.update_yaxes(title_text="Гамма", row=2, col=2)
    fig.update_yaxes(title_text="Дисперсия", row=3, col=1)
    fig.update_yaxes(title_text="Центр масс x", row=3, col=2)

    return fig


if __name__ == "__main__":
    # app.run(host='0.0.0.0', port=30049, debug=False)
    app.run(debug=True)
