import marimo

__generated_with = "0.23.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Лабораторная работа №5
    ## Интерполяция таблично заданных функций
    ### Вариант 23

    Дана таблица значений функции:

    \[
    \begin{array}{c|ccccc}
    x & 0.324 & 0.718 & 1.315 & 2.035 & 2.893 \\
    \hline
    y & -2.052 & -1.597 & -0.231 & 2.808 & 8.011
    \end{array}
    \]

    Необходимо:

    1. Построить интерполяционный многочлен Лагранжа.
    2. Вычислить \(L_4(x_1+x_2)\).
    3. Построить таблицы конечных и разделённых разностей.
    4. Построить полином Ньютона и вычислить \(N_4(x_1+x_2)\).
    5. Построить линейный и квадратичный сплайны.
    6. Построить графики.
    """)
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    return np, pd, plt


@app.cell
def _(np, pd):
    x = np.array([0.324, 0.718, 1.315, 2.035, 2.893], dtype=float)
    y = np.array([-2.052, -1.597, -0.231, 2.808, 8.011], dtype=float)

    x_value = x[1] + x[2]

    df_data = pd.DataFrame({
        "x": x,
        "y": y
    })

    df_data
    return x, x_value, y


@app.cell(hide_code=True)
def _(mo, x, x_value):
    mo.md(f"""
    Вычисляем значение интерполяционных многочленов в точке:

    \\[
    x_1+x_2 = {x[1]} + {x[2]} = {x_value:.3f}
    \\]
    """)
    return


@app.function
def lagrange_polynomial(x_nodes, y_nodes, x_point):
    n = len(x_nodes)
    result = 0

    for i in range(n):
        basis = 1

        for j in range(n):
            if i != j:
                basis *= (x_point - x_nodes[j]) / (x_nodes[i] - x_nodes[j])

        result += y_nodes[i] * basis

    return result


@app.cell
def _(x, x_value, y):
    L_value = lagrange_polynomial(x, y, x_value)

    L_value
    return (L_value,)


@app.cell
def _(np, pd, x, y):
    def finite_difference_table(y_nodes):
        n = len(y_nodes)
        table = np.full((n, n), np.nan)
        table[:, 0] = y_nodes

        for j in range(1, n):
            for i in range(n - j):
                table[i, j] = table[i + 1, j - 1] - table[i, j - 1]

        return table


    finite_table = finite_difference_table(y)

    df_finite = pd.DataFrame(
        finite_table,
        columns=["y", "Δy", "Δ²y", "Δ³y", "Δ⁴y"]
    )

    df_finite.insert(0, "x", x)

    df_finite
    return


@app.cell
def _(np, pd, x, y):
    def divided_difference_table(x_nodes, y_nodes):
        n = len(x_nodes)
        table = np.full((n, n), np.nan)
        table[:, 0] = y_nodes

        for j in range(1, n):
            for i in range(n - j):
                table[i, j] = (
                    table[i + 1, j - 1] - table[i, j - 1]
                ) / (x_nodes[i + j] - x_nodes[i])

        return table


    div_table = divided_difference_table(x, y)

    df_divided = pd.DataFrame(
        div_table,
        columns=[
            "y",
            "1-го порядка",
            "2-го порядка",
            "3-го порядка",
            "4-го порядка"
        ]
    )

    df_divided.insert(0, "x", x)

    df_divided
    return (div_table,)


@app.function
def newton_polynomial(x_nodes, divided_table, x_point):
    n = len(x_nodes)
    result = divided_table[0, 0]
    product = 1

    for i in range(1, n):
        product *= x_point - x_nodes[i - 1]
        result += divided_table[0, i] * product

    return result


@app.cell
def _(div_table, x, x_value):
    N_value = newton_polynomial(x, div_table, x_value)

    N_value
    return (N_value,)


@app.cell
def _(pd, x, y):
    linear_spline_coeffs = []

    for i in range(len(x) - 1):
        a = (y[i + 1] - y[i]) / (x[i + 1] - x[i])
        b = y[i] - a * x[i]

        linear_spline_coeffs.append({
            "Интервал": f"[{x[i]}, {x[i + 1]}]",
            "a": a,
            "b": b,
            "Формула": f"{a:.6f}x + ({b:.6f})"
        })

    df_linear_spline = pd.DataFrame(linear_spline_coeffs)

    df_linear_spline
    return


@app.cell
def _(np, pd, x, y):
    quad_1 = np.polyfit(x[:3], y[:3], 2)
    quad_2 = np.polyfit(x[2:], y[2:], 2)

    df_quadratic_spline = pd.DataFrame({
        "Интервал": [
            f"[{x[0]}, {x[2]}]",
            f"[{x[2]}, {x[4]}]"
        ],
        "a": [quad_1[0], quad_2[0]],
        "b": [quad_1[1], quad_2[1]],
        "c": [quad_1[2], quad_2[2]],
        "Формула": [
            f"{quad_1[0]:.6f}x² + ({quad_1[1]:.6f})x + ({quad_1[2]:.6f})",
            f"{quad_2[0]:.6f}x² + ({quad_2[1]:.6f})x + ({quad_2[2]:.6f})"
        ]
    })

    df_quadratic_spline
    return quad_1, quad_2


@app.cell
def _(div_table, np, plt, quad_1, quad_2, x, y):
    def linear_spline_value(x_point):
        for i in range(len(x) - 1):
            if x[i] <= x_point <= x[i + 1]:
                a = (y[i + 1] - y[i]) / (x[i + 1] - x[i])
                b = y[i] - a * x[i]
                return a * x_point + b

        return np.nan


    def quadratic_spline_value(x_point):
        if x[0] <= x_point <= x[2]:
            return np.polyval(quad_1, x_point)
        elif x[2] <= x_point <= x[4]:
            return np.polyval(quad_2, x_point)
        else:
            return np.nan


    x_plot = np.linspace(x[0], x[-1], 300)

    y_lagrange = np.array([
        lagrange_polynomial(x, y, value)
        for value in x_plot
    ])

    y_newton = np.array([
        newton_polynomial(x, div_table, value)
        for value in x_plot
    ])

    y_linear_spline = np.array([
        linear_spline_value(value)
        for value in x_plot
    ])

    y_quadratic_spline = np.array([
        quadratic_spline_value(value)
        for value in x_plot
    ])

    plt.figure(figsize=(9, 6))

    plt.plot(x_plot, y_lagrange, label="Полином Лагранжа")
    plt.plot(x_plot, y_newton, linestyle="--", label="Полином Ньютона")
    plt.plot(x_plot, y_linear_spline, label="Линейный сплайн")
    plt.plot(x_plot, y_quadratic_spline, label="Квадратичный сплайн")
    plt.scatter(x, y, label="Узлы интерполяции")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Интерполяция таблично заданной функции")
    plt.grid(True)
    plt.legend()

    plt.show()
    return


@app.cell
def _(L_value, N_value, pd, x_value):
    df_result = pd.DataFrame({
        "Величина": [
            "x1 + x2",
            "L4(x1 + x2)",
            "N4(x1 + x2)"
        ],
        "Значение": [
            x_value,
            L_value,
            N_value
        ],
        "Округление до 0.001": [
            round(x_value, 3),
            round(L_value, 3),
            round(N_value, 3)
        ]
    })

    df_result
    return


@app.cell(hide_code=True)
def _(L_value, N_value, mo, x_value):
    mo.md(f"""
    ## Итоговый ответ

    Для варианта 23:

    \\[
    x_1+x_2 = {x_value:.3f}
    \\]

    Значение интерполяционного многочлена Лагранжа:

    \\[
    L_4({x_value:.3f}) = {L_value:.3f}
    \\]

    Значение интерполяционного многочлена Ньютона:

    \\[
    N_4({x_value:.3f}) = {N_value:.3f}
    \\]

    Так как многочлены Лагранжа и Ньютона являются разными формами
    одного и того же интерполяционного полинома, значения совпадают:

    \\[
    \\boxed{{L_4({x_value:.3f}) = N_4({x_value:.3f}) \\approx {L_value:.3f}}}
    \\]
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
