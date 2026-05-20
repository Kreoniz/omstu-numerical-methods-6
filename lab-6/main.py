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
    # Лабораторная работа №6
    ## Аппроксимация функции методом наименьших квадратов
    ### Вариант 23

    Дана таблица значений функции:

    \[
    \begin{array}{c|ccccccccc}
    x & 0.092 & 0.448 & 0.803 & 1.159 & 1.511 & 1.871 & 2.227 & 2.582 & 2.938 \\
    \hline
    y & 2.161 & 1.824 & 1.214 & 0.431 & -0.792 & -2.004 & -3.635 & -5.537 & -7.438
    \end{array}
    \]

    Нужно выбрать два аппроксимирующих закона и найти их параметры методом наименьших квадратов.
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
    x = np.array([0.092, 0.448, 0.803, 1.159, 1.511, 1.871, 2.227, 2.582, 2.938])
    y = np.array([2.161, 1.824, 1.214, 0.431, -0.792, -2.004, -3.635, -5.537, -7.438])

    df_data = pd.DataFrame({
        "x": x,
        "y": y
    })

    df_data
    return x, y


@app.cell
def _(plt, x, y):
    plt.figure(figsize=(8, 5))

    plt.scatter(x, y, label="Исходные точки")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Исходные табличные данные")
    plt.grid(True)
    plt.legend()

    plt.show()
    return


@app.cell
def _(np, pd, x, y):
    # Для закона y = ax^2 + bx + c
    # Матрица признаков: [x^2, x, 1]

    Phi_quad = np.column_stack([
        x**2,
        x,
        np.ones_like(x)
    ])

    normal_matrix_quad = Phi_quad.T @ Phi_quad
    normal_vector_quad = Phi_quad.T @ y

    df_normal_quad = pd.DataFrame(
        normal_matrix_quad,
        columns=["a", "b", "c"],
        index=["уравнение a", "уравнение b", "уравнение c"]
    )

    df_normal_quad["Правая часть"] = normal_vector_quad

    df_normal_quad
    return Phi_quad, normal_matrix_quad, normal_vector_quad


@app.cell
def _(Phi_quad, normal_matrix_quad, normal_vector_quad, np, pd, x, y):
    coef_quad = np.linalg.solve(normal_matrix_quad, normal_vector_quad)

    a_quad, b_quad, c_quad = coef_quad

    y_quad = Phi_quad @ coef_quad

    S_quad = np.sum((y - y_quad) ** 2)
    rmse_quad = np.sqrt(S_quad / len(x))

    df_quad_result = pd.DataFrame({
        "Параметр": ["a", "b", "c", "Невязка S", "Среднеквадратическая ошибка"],
        "Значение": [a_quad, b_quad, c_quad, S_quad, rmse_quad]
    })

    df_quad_result
    return S_quad, a_quad, b_quad, c_quad, rmse_quad, y_quad


@app.cell
def _(np, pd, x, y):
    # Для закона y = ax + b * e^(-x) + c
    # Матрица признаков: [x, e^(-x), 1]

    Phi_exp = np.column_stack([
        x,
        np.exp(-x),
        np.ones_like(x)
    ])

    normal_matrix_exp = Phi_exp.T @ Phi_exp
    normal_vector_exp = Phi_exp.T @ y

    df_normal_exp = pd.DataFrame(
        normal_matrix_exp,
        columns=["a", "b", "c"],
        index=["уравнение a", "уравнение b", "уравнение c"]
    )

    df_normal_exp["Правая часть"] = normal_vector_exp

    df_normal_exp
    return Phi_exp, normal_matrix_exp, normal_vector_exp


@app.cell
def _(Phi_exp, normal_matrix_exp, normal_vector_exp, np, pd, x, y):
    coef_exp = np.linalg.solve(normal_matrix_exp, normal_vector_exp)

    a_exp, b_exp, c_exp = coef_exp

    y_exp = Phi_exp @ coef_exp

    S_exp = np.sum((y - y_exp) ** 2)
    rmse_exp = np.sqrt(S_exp / len(x))

    df_exp_result = pd.DataFrame({
        "Параметр": ["a", "b", "c", "Невязка S", "Среднеквадратическая ошибка"],
        "Значение": [a_exp, b_exp, c_exp, S_exp, rmse_exp]
    })

    df_exp_result
    return S_exp, a_exp, b_exp, c_exp, rmse_exp, y_exp


@app.cell
def _(pd, x, y, y_exp, y_quad):
    df_compare_values = pd.DataFrame({
        "x": x,
        "y исходное": y,
        "Квадратичный закон": y_quad,
        "Ошибка 1": y - y_quad,
        "Закон ax + b e^(-x) + c": y_exp,
        "Ошибка 2": y - y_exp
    })

    df_compare_values
    return


@app.cell
def _(a_exp, a_quad, b_exp, b_quad, c_exp, c_quad, np, plt, x, y):
    x_plot = np.linspace(x.min(), x.max(), 300)

    y_quad_plot = a_quad * x_plot**2 + b_quad * x_plot + c_quad
    y_exp_plot = a_exp * x_plot + b_exp * np.exp(-x_plot) + c_exp

    plt.figure(figsize=(9, 6))

    plt.scatter(x, y, label="Исходные точки")
    plt.plot(x_plot, y_quad_plot, label="Квадратичный закон")
    plt.plot(x_plot, y_exp_plot, label="ax + b e^(-x) + c")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Аппроксимация функции методом наименьших квадратов")
    plt.grid(True)
    plt.legend()

    plt.show()
    return


@app.cell
def _(
    S_exp,
    S_quad,
    a_exp,
    a_quad,
    b_exp,
    b_quad,
    c_exp,
    c_quad,
    pd,
    rmse_exp,
    rmse_quad,
):
    df_final_compare = pd.DataFrame({
        "Закон": [
            "y = ax² + bx + c",
            "y = ax + b e^(-x) + c"
        ],
        "Формула": [
            f"y = {a_quad:.3f}x² + ({b_quad:.3f})x + ({c_quad:.3f})",
            f"y = {a_exp:.3f}x + ({b_exp:.3f})e^(-x) + ({c_exp:.3f})"
        ],
        "Невязка S": [
            S_quad,
            S_exp
        ],
        "Среднеквадратическая ошибка": [
            rmse_quad,
            rmse_exp
        ]
    })

    df_final_compare
    return


@app.cell(hide_code=True)
def _(
    S_exp,
    S_quad,
    a_exp,
    a_quad,
    b_exp,
    b_quad,
    c_exp,
    c_quad,
    mo,
    rmse_quad,
):
    best_name = "квадратичный закон" if S_quad < S_exp else "закон ax + b e^(-x) + c"

    mo.md(
        f"""
        ## Итоговый ответ

        Для варианта 23 были выбраны два аппроксимирующих закона:

        1. Квадратичный закон:

        \\[
        \\tilde y_1 = {a_quad:.3f}x^2 + ({b_quad:.3f})x + ({c_quad:.3f})
        \\]

        Невязка:

        \\[
        S_1 = {S_quad:.3f}
        \\]

        2. Закон вида:

        \\[
        \\tilde y_2 = ax + be^{{-x}} + c
        \\]

        После нахождения параметров:

        \\[
        \\tilde y_2 = {a_exp:.3f}x + ({b_exp:.3f})e^{{-x}} + ({c_exp:.3f})
        \\]

        Невязка:

        \\[
        S_2 = {S_exp:.3f}
        \\]

        Так как меньшая невязка получилась у первого закона, лучшей аппроксимацией является:

        \\[
        \\boxed{{\\tilde y = {a_quad:.3f}x^2 + ({b_quad:.3f})x + ({c_quad:.3f})}}
        \\]

        Среднеквадратическая ошибка лучшей аппроксимации:

        \\[
        s = {rmse_quad:.3f}
        \\]
        """
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
