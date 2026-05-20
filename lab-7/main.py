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
    # Лабораторная работа №7
    ## Численное интегрирование
    ### Вариант 23

    Нужно вычислить интеграл:

    \[
    I = \int_0^3 \left(2x^2 - 1 + \sqrt{x}\right)dx
    \]

    при числе интервалов:

    \[
    n = 6
    \]

    Используем методы:

    1. Метод левых прямоугольников
    2. Метод правых прямоугольников
    3. Метод трапеций
    4. Метод парабол Симпсона
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
    a = 0
    b = 3
    n = 6

    h = (b - a) / n

    def f(x):
        return 2 * x**2 - 1 + np.sqrt(x)

    x_nodes = np.linspace(a, b, n + 1)
    y_nodes = f(x_nodes)

    df_nodes = pd.DataFrame({
        "i": range(n + 1),
        "x_i": x_nodes,
        "f(x_i)": y_nodes
    })

    df_nodes
    return a, b, f, h, x_nodes, y_nodes


@app.cell
def _(h, np, y_nodes):
    I_left = h * np.sum(y_nodes[:-1])

    I_left
    return (I_left,)


@app.cell
def _(h, np, y_nodes):
    I_right = h * np.sum(y_nodes[1:])

    I_right
    return (I_right,)


@app.cell
def _(h, np, y_nodes):
    I_trapezoid = h * (
        (y_nodes[0] + y_nodes[-1]) / 2 + np.sum(y_nodes[1:-1])
    )

    I_trapezoid
    return (I_trapezoid,)


@app.cell
def _(h, np, y_nodes):
    I_simpson = h / 3 * (
        y_nodes[0]
        + y_nodes[-1]
        + 4 * np.sum(y_nodes[1:-1:2])
        + 2 * np.sum(y_nodes[2:-1:2])
    )

    I_simpson
    return (I_simpson,)


@app.cell
def _(I_left, I_right, I_simpson, I_trapezoid, pd):
    df_results = pd.DataFrame({
        "Метод": [
            "Левые прямоугольники",
            "Правые прямоугольники",
            "Трапеции",
            "Симпсон"
        ],
        "Значение интеграла": [
            I_left,
            I_right,
            I_trapezoid,
            I_simpson
        ],
        "Округление до 0.001": [
            round(I_left, 3),
            round(I_right, 3),
            round(I_trapezoid, 3),
            round(I_simpson, 3)
        ]
    })

    df_results
    return


@app.cell
def _(I_left, I_right, I_simpson, I_trapezoid, b, pd):
    I_exact = (2 * b**3) / 3 - b + (2 * b**1.5) / 3

    df_check = pd.DataFrame({
        "Метод": [
            "Левые прямоугольники",
            "Правые прямоугольники",
            "Трапеции",
            "Симпсон",
            "Точное значение"
        ],
        "Значение": [
            I_left,
            I_right,
            I_trapezoid,
            I_simpson,
            I_exact
        ],
        "Абсолютная ошибка": [
            abs(I_left - I_exact),
            abs(I_right - I_exact),
            abs(I_trapezoid - I_exact),
            abs(I_simpson - I_exact),
            0
        ]
    })

    df_check
    return (I_exact,)


@app.cell
def _(a, b, f, np, plt, x_nodes, y_nodes):
    x_plot = np.linspace(a, b, 300)
    y_plot = f(x_plot)

    plt.figure(figsize=(8, 5))

    plt.plot(x_plot, y_plot, label=r"$f(x)=2x^2-1+\sqrt{x}$")
    plt.scatter(x_nodes, y_nodes, label="Узлы разбиения")

    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("График подынтегральной функции")
    plt.grid(True)
    plt.legend()

    plt.show()
    return


@app.cell(hide_code=True)
def _(I_exact, I_left, I_right, I_simpson, I_trapezoid, mo):
    mo.md(f"""
    ## Итоговый ответ

    Для варианта 23:

    \\[
    I = \\int_0^3 \\left(2x^2 - 1 + \\sqrt{{x}}\\right)dx
    \\]

    При \\(n=6\\), \\(h=0.5\\):

    Метод левых прямоугольников:

    \\[
    I_{{лев}} \\approx {I_left:.3f}
    \\]

    Метод правых прямоугольников:

    \\[
    I_{{прав}} \\approx {I_right:.3f}
    \\]

    Метод трапеций:

    \\[
    I_{{тр}} \\approx {I_trapezoid:.3f}
    \\]

    Метод Симпсона:

    \\[
    I_{{С}} \\approx {I_simpson:.3f}
    \\]

    Точное значение для проверки:

    \\[
    I_{{точн}} \\approx {I_exact:.3f}
    \\]

    Наиболее близким к точному значению оказался метод Симпсона:

    \\[
    \\boxed{{I \\approx {I_simpson:.3f}}}
    \\]
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
