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
    # Лабораторная работа №8
    ## Численное решение обыкновенных дифференциальных уравнений
    ### Вариант 23

    ## Задача Коши

    \[
    y'=-x+y
    \]

    \[
    y(1.1)=0
    \]

    \[
    x\in[1.1;1.6], \quad h=0.1
    \]

    Методы решения:

    1. Метод Эйлера
    2. Модифицированный метод Эйлера
    3. Метод Рунге-Кутта 4-го порядка
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
    a = 1.1
    b = 1.6
    h = 0.1
    y0 = 0

    def f(x, y):
        return -x + y

    x_values = np.round(np.arange(a, b + h / 2, h), 10)

    df_x = pd.DataFrame({
        "i": range(len(x_values)),
        "x_i": x_values
    })

    df_x
    return f, h, x_values, y0


@app.cell
def _(f, h, np, pd, x_values, y0):
    def euler_method(x_values, y0, h):
        y_values = [y0]

        for x in x_values[:-1]:
            y = y_values[-1]
            y_next = y + h * f(x, y)
            y_values.append(y_next)

        return np.array(y_values)


    y_euler = euler_method(x_values, y0, h)

    df_euler = pd.DataFrame({
        "i": range(len(x_values)),
        "x_i": x_values,
        "y_i, метод Эйлера": y_euler
    })

    df_euler
    return (y_euler,)


@app.cell
def _(f, h, np, pd, x_values, y0):
    def modified_euler_method(x_values, y0, h):
        y_values = [y0]
        y_predict_values = [np.nan]

        for x in x_values[:-1]:
            y = y_values[-1]

            y_predict = y + h * f(x, y)
            y_next = y + h * (f(x, y) + f(x + h, y_predict)) / 2

            y_predict_values.append(y_predict)
            y_values.append(y_next)

        return np.array(y_values), np.array(y_predict_values)


    y_modified, y_predict = modified_euler_method(x_values, y0, h)

    df_modified = pd.DataFrame({
        "i": range(len(x_values)),
        "x_i": x_values,
        "прогноз y~": y_predict,
        "y_i, модифицированный Эйлер": y_modified
    })

    df_modified
    return (y_modified,)


@app.cell
def _(f, h, np, pd, x_values, y0):
    def runge_kutta_4_method(x_values, y0, h):
        y_values = [y0]
        rows = []

        for i, x in enumerate(x_values[:-1]):
            y = y_values[-1]

            k0 = h * f(x, y)
            k1 = h * f(x + h / 2, y + k0 / 2)
            k2 = h * f(x + h / 2, y + k1 / 2)
            k3 = h * f(x + h, y + k2)

            y_next = y + (k0 + 2 * k1 + 2 * k2 + k3) / 6

            rows.append({
                "i": i,
                "x_i": x,
                "y_i": y,
                "k0": k0,
                "k1": k1,
                "k2": k2,
                "k3": k3,
                "y_{i+1}": y_next
            })

            y_values.append(y_next)

        return np.array(y_values), pd.DataFrame(rows)


    y_rk4, df_rk4_steps = runge_kutta_4_method(x_values, y0, h)

    df_rk4_steps
    return (y_rk4,)


@app.cell
def _(pd, x_values, y_euler, y_modified, y_rk4):
    df_cauchy_result = pd.DataFrame({
        "x": x_values,
        "Эйлер": y_euler,
        "Модифицированный Эйлер": y_modified,
        "Рунге-Кутта 4": y_rk4
    })

    df_cauchy_result
    return


@app.cell
def _(plt, x_values, y_euler, y_modified, y_rk4):
    plt.figure(figsize=(8, 5))

    plt.plot(x_values, y_euler, marker="o", label="Метод Эйлера")
    plt.plot(x_values, y_modified, marker="o", label="Модифицированный Эйлер")
    plt.plot(x_values, y_rk4, marker="o", label="Рунге-Кутта 4")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Решение задачи Коши")
    plt.grid(True)
    plt.legend()

    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Краевая задача

    Решить методом конечных разностей:

    \[
    y''-xy'-0.6y=4
    \]

    \[
    y'(2)=1
    \]

    \[
    y'(2.3)=3
    \]

    \[
    h=0.1
    \]

    Сетка:

    \[
    x_1=2.0,\quad x_2=2.1,\quad x_3=2.2,\quad x_4=2.3
    \]
    """)
    return


@app.cell
def _(np, pd):
    a_bvp = 2.0
    b_bvp = 2.3
    h_bvp = 0.1

    x_bvp = np.round(np.arange(a_bvp, b_bvp + h_bvp / 2, h_bvp), 10)

    df_bvp_grid = pd.DataFrame({
        "k": range(1, len(x_bvp) + 1),
        "x_k": x_bvp
    })

    df_bvp_grid
    return h_bvp, x_bvp


@app.cell
def _(h_bvp, np, pd, x_bvp):
    m = len(x_bvp)

    A = np.zeros((m, m))
    d = np.zeros(m)

    # Левая граница: y'(2) = 1
    A[0, 0] = -1 / h_bvp
    A[0, 1] = 1 / h_bvp
    d[0] = 1

    # Внутренние узлы
    for i in range(1, m - 1):
        x_i = x_bvp[i]

        # y'' - x y' - 0.6y = 4
        A[i, i - 1] = 1 / h_bvp**2 + x_i / (2 * h_bvp)
        A[i, i] = -2 / h_bvp**2 - 0.6
        A[i, i + 1] = 1 / h_bvp**2 - x_i / (2 * h_bvp)
        d[i] = 4

    # Правая граница: y'(2.3) = 3
    A[-1, -2] = -1 / h_bvp
    A[-1, -1] = 1 / h_bvp
    d[-1] = 3

    df_system = pd.DataFrame(
        A,
        columns=[f"y{i}" for i in range(1, m + 1)],
        index=[f"уравнение {i}" for i in range(1, m + 1)]
    )

    df_system["Правая часть"] = d

    df_system
    return A, d, m


@app.cell
def _(A, d, m, np, pd, x_bvp):
    y_bvp = np.linalg.solve(A, d)

    df_bvp_result = pd.DataFrame({
        "k": range(1, m + 1),
        "x_k": x_bvp,
        "y_k": y_bvp,
        "Округление до 0.001": np.round(y_bvp, 3)
    })

    df_bvp_result
    return (y_bvp,)


@app.cell
def _(plt, x_bvp, y_bvp):
    plt.figure(figsize=(8, 5))

    plt.plot(x_bvp, y_bvp, marker="o", label="Метод конечных разностей")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Решение краевой задачи")
    plt.grid(True)
    plt.legend()

    plt.show()
    return


@app.cell(hide_code=True)
def _(mo, x_values, y_bvp, y_euler, y_modified, y_rk4):
    mo.md(f"""
    ## Итоговый ответ

    ### Задача Коши

    Для уравнения

    \\[
    y'=-x+y, \\quad y(1.1)=0
    \\]

    на отрезке

    \\[
    x\\in[1.1;1.6], \\quad h=0.1
    \\]

    получены значения:

    | x | Эйлер | Модифицированный Эйлер | Рунге-Кутта 4 |
    |---|---:|---:|---:|
    | {x_values[0]:.1f} | {y_euler[0]:.3f} | {y_modified[0]:.3f} | {y_rk4[0]:.3f} |
    | {x_values[1]:.1f} | {y_euler[1]:.3f} | {y_modified[1]:.3f} | {y_rk4[1]:.3f} |
    | {x_values[2]:.1f} | {y_euler[2]:.3f} | {y_modified[2]:.3f} | {y_rk4[2]:.3f} |
    | {x_values[3]:.1f} | {y_euler[3]:.3f} | {y_modified[3]:.3f} | {y_rk4[3]:.3f} |
    | {x_values[4]:.1f} | {y_euler[4]:.3f} | {y_modified[4]:.3f} | {y_rk4[4]:.3f} |
    | {x_values[5]:.1f} | {y_euler[5]:.3f} | {y_modified[5]:.3f} | {y_rk4[5]:.3f} |

    Наиболее точным из трёх методов является метод Рунге-Кутта 4-го порядка.

    ### Краевая задача

    Для уравнения

    \\[
    y''-xy'-0.6y=4
    \\]

    при условиях

    \\[
    y'(2)=1, \\quad y'(2.3)=3
    \\]

    получено решение:

    \\[
    y(2.0)\\approx {y_bvp[0]:.3f}
    \\]

    \\[
    y(2.1)\\approx {y_bvp[1]:.3f}
    \\]

    \\[
    y(2.2)\\approx {y_bvp[2]:.3f}
    \\]

    \\[
    y(2.3)\\approx {y_bvp[3]:.3f}
    \\]
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
