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
    # Лабораторная работа №4
    ## Вариант 23

    Решить систему нелинейных уравнений итерационными методами
    с точностью $\varepsilon = 0.001$:

    \[
    \begin{cases}
    \sin(x+1)+y=1.2,\\
    2x-\cos y=2.
    \end{cases}
    \]

    Методы решения:

    1. Метод простых итераций
    2. Метод Зейделя
    3. Метод Ньютона
    4. Дополнительно: метод наискорейшего спуска
    """)
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd

    return np, pd


@app.cell
def _(np):
    eps = 0.001
    max_iter = 1000

    x0 = 1.0
    y0 = 0.0

    def F(x, y):
        """
        Левая часть системы в виде F(x, y) = 0.
        """
        f1 = np.sin(x + 1) + y - 1.2
        f2 = 2 * x - np.cos(y) - 2
        return np.array([f1, f2], dtype=float)

    F(x0, y0)
    return F, eps, max_iter, x0, y0


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Преобразуем систему к виду, удобному для итерационных методов:

    \[
    2x-\cos y=2
    \]

    Отсюда:

    \[
    x=\frac{2+\cos y}{2}
    \]

    Из первого уравнения:

    \[
    \sin(x+1)+y=1.2
    \]

    Отсюда:

    \[
    y=1.2-\sin(x+1)
    \]

    Поэтому итерационные формулы:

    \[
    x_{k+1}=\frac{2+\cos y_k}{2}
    \]

    \[
    y_{k+1}=1.2-\sin(x_k+1)
    \]
    """)
    return


@app.cell
def _(F, np, pd):
    def simple_iteration(x_start, y_start, eps=0.001, max_iter=1000):
        x = x_start
        y = y_start

        rows = []

        for k in range(1, max_iter + 1):
            x_new = (2 + np.cos(y)) / 2
            y_new = 1.2 - np.sin(x + 1)

            error = max(abs(x_new - x), abs(y_new - y))
            f1, f2 = F(x_new, y_new)

            rows.append({
                "k": k,
                "x": x_new,
                "y": y_new,
                "Погрешность": error,
                "F1": f1,
                "F2": f2
            })

            if error < eps:
                return x_new, y_new, pd.DataFrame(rows)

            x = x_new
            y = y_new

        raise ValueError("Метод простых итераций не сошелся")

    return (simple_iteration,)


@app.cell
def _(eps, max_iter, simple_iteration, x0, y0):
    x_iter, y_iter, df_iter = simple_iteration(x0, y0, eps, max_iter)

    df_iter
    return df_iter, x_iter, y_iter


@app.cell
def _(F, np, pd):
    def seidel_method(x_start, y_start, eps=0.001, max_iter=1000):
        x = x_start
        y = y_start

        rows = []

        for k in range(1, max_iter + 1):
            x_old = x
            y_old = y

            x = (2 + np.cos(y)) / 2
            y = 1.2 - np.sin(x + 1)

            error = max(abs(x - x_old), abs(y - y_old))
            f1, f2 = F(x, y)

            rows.append({
                "k": k,
                "x": x,
                "y": y,
                "Погрешность": error,
                "F1": f1,
                "F2": f2
            })

            if error < eps:
                return x, y, pd.DataFrame(rows)

        raise ValueError("Метод Зейделя не сошелся")

    return (seidel_method,)


@app.cell
def _(eps, max_iter, seidel_method, x0, y0):
    x_seidel, y_seidel, df_seidel = seidel_method(x0, y0, eps, max_iter)

    df_seidel
    return df_seidel, x_seidel, y_seidel


@app.cell
def _(F, np, pd):
    def newton_method(x_start, y_start, eps=0.001, max_iter=1000):
        x = x_start
        y = y_start

        rows = []

        for k in range(1, max_iter + 1):
            f = F(x, y)

            J = np.array([
                [np.cos(x + 1), 1],
                [2, np.sin(y)]
            ], dtype=float)

            delta = np.linalg.solve(J, -f)

            x_new = x + delta[0]
            y_new = y + delta[1]

            error = max(abs(x_new - x), abs(y_new - y))
            f1, f2 = F(x_new, y_new)

            rows.append({
                "k": k,
                "x": x_new,
                "y": y_new,
                "Погрешность": error,
                "F1": f1,
                "F2": f2
            })

            if error < eps:
                return x_new, y_new, pd.DataFrame(rows)

            x = x_new
            y = y_new

        raise ValueError("Метод Ньютона не сошелся")

    return (newton_method,)


@app.cell
def _(eps, max_iter, newton_method, x0, y0):
    x_newton, y_newton, df_newton = newton_method(x0, y0, eps, max_iter)

    df_newton
    return df_newton, x_newton, y_newton


@app.cell
def _(
    df_iter,
    df_newton,
    df_seidel,
    pd,
    x_iter,
    x_newton,
    x_seidel,
    y_iter,
    y_newton,
    y_seidel,
):
    df_compare = pd.DataFrame({
        "Метод": [
            "Простые итерации",
            "Зейдель",
            "Ньютон"
        ],
        "x": [
            x_iter,
            x_seidel,
            x_newton
        ],
        "y": [
            y_iter,
            y_seidel,
            y_newton
        ],
        "x, округление до 0.001": [
            round(x_iter, 3),
            round(x_seidel, 3),
            round(x_newton, 3)
        ],
        "y, округление до 0.001": [
            round(y_iter, 3),
            round(y_seidel, 3),
            round(y_newton, 3)
        ],
        "Количество итераций": [
            len(df_iter),
            len(df_seidel),
            len(df_newton)
        ]
    })

    df_compare
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Дополнительное задание

    Метод наискорейшего спуска применяется к функции:

    \[
    \Phi(x,y)=F_1^2(x,y)+F_2^2(x,y)
    \]

    где

    \[
    F_1(x,y)=\sin(x+1)+y-1.2
    \]

    \[
    F_2(x,y)=2x-\cos y-2
    \]

    Минимум функции \(\Phi(x,y)\) соответствует решению системы,
    потому что в точке решения:

    \[
    F_1(x,y)=0,\quad F_2(x,y)=0
    \]

    Следовательно:

    \[
    \Phi(x,y)=0
    \]
    """)
    return


@app.cell
def _(F, np):
    def phi(x, y):
        f = F(x, y)
        return f[0] ** 2 + f[1] ** 2


    def grad_phi(x, y):
        f1, f2 = F(x, y)

        # Производные:
        # F1 = sin(x + 1) + y - 1.2
        # F2 = 2x - cos(y) - 2

        f1_x = np.cos(x + 1)
        f1_y = 1

        f2_x = 2
        f2_y = np.sin(y)

        grad_x = 2 * f1 * f1_x + 2 * f2 * f2_x
        grad_y = 2 * f1 * f1_y + 2 * f2 * f2_y

        return np.array([grad_x, grad_y], dtype=float)

    return grad_phi, phi


@app.cell
def _(np):
    def golden_section_search(func, a=0.0, b=1.0, eps=1e-6):
        """
        Метод золотого сечения для поиска минимума функции одной переменной.
        """
        gr = (np.sqrt(5) - 1) / 2

        c = b - gr * (b - a)
        d = a + gr * (b - a)

        while abs(b - a) > eps:
            if func(c) < func(d):
                b = d
            else:
                a = c

            c = b - gr * (b - a)
            d = a + gr * (b - a)

        return (a + b) / 2

    return (golden_section_search,)


@app.cell
def _(F, golden_section_search, grad_phi, np, pd, phi):
    def steepest_descent(x_start, y_start, eps=0.001, max_iter=1000):
        x = x_start
        y = y_start

        rows = []

        for k in range(1, max_iter + 1):
            grad = grad_phi(x, y)

            grad_norm = np.linalg.norm(grad)

            if grad_norm < eps:
                return x, y, pd.DataFrame(rows)

            direction = -grad

            def one_dimensional_phi(alpha):
                point = np.array([x, y]) + alpha * direction
                return phi(point[0], point[1])

            alpha = golden_section_search(one_dimensional_phi, 0.0, 1.0)

            x_new = x + alpha * direction[0]
            y_new = y + alpha * direction[1]

            error = max(abs(x_new - x), abs(y_new - y))
            f1, f2 = F(x_new, y_new)

            rows.append({
                "k": k,
                "alpha": alpha,
                "x": x_new,
                "y": y_new,
                "Phi": phi(x_new, y_new),
                "Погрешность": error,
                "F1": f1,
                "F2": f2
            })

            if error < eps and phi(x_new, y_new) < eps:
                return x_new, y_new, pd.DataFrame(rows)

            x = x_new
            y = y_new

        raise ValueError("Метод наискорейшего спуска не сошелся")

    return (steepest_descent,)


@app.cell
def _(eps, max_iter, steepest_descent, x0, y0):
    x_sd, y_sd, df_sd = steepest_descent(x0, y0, eps, max_iter)

    df_sd
    return df_sd, x_sd, y_sd


@app.cell
def _(
    df_iter,
    df_newton,
    df_sd,
    df_seidel,
    pd,
    x_iter,
    x_newton,
    x_sd,
    x_seidel,
    y_iter,
    y_newton,
    y_sd,
    y_seidel,
):
    df_all = pd.DataFrame({
        "Метод": [
            "Простые итерации",
            "Зейдель",
            "Ньютон",
            "Наискорейший спуск"
        ],
        "x": [
            x_iter,
            x_seidel,
            x_newton,
            x_sd
        ],
        "y": [
            y_iter,
            y_seidel,
            y_newton,
            y_sd
        ],
        "x, округление до 0.001": [
            round(x_iter, 3),
            round(x_seidel, 3),
            round(x_newton, 3),
            round(x_sd, 3)
        ],
        "y, округление до 0.001": [
            round(y_iter, 3),
            round(y_seidel, 3),
            round(y_newton, 3),
            round(y_sd, 3)
        ],
        "Количество итераций": [
            len(df_iter),
            len(df_seidel),
            len(df_newton),
            len(df_sd)
        ]
    })

    df_all
    return


@app.cell(hide_code=True)
def _(mo, x_iter, x_newton, x_sd, x_seidel, y_iter, y_newton, y_sd, y_seidel):
    mo.md(f"""
    ## Итоговый ответ

    Система:

    \\[
    \\begin{{cases}}
    \\sin(x+1)+y=1.2,\\\\
    2x-\\cos y=2.
    \\end{{cases}}
    \\]

    Решение, найденное методом простых итераций:

    \\[
    x \\approx {x_iter:.3f}, \\quad y \\approx {y_iter:.3f}
    \\]

    Решение, найденное методом Зейделя:

    \\[
    x \\approx {x_seidel:.3f}, \\quad y \\approx {y_seidel:.3f}
    \\]

    Решение, найденное методом Ньютона:

    \\[
    x \\approx {x_newton:.3f}, \\quad y \\approx {y_newton:.3f}
    \\]

    Решение, найденное методом наискорейшего спуска:

    \\[
    x \\approx {x_sd:.3f}, \\quad y \\approx {y_sd:.3f}
    \\]

    Окончательное решение с точностью 0.001:

    \\[
    \\boxed{{x \\approx {x_newton:.3f}, \\quad y \\approx {y_newton:.3f}}}
    \\]
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
