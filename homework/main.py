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
    # Семестровое задание по дисциплине «Численные методы»
    ## Вариант 23

    В работе выполняются три задания:

    1. Вычисление тройного интеграла методом Монте-Карло.
    2. Решение системы линейных уравнений методом Монте-Карло.
    3. Определение площади фигуры методом Монте-Карло.

    Во всех задачах используются случайные точки и статистическое усреднение.
    """)
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    return np, pd, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Задание 1. Интеграл методом Монте-Карло

    \[
    I=\int_0^1\int_1^2\int_0^3
    \left(x+\frac{y}{5}+z\right)\,dx\,dy\,dz
    \]

    Область интегрирования:

    \[
    0\le x\le 1,\quad 1\le y\le 2,\quad 0\le z\le 3
    \]

    Объём области:

    \[
    V=3
    \]
    """)
    return


@app.cell
def _():
    def f_integral(x, y, z):
        return x + y / 5 + z


    x_min, x_max = 0, 1
    y_min, y_max = 1, 2
    z_min, z_max = 0, 3

    volume = (x_max - x_min) * (y_max - y_min) * (z_max - z_min)

    volume
    return f_integral, volume, x_max, x_min, y_max, y_min, z_max, z_min


@app.cell
def _(f_integral, np, volume, x_max, x_min, y_max, y_min, z_max, z_min):
    def monte_carlo_integral(N=100_000, seed=23):
        rng = np.random.default_rng(seed)

        x_rand = rng.uniform(x_min, x_max, N)
        y_rand = rng.uniform(y_min, y_max, N)
        z_rand = rng.uniform(z_min, z_max, N)

        values = f_integral(x_rand, y_rand, z_rand)

        integral_estimate = volume * np.mean(values)
        standard_error = volume * np.std(values, ddof=1) / np.sqrt(N)

        return integral_estimate, standard_error


    I_mc, I_error = monte_carlo_integral(N=100_000, seed=23)

    I_mc, I_error
    return (I_mc,)


@app.cell
def _(I_mc, pd):
    I_exact = 6.9

    df_integral = pd.DataFrame({
        "Метод": ["Монте-Карло", "Точное значение"],
        "Значение интеграла": [I_mc, I_exact],
        "Округление до 0.001": [round(I_mc, 3), round(I_exact, 3)],
        "Абсолютная ошибка": [abs(I_mc - I_exact), 0],
    })

    df_integral
    return (I_exact,)


@app.cell(hide_code=True)
def _(I_exact, I_mc, mo):
    mo.md(f"""
    ### Вывод по заданию 1

    При использовании метода Монте-Карло получено:

    \\[
    I_{{MC}} \\approx {I_mc:.3f}
    \\]

    Точное значение интеграла:

    \\[
    I_{{точн}} = {I_exact:.3f}
    \\]

    Абсолютная ошибка:

    \\[
    |I_{{MC}}-I_{{точн}}| \\approx {abs(I_mc - I_exact):.5f}
    \\]

    Метод Монте-Карло дал результат, близкий к точному значению.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Задание 2. Решение СЛАУ методом Монте-Карло

    Система варианта 23:

    \[
    \begin{cases}
    x_1=0.4x_1+0.2x_2+0.7,\\
    x_2=0.2x_1+0.1x_2+0.6.
    \end{cases}
    \]

    В матричном виде:

    \[
    X=AX+\beta
    \]

    где

    \[
    A=
    \begin{pmatrix}
    0.4 & 0.2\\
    0.2 & 0.1
    \end{pmatrix},
    \quad
    \beta=
    \begin{pmatrix}
    0.7\\
    0.6
    \end{pmatrix}
    \]
    """)
    return


@app.cell
def _(np, pd):
    A = np.array([
        [0.4, 0.2],
        [0.2, 0.1]
    ], dtype=float)

    beta = np.array([0.7, 0.6], dtype=float)

    df_slae_data = pd.DataFrame(
        np.column_stack([A, beta]),
        columns=["alpha_1", "alpha_2", "beta"],
        index=["x1", "x2"]
    )

    df_slae_data
    return A, beta


@app.cell
def _(A, beta, np, pd):
    X_exact = np.linalg.solve(np.eye(2) - A, beta)

    df_exact_slae = pd.DataFrame({
        "Переменная": ["x1", "x2"],
        "Точное решение": X_exact,
        "Округление до 0.001": np.round(X_exact, 3)
    })

    df_exact_slae
    return (X_exact,)


@app.cell
def _(A, X_exact, beta, np, pd):
    def monte_carlo_linear_system(A, beta, N=100_000, max_steps=80, seed=23):
        """
        Решение системы X = A X + beta методом Монте-Карло.

        Для каждой компоненты x_i запускаются случайные траектории,
        начинающиеся из состояния i.

        Оценка строится по сумме:
        beta_i + A beta_i + A^2 beta_i + ...
        """
        rng = np.random.default_rng(seed)

        n = len(beta)

        # Матрица вероятностей переходов.
        # Берём равновероятный переход в одно из двух состояний.
        P = np.full((n, n), 1 / n)

        estimates = np.zeros((n, N))

        for start_state in range(n):
            states = np.full(N, start_state, dtype=int)
            weights = np.ones(N)
            sums = np.zeros(N)

            for _ in range(max_steps):
                sums += weights * beta[states]

                random_values = rng.random(N)
                next_states = (random_values >= 0.5).astype(int)

                transition_weights = A[states, next_states] / P[states, next_states]
                weights *= transition_weights
                states = next_states

            estimates[start_state] = sums

        X_mc = estimates.mean(axis=1)
        X_std_error = estimates.std(axis=1, ddof=1) / np.sqrt(N)

        return X_mc, X_std_error


    X_mc, X_mc_error = monte_carlo_linear_system(A, beta, N=100_000, seed=23)

    df_mc_slae = pd.DataFrame({
        "Переменная": ["x1", "x2"],
        "Монте-Карло": X_mc,
        "Стандартная ошибка": X_mc_error,
        "Точное решение": X_exact,
        "Абсолютная ошибка": np.abs(X_mc - X_exact),
        "Округление Монте-Карло": np.round(X_mc, 3)
    })

    df_mc_slae
    return (X_mc,)


@app.cell
def _(A, X_exact, X_mc, beta, pd):
    residual_mc = X_mc - (A @ X_mc + beta)
    residual_exact = X_exact - (A @ X_exact + beta)

    df_residuals = pd.DataFrame({
        "Переменная": ["x1", "x2"],
        "Невязка Монте-Карло": residual_mc,
        "Невязка точного решения": residual_exact
    })

    df_residuals
    return


@app.cell(hide_code=True)
def _(X_exact, X_mc, mo):
    mo.md(f"""
    ### Вывод по заданию 2

    Методом Монте-Карло получено приближённое решение:

    \\[
    x_1 \\approx {X_mc[0]:.3f}, \\quad x_2 \\approx {X_mc[1]:.3f}
    \\]

    Точное решение системы:

    \\[
    x_1 = {X_exact[0]:.3f}, \\quad x_2 = {X_exact[1]:.3f}
    \\]

    Абсолютные ошибки:

    \\[
    |\\Delta x_1| \\approx {abs(X_mc[0] - X_exact[0]):.5f}
    \\]

    \\[
    |\\Delta x_2| \\approx {abs(X_mc[1] - X_exact[1]):.5f}
    \\]

    Следовательно, метод Монте-Карло даёт результат, близкий к точному решению,
    но из-за случайного характера метода небольшая погрешность сохраняется.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Задание 3. Площадь фигуры методом Монте-Карло

    Вариант 11

    Ограничения варианта 11:

    \[
    \begin{cases}
    -x^3-y^4<2,\\
    3x+y^2<2,\\
    -2<x<2,\\
    -2<y<2.
    \end{cases}
    \]

    Ограничивающий прямоугольник:

    \[
    -2<x<2,\quad -2<y<2
    \]

    \[
    S_{\text{прям}}=16
    \]
    """)
    return


@app.cell
def _():
    x_area_min, x_area_max = -2, 2
    y_area_min, y_area_max = -2, 2

    rectangle_area = (x_area_max - x_area_min) * (y_area_max - y_area_min)


    def inside_area(x, y):
        """
        Проверяет, принадлежит ли точка (x, y) заданной фигуре.

        Ограничения:
        -x^3 - y^4 < 2
        3x + y^2 < 2
        -2 < x < 2
        -2 < y < 2
        """
        return (
            (-x**3 - y**4 < 2)
            & (3 * x + y**2 < 2)
            & (x_area_min < x)
            & (x < x_area_max)
            & (y_area_min < y)
            & (y < y_area_max)
        )


    rectangle_area
    return (
        inside_area,
        rectangle_area,
        x_area_max,
        x_area_min,
        y_area_max,
        y_area_min,
    )


@app.cell
def _(
    inside_area,
    np,
    rectangle_area,
    x_area_max,
    x_area_min,
    y_area_max,
    y_area_min,
):
    def monte_carlo_area(N=100_000, seed=23):
        rng = np.random.default_rng(seed)

        x_rand = rng.uniform(x_area_min, x_area_max, N)
        y_rand = rng.uniform(y_area_min, y_area_max, N)

        mask = inside_area(x_rand, y_rand)

        K = np.sum(mask)
        area_estimate = rectangle_area * K / N

        p = K / N
        standard_error = rectangle_area * np.sqrt(p * (1 - p) / N)

        return area_estimate, standard_error, K, x_rand, y_rand, mask


    S_mc, S_error, K, x_random_area, y_random_area, mask_area = monte_carlo_area(
        N=100_000,
        seed=23
    )

    S_mc, S_error, K
    return K, S_mc, mask_area, x_random_area, y_random_area


@app.cell
def _(np, x_area_min):
    def y_length_for_area(x):
        """
        Длина вертикального сечения фигуры при фиксированном x.
        Используется для контрольного численного интегрирования.
        """
        if x <= x_area_min or x >= 2 / 3:
            return 0.0

        upper = min(2.0, np.sqrt(max(0.0, 2 - 3 * x)))

        lower_condition = -2 - x**3

        if lower_condition > 0:
            lower = lower_condition ** 0.25
        else:
            lower = 0.0

        length = 2 * max(0.0, upper - lower)

        return length


    x_dense = np.linspace(x_area_min, 2 / 3, 200_000)
    y_lengths = np.array([y_length_for_area(value) for value in x_dense])

    S_reference = np.trapezoid(y_lengths, x_dense)

    S_reference
    return (S_reference,)


@app.cell
def _(K, S_mc, S_reference, pd, rectangle_area):
    df_area = pd.DataFrame({
        "Величина": [
            "Количество случайных точек N",
            "Количество точек внутри K",
            "Площадь прямоугольника",
            "Площадь методом Монте-Карло",
            "Контрольное численное значение",
            "Абсолютная ошибка",
            "Относительная ошибка, %"
        ],
        "Значение": [
            100_000,
            K,
            rectangle_area,
            S_mc,
            S_reference,
            abs(S_mc - S_reference),
            abs(S_mc - S_reference) / S_reference * 100
        ]
    })

    df_area
    return


@app.cell
def _(
    mask_area,
    plt,
    x_area_max,
    x_area_min,
    x_random_area,
    y_area_max,
    y_area_min,
    y_random_area,
):
    plot_N = 20_000

    x_plot_points = x_random_area[:plot_N]
    y_plot_points = y_random_area[:plot_N]
    mask_plot = mask_area[:plot_N]

    plt.figure(figsize=(7, 7))

    plt.scatter(
        x_plot_points[~mask_plot],
        y_plot_points[~mask_plot],
        s=3,
        alpha=0.25,
        label="Вне фигуры"
    )

    plt.scatter(
        x_plot_points[mask_plot],
        y_plot_points[mask_plot],
        s=3,
        alpha=0.25,
        label="Внутри фигуры"
    )

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Фигура и случайные точки метода Монте-Карло")
    plt.xlim(x_area_min, x_area_max)
    plt.ylim(y_area_min, y_area_max)
    plt.grid(True)
    plt.legend()
    plt.show()
    return


@app.cell
def _(inside_area, np, plt, x_area_max, x_area_min, y_area_max, y_area_min):
    grid_size = 500

    x_grid = np.linspace(x_area_min, x_area_max, grid_size)
    y_grid = np.linspace(y_area_min, y_area_max, grid_size)

    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)

    Z_inside = inside_area(X_grid, Y_grid)

    plt.figure(figsize=(7, 7))

    plt.contourf(X_grid, Y_grid, Z_inside, levels=[0.5, 1], alpha=0.6)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Приближённый вид заданной фигуры")
    plt.xlim(x_area_min, x_area_max)
    plt.ylim(y_area_min, y_area_max)
    plt.grid(True)

    plt.show()
    return


@app.cell(hide_code=True)
def _(K, S_mc, S_reference, mo, rectangle_area):
    mo.md(f"""
    ### Вывод по заданию 3

    Ограничивающий прямоугольник:

    \\[
    -2<x<2, \\quad -2<y<2
    \\]

    Его площадь:

    \\[
    S_{{прям}} = {rectangle_area:.3f}
    \\]

    Было сгенерировано:

    \\[
    N=100000
    \\]

    Количество точек внутри фигуры:

    \\[
    K={K}
    \\]

    Площадь фигуры методом Монте-Карло:

    \\[
    S_{{MC}} \\approx {S_mc:.3f}
    \\]

    Контрольное численное значение площади:

    \\[
    S_{{контр}} \\approx {S_reference:.3f}
    \\]

    Абсолютная ошибка:

    \\[
    |S_{{MC}}-S_{{контр}}| \\approx {abs(S_mc - S_reference):.3f}
    \\]

    Относительная ошибка:

    \\[
    \\delta \\approx {abs(S_mc - S_reference) / S_reference * 100:.2f}\\%
    \\]
    """)
    return


@app.cell
def _(I_exact, I_mc, S_mc, S_reference, X_exact, X_mc, pd):
    df_final = pd.DataFrame({
        "Задание": [
            "1. Интеграл методом Монте-Карло",
            "2. СЛАУ методом Монте-Карло: x1",
            "2. СЛАУ методом Монте-Карло: x2",
            "3. Площадь фигуры методом Монте-Карло"
        ],
        "Результат Монте-Карло": [
            I_mc,
            X_mc[0],
            X_mc[1],
            S_mc
        ],
        "Контрольное / точное значение": [
            I_exact,
            X_exact[0],
            X_exact[1],
            S_reference
        ],
        "Абсолютная ошибка": [
            abs(I_mc - I_exact),
            abs(X_mc[0] - X_exact[0]),
            abs(X_mc[1] - X_exact[1]),
            abs(S_mc - S_reference)
        ]
    })

    df_final
    return


@app.cell(hide_code=True)
def _(I_exact, I_mc, S_mc, S_reference, X_exact, X_mc, mo):
    mo.md(f"""
    # Финальный вывод

    В семестровом задании для варианта 23 были выполнены три задачи.

    ## 1. Интеграл

    Методом Монте-Карло вычислен интеграл:

    \\[
    I=\\int_0^1\\int_1^2\\int_0^3
    \\left(x+\\frac{{y}}{{5}}+z\\right)\\,dx\\,dy\\,dz
    \\]

    Получено:

    \\[
    I_{{MC}}\\approx {I_mc:.3f}
    \\]

    Точное значение:

    \\[
    I_{{точн}}={I_exact:.3f}
    \\]

    ## 2. Система линейных уравнений

    Решалась система:

    \\[
    \\begin{{cases}}
    x_1=0.4x_1+0.2x_2+0.7,\\\\
    x_2=0.2x_1+0.1x_2+0.6.
    \\end{{cases}}
    \\]

    Методом Монте-Карло получено:

    \\[
    x_1\\approx {X_mc[0]:.3f}, \\quad x_2\\approx {X_mc[1]:.3f}
    \\]

    Точное решение:

    \\[
    x_1={X_exact[0]:.3f}, \\quad x_2={X_exact[1]:.3f}
    \\]

    ## 3. Площадь фигуры

    Для фигуры с ограничениями:

    \\[
    \\begin{{cases}}
    -x^3-y^4<2,\\\\
    3x+y^2<2,\\\\
    -2<x<2,\\\\
    -2<y<2
    \\end{{cases}}
    \\]

    получена площадь:

    \\[
    S_{{MC}}\\approx {S_mc:.3f}
    \\]

    Контрольное значение:

    \\[
    S_{{контр}}\\approx {S_reference:.3f}
    \\]

    Все три задачи решены методом Монте-Карло. Результаты близки к контрольным значениям,
    а небольшие отличия объясняются случайным характером метода.
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
