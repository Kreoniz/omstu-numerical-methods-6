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
    # Лабораторная работа №3
    ## Вариант 23

    Решить систему линейных алгебраических уравнений методами:

    1. Метод Гаусса
    2. Метод обратной матрицы

    Точность: $\varepsilon = 0.001$
    """)
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd

    return np, pd


@app.cell
def _(np, pd):
    eps = 0.001

    A = np.array([
        [ 2.38,  1.54, -4.27,  0.96],
        [ 1.72, -2.89,  1.35, -3.42],
        [ 3.61,  0.78, -2.53,  1.84],
        [-0.85,  3.26,  1.97, -0.69],
    ], dtype=float)

    b = np.array([3.85, -5.16, 4.29, 2.71], dtype=float)

    df_system = pd.DataFrame(
        np.column_stack([A, b]),
        columns=["x1", "x2", "x3", "x4", "b"]
    )

    df_system
    return A, b


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Система уравнений варианта 23:

    \[
    \begin{cases}
    2.38x_1 + 1.54x_2 - 4.27x_3 + 0.96x_4 = 3.85,\\
    1.72x_1 - 2.89x_2 + 1.35x_3 - 3.42x_4 = -5.16,\\
    3.61x_1 + 0.78x_2 - 2.53x_3 + 1.84x_4 = 4.29,\\
    -0.85x_1 + 3.26x_2 + 1.97x_3 - 0.69x_4 = 2.71.
    \end{cases}
    \]
    """)
    return


@app.cell
def _(np):
    def gauss_solve(A, b):
        A = A.astype(float).copy()
        b = b.astype(float).copy()
        n = len(b)

        # Прямой ход
        for k in range(n - 1):
            # Выбор главного элемента
            max_row = k + np.argmax(np.abs(A[k:, k]))

            if abs(A[max_row, k]) < 1e-12:
                raise ValueError("Система не имеет единственного решения")

            # Перестановка строк
            if max_row != k:
                A[[k, max_row]] = A[[max_row, k]]
                b[[k, max_row]] = b[[max_row, k]]

            # Исключение неизвестных
            for i in range(k + 1, n):
                factor = A[i, k] / A[k, k]
                A[i, k:] -= factor * A[k, k:]
                b[i] -= factor * b[k]

        # Обратный ход
        x = np.zeros(n)

        for i in range(n - 1, -1, -1):
            x[i] = (b[i] - np.dot(A[i, i + 1:], x[i + 1:])) / A[i, i]

        return x

    return (gauss_solve,)


@app.cell
def _(A, b, gauss_solve, np, pd):
    x_gauss = gauss_solve(A, b)
    residual_gauss = b - A @ x_gauss

    df_gauss = pd.DataFrame({
        "Переменная": ["x1", "x2", "x3", "x4"],
        "Значение": x_gauss,
        "Округление до 0.001": np.round(x_gauss, 3),
        "Невязка": residual_gauss
    })

    df_gauss
    return (x_gauss,)


@app.cell
def _(A, b, np, pd):
    A_inv = np.linalg.inv(A)
    x_inverse = A_inv @ b
    residual_inverse = b - A @ x_inverse

    df_inverse = pd.DataFrame({
        "Переменная": ["x1", "x2", "x3", "x4"],
        "Значение": x_inverse,
        "Округление до 0.001": np.round(x_inverse, 3),
        "Невязка": residual_inverse
    })

    df_inverse
    return (x_inverse,)


@app.cell
def _(np, pd, x_gauss, x_inverse):
    difference = np.abs(x_gauss - x_inverse)

    df_compare = pd.DataFrame({
        "Переменная": ["x1", "x2", "x3", "x4"],
        "Метод Гаусса": np.round(x_gauss, 6),
        "Метод обратной матрицы": np.round(x_inverse, 6),
        "Разность": difference
    })

    df_compare
    return


@app.cell(hide_code=True)
def _(mo, x_gauss, x_inverse):
    mo.md(f"""
    ## Ответ

    Метод Гаусса:

    \\[
    x_1 = {x_gauss[0]:.3f},\\quad
    x_2 = {x_gauss[1]:.3f},\\quad
    x_3 = {x_gauss[2]:.3f},\\quad
    x_4 = {x_gauss[3]:.3f}
    \\]

    Метод обратной матрицы:

    \\[
    x_1 = {x_inverse[0]:.3f},\\quad
    x_2 = {x_inverse[1]:.3f},\\quad
    x_3 = {x_inverse[2]:.3f},\\quad
    x_4 = {x_inverse[3]:.3f}
    \\]

    Окончательное решение с точностью 0.001:

    \\[
    \\boxed{{x_1 = {x_gauss[0]:.3f},\\ x_2 = {x_gauss[1]:.3f},\\ x_3 = {x_gauss[2]:.3f},\\ x_4 = {x_gauss[3]:.3f}}}
    \\]
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Дополнительное задание к ЛР №3

    Найти собственные числа и собственные векторы матрицы варианта 23:

    \[
    A =
    \begin{pmatrix}
    2.73 & -1.15 & 2.34 & 0.28\\
    7.56 & 2.34 & 1.94 & 0.07\\
    5.68 & 0.39 & 7.49 & 2.34\\
    -1.58 & 0.37 & 0.01 & 2.97
    \end{pmatrix}
    \]
    """)
    return


@app.cell
def _(np, pd):
    A_eig = np.array([
        [ 2.73, -1.15,  2.34, 0.28],
        [ 7.56,  2.34,  1.94, 0.07],
        [ 5.68,  0.39,  7.49, 2.34],
        [-1.58,  0.37,  0.01, 2.97],
    ], dtype=float)

    df_A_eig = pd.DataFrame(
        A_eig,
        columns=["1", "2", "3", "4"],
        index=["1", "2", "3", "4"]
    )

    df_A_eig
    return (A_eig,)


@app.function
def format_complex(z, digits=6):
    z = complex(z)

    if abs(z.imag) < 1e-10:
        return f"{z.real:.{digits}f}"

    sign = "+" if z.imag >= 0 else "-"
    return f"{z.real:.{digits}f} {sign} {abs(z.imag):.{digits}f}i"


@app.cell
def _(A_eig, np, pd):
    eigenvalues, eigenvectors = np.linalg.eig(A_eig)

    df_eigenvalues = pd.DataFrame({
        "Номер": [1, 2, 3, 4],
        "Собственное число": [format_complex(value) for value in eigenvalues]
    })

    df_eigenvalues
    return eigenvalues, eigenvectors


@app.cell
def _(eigenvectors, pd):
    df_eigenvectors = pd.DataFrame(
        eigenvectors,
        columns=[f"v{i}" for i in range(1, 5)],
        index=["x1", "x2", "x3", "x4"]
    )

    df_eigenvectors_formatted = df_eigenvectors.map(format_complex)

    df_eigenvectors_formatted
    return


@app.cell
def _(A_eig, eigenvalues, eigenvectors, np, pd):
    checks = []

    for i in range(len(eigenvalues)):
        lambda_i = eigenvalues[i]
        v_i = eigenvectors[:, i]

        left = A_eig @ v_i
        right = lambda_i * v_i
        error = np.linalg.norm(left - right)

        checks.append({
            "Номер": i + 1,
            "λ": format_complex(lambda_i),
            "||Av - λv||": error
        })

    df_check = pd.DataFrame(checks)

    df_check
    return


@app.cell(hide_code=True)
def _(eigenvalues, mo):
    mo.md(f"""
    ## Ответ

    Собственные числа матрицы:

    \\[
    \\lambda_1 \\approx {format_complex(eigenvalues[0], 3)}
    \\]

    \\[
    \\lambda_2 \\approx {format_complex(eigenvalues[1], 3)}
    \\]

    \\[
    \\lambda_3 \\approx {format_complex(eigenvalues[2], 3)}
    \\]

    \\[
    \\lambda_4 \\approx {format_complex(eigenvalues[3], 3)}
    \\]

    Так как матрица несимметричная, среди собственных чисел появилась
    комплексно-сопряжённая пара.

    Проверка выполнена по формуле:

    \\[
    Av = \\lambda v
    \\]

    Значения нормы ошибки близки к нулю, следовательно, собственные числа
    и собственные векторы найдены верно.
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
