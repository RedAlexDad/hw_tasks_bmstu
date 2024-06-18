import numpy as np


def solve_circuit(R1, R2, R3, R4, Vbias):
    # Составляем матрицу коэффициентов и вектор правых частей для системы уравнений
    A = np.array(
        [
            [1 / R1 + 1 / R2, -1 / R2],
            [-1 / R2, 1 / R2 + 1 / (R3 + R4)]
        ]
    )

    B = np.array([Vbias / R1, 0])

    # Решаем систему уравнений
    V = np.linalg.solve(A, B)

    V1 = V[0]
    V2 = V[1]
    # Так как узел 3 заземлен
    V3 = 0

    # Находим ток через источник напряжения
    I_R1 = (V1 - Vbias) / R1
    I_R2 = (V1 - V2) / R2

    I_source = I_R1 + I_R2

    return V1, V2, V3, I_source


def main():
    R1 = float(input("Введите значение R1 (в Омах): "))
    R2 = float(input("Введите значение R2 (в Омах): "))
    R3 = float(input("Введите значение R3 (в Омах): "))
    R4 = float(input("Введите значение R4 (в Омах): "))
    Vbias = float(input("Введите значение Vbias (в Вольтах): "))

    V1, V2, V3, I_source = solve_circuit(R1, R2, R3, R4, Vbias)

    print(f"Напряжение в узле 1 (V1): {V1:.2f} В")
    print(f"Напряжение в узле 2 (V2): {V2:.2f} В")
    print(f"Напряжение в узле 3 (V3): {V3:.2f} В")
    print(f"Ток через источник напряжения (I): {I_source:.2f} А")


if __name__ == "__main__":
    main()
