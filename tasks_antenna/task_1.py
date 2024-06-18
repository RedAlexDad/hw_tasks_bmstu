import numpy as np

# Параметры задачи
c = 3e8             # скорость света, м/с
f = 6e9             # частота, Гц
lambda_ = c / f     # длина волны, м
d = lambda_ / 2     # расстояние между элементами, м
theta = 45          # угол основного направления луча, градусы

# Расчет фазовый сдвиг
def calculate_phase_shift(n, d, lambda_, theta):
    delta_phi = -2 * np.pi * d / lambda_ * np.sin(np.radians(theta))
    phi_n = (n - 1) * delta_phi
    return phi_n

# Рассчитываем фазовый сдвиг для 4-го элемента
n = 4
phase_shift = calculate_phase_shift(n, d, lambda_, theta)

print(f"Фазовый сдвиг для {n}-го элемента: {phase_shift} радиан")
