import numpy as np
import matplotlib.pyplot as plt

# Параметры антенны
d = 0.5  # расстояние между элементами в длинах волн
theta = np.linspace(0, 2 * np.pi, 360)  # углы от 0 до 2pi

# Комплексные амплитуды для каждого набора
amplitudes_a = np.array([1 + 0j, -1 + 0j, 1 + 0j, -1 + 0j, 1 + 0j, -1 + 0j])
amplitudes_b = np.array([1 + 0j, 0 + 1j, -1 + 0j, 0 + -1j, 1 + 0j, 0 + 1j])
amplitudes_c = np.array([1 + 0j, -0.61 + 0.8j, -0.27 - 0.96j, 0.93 + 0.37j, -0.86 + 0.51j, 0.11 - 0.99j])


def array_factor(amplitudes, theta, d):
    n = len(amplitudes)
    af = np.zeros_like(theta, dtype=np.complex_)

    for i in range(n):
        af += amplitudes[i] * np.exp(1j * 2 * np.pi * d * i * np.cos(theta))

    return af


def plot_pattern(amplitudes, d):
    af = array_factor(amplitudes, theta, d)
    af_normalized = np.abs(af) / np.max(np.abs(af))  # нормализуем амплитуду
    plt.polar(theta, af_normalized)
    plt.title('Диаграмма направленности')
    plt.show()

    # Направление основного луча
    max_index = np.argmax(af_normalized)
    main_beam_direction = np.degrees(theta[max_index])

    return main_beam_direction


# Диаграммы направленности и нахождение основного направления луча для каждого набора
main_beam_a = plot_pattern(amplitudes_a, d)
main_beam_b = plot_pattern(amplitudes_b, d)
main_beam_c = plot_pattern(amplitudes_c, d)

print(f'Основное направление луча для набора a: {main_beam_a} градусов')
print(f'Основное направление луча для набора b: {main_beam_b} градусов')
print(f'Основное направление луча для набора c: {main_beam_c} градусов')
