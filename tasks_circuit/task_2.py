import numpy as np

def calculate_resistances_and_currents(U, R1, R2, R3, R4, R5, R6, R7, R8, R9, R10):
    R67 = R7 * R6 / (R7 + R6)

    R34 = R3 * R4 / (R3 + R4)

    R89 = R8 + R9

    R8910 = R89 * R10 / (R89 + R10)

    R345 = R34 + R5

    R3_7 = R67 * R345 / (R67 + R345)

    R2_7 = R2 + R3_7

    R2_10 = R8910 * R2_7 / (R8910 + R2_7)

    R_ekv = R1 + R2_10

    I = U / R_ekv

    I2 = I * R8910 / (R8910 + R2_7)

    I1 = I - I2

    I4 = I1 * R10 / (R10 + R89)

    I3 = I2 - I4

    I5 = I2 * R345 / (R345 + R67)

    I6 = I2 - I5
    
    I7 = I6 * R4 / (R4 + R3)

    I8 = I6 - I7
    
    I9 = I5 * R6 / (R6 + R7)

    I10 = I5 - I9

    return I, I1, I2, I3, I4, I5, I6, I7, I8, I9, I10


def main():
    R1 = float(input("Введите значение R1 (в Омах): "))
    R2 = float(input("Введите значение R2 (в Омах): "))
    R3 = float(input("Введите значение R3 (в Омах): "))
    R4 = float(input("Введите значение R4 (в Омах): "))
    R5 = float(input("Введите значение R5 (в Омах): "))
    R6 = float(input("Введите значение R6 (в Омах): "))
    R7 = float(input("Введите значение R7 (в Омах): "))
    R8 = float(input("Введите значение R8 (в Омах): "))
    R9 = float(input("Введите значение R9 (в Омах): "))
    R10 = float(input("Введите значение R10 (в Омах): "))
    U = float(input("Введите значение U (в Вольтах): "))

    I, I1, I2, I3, I4, I5, I6, I7, I8, I9, I10 = calculate_resistances_and_currents(U, R1, R2, R3, R4, R5, R6, R7, R8, R9, R10)

    print(f"Ток через источник напряжения (I): {I:.3f} А (R1)")
    print(f"Ток I1 = {I1:.3f} А (ток через R8910)")
    print(f"Ток I2 = {I2:.3f} А (ток через R2_7)")
    print(f"Ток I3 = {I3:.3f} А (ток через R10)")
    print(f"Ток I4 = {I4:.3f} А (ток через R89)")
    print(f"Ток I5 = {I5:.3f} А (ток через R67)")
    print(f"Ток I6 = {I6:.3f} А (ток через R345)")
    print(f"Ток I7 = {I7:.3f} А (ток через R3)")
    print(f"Ток I8 = {I8:.3f} А (ток через R4)")
    print(f"Ток I9 = {I9:.3f} А (ток через R7)")
    print(f"Ток I10 = {I10:.3f} А (ток через R6)")


if __name__ == "__main__":
    main()
