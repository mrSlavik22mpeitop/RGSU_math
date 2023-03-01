from scipy.spatial import HalfspaceIntersection, ConvexHull
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
import math



q = int(input("Количество ограничений в системе:"))
w = int(input("Количество переменных x в задаче:"))
e = q + w

c = [0]*e
b = [0] * q
A = [[0]*e for i in range(q)]

print("Введите значения коэффициентов целевой функции:")
for i in range(w):
    c[i] = float(input(f"x{i+1}="))
u = input("Какой тип решения (min или max):")
if u == 'min':
    c = list(map(lambda x:-x, c))

print("Введите ограничения:")
for j in range(q):
    for i in range(w):
        A[j][i] = float(input(f"x{i + 1}="))
    b[j] = float(input(f"b{j+1}="))
    rut = input("Введите y если ограничение <=:")
    if rut == 'y':
        A[j][w+j] = 1.0
    else:
        A[j][w+j] = -1.0

print(c)
print(b)
print(A)


def to_tableau(c, A, b):
    xb = [eq + [x] for eq, x in zip(A, b)]
    z = c + [0]
    return xb + [z]


def can_be_improved(tableau):
    z = tableau[-1]
    return any(x > 0 for x in z[:-1])




def get_pivot_position(tableau):
    z = tableau[-1]
    column = next(i for i, x in enumerate(z[:-1]) if x > 0)

    restrictions = []
    for eq in tableau[:-1]:
        el = eq[column]
        restrictions.append(math.inf if el <= 0 else eq[-1] / el)

    if (all([r == math.inf for r in restrictions])):
        raise Exception("Linear program is unbounded.")

    row = restrictions.index(min(restrictions))
    return row, column


def pivot_step(tableau, pivot_position):
    new_tableau = [[] for eq in tableau]

    i, j = pivot_position
    pivot_value = tableau[i][j]
    new_tableau[i] = np.array(tableau[i]) / pivot_value

    for eq_i, eq in enumerate(tableau):
        if eq_i != i:
            multiplier = np.array(new_tableau[i]) * tableau[eq_i][j]
            new_tableau[eq_i] = np.array(tableau[eq_i]) - multiplier

    return new_tableau


def is_basic(column):
    return sum(column) == 1 and len([c for c in column if c == 0]) == len(column) - 1


def get_solution(tableau):
    columns = np.array(tableau).T
    solutions = []
    for column in columns[:-1]:
        solution = 0
        if is_basic(column):
            one_index = column.tolist().index(1)
            solution = columns[-1][one_index]
        solutions.append(solution)

    return solutions


def simplex(c, A, b):
    tableau = to_tableau(c, A, b)

    while can_be_improved(tableau):
        pivot_position = get_pivot_position(tableau)
        tableau = pivot_step(tableau, pivot_position)

    return get_solution(tableau)



solution = simplex(c, A, b)
for i, j in enumerate(solution, start=1):
    print(f"x{i}={j}")

solution = [solution[i]*c[i] for i in range(len(solution))]
anime = sum(solution)
if u == 'min':
    print(f"Значение целевой функции: {-anime}")
else:
    print(f"Значение целевой функции: {anime}")




