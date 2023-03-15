from scipy.spatial import HalfspaceIntersection, ConvexHull
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
import math

from scipy.optimize import linprog






q = int(input("Количество ограничений в системе:"))
w = int(input("Количество переменных x в задаче:"))
e = q + w



obj = [0]*w

c = [0]*e
b = [0] * q
lhs_ineq = []
rhs_ineq = []
lhs_eq = []
rhs_eq = []

A = [[0]*e for i in range(q)]


print("Введите значения коэффициентов целевой функции:")
for i in range(w):
    top = float(input(f"x{i+1}="))
    obj[i] = top
    c[i] = top

u = input("Какой тип решения (min или max):")
if u == 'max':
    c = list(map(lambda x:-x, c))
    obj = list(map(lambda x:-x, obj))




print("Введите ограничения:")
for j in range(q):
    x_1 = []
    for i in range(w):
        vitek = float(input(f"x{i + 1}="))
        x_1.append(vitek)
        A[j][i] = vitek
    jotaro = float(input(f"b{j+1}="))
    b[j] = jotaro
    # loli = list(map(float, input("Введите коэффициенты ограничения:").split()))
    # f = float(input("Введите значение правой части:"))
    lol = input("Является ли ограничение уравнением: yes or no:")
    if lol == "yes":
        lhs_eq.append(x_1)
        rhs_eq.append(jotaro)
    else:
        rut = input("Введите y если ограничение <=:")
        if rut == 'y':
            A[j][w + j] = 1.0
        else:
            x_1 = list(map(lambda x: -x, x_1))
            jotaro *= -1
            A[j][w + j] = -1.0
        lhs_ineq.append(x_1)
        rhs_ineq.append(jotaro)

    # for i in range(w):
    #     vitek = float(input(f"x{i + 1}="))
    #     lhs_ineq[j][i] = vitek
    #     A[j][i] = vitek
    # jotaro = float(input(f"b{j+1}="))
    # rhs_ineq[j] = jotaro
    # b[j] = jotaro
    # rut = input("Введите y если ограничение <=:")
    # if rut == 'y':
    #     A[j][w+j] = 1.0
    # else :
    #     lhs_ineq[j] = list(map(lambda x: -x, lhs_ineq[j]))
    #     rhs_ineq[j] *= -1
    #     A[j][w+j] = -1.0


freedom = int(input("Введите значение свободного члена:"))

# bnd = [(0, float("inf")),  # Границы x
#        (0, float("inf"))]  # Границы y




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
        raise Exception("Функция не ограничена")

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



try:




    print(obj)
    print(lhs_ineq)
    print(rhs_ineq)
    print(lhs_eq)
    print(rhs_eq)


    opt = linprog(c=obj, A_ub=lhs_ineq or None, b_ub=rhs_ineq or None, A_eq=lhs_eq or None, b_eq=rhs_eq or None,
                  method="highs")
    if opt.success:

        print(opt)
        print(opt.x)
        if u == 'max':
           print(f"Значение целевой функции: {(opt.fun)*-1 + freedom}")
        else:
           print(f"Значение целевой функции: {(opt.fun) + freedom}")

        solution = simplex(c, A, b)

        solution = [solution[i] * c[i] for i in range(len(solution))]
        anime = sum(solution)
    else:
        print("Функция не ограничена. Оптимальное решение отсутствует.")


except Exception as jojo:
    print(jojo)







