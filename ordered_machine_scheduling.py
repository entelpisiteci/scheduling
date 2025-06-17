import pyomo.environ as pyo
from pyomo import environ as pe
import pandas as pd

veri_dosyasi = r"veri.xlsx"
df_siparis = pd.read_excel(veri_dosyasi, sheet_name="Sipariş Bilgileri")
df_makine = pd.read_excel(veri_dosyasi, sheet_name="Makine Bilgileri")

J = df_siparis["J"].tolist()
M = df_makine["M"].tolist()

P = df_siparis.set_index('J')['P_j'].to_dict()
B = df_siparis.set_index('J')['B_j'].to_dict()
T = df_siparis.set_index('J')['T_j'].to_dict()
R = df_makine.set_index('M')['R_m'].to_dict()
W = df_makine.set_index('M')['W_m'].to_dict()
H = df_makine.set_index('M')['H_m'].to_dict()

U = {}
for index, row in df_siparis.iterrows():
    for m in M:
        U[(row["J"], m)] = (row[df_makine[df_makine["M"] == m]["Görevi / Tipi"].iloc[0]] == 1)

L = 100000

model = pyo.ConcreteModel()

model.J = pyo.Set(initialize=J)
model.M = pyo.Set(initialize=M)
model.JxJ = pyo.Set(initialize=[(i, j) for i in J for j in J if i != j])
model.JxJxM = pyo.Set(initialize=[(i, j, m) for i in J for j in J for m in M if i != j])
model.JxM = pyo.Set(initialize=[(j, m) for j in J for m in M])

model.q = pyo.Var(model.J, model.M, within=pyo.Integers, initialize=0)
model.y = pyo.Var(model.J, model.M, within=pyo.Binary, initialize=0)
model.Q = pyo.Var(model.JxJxM, within=pyo.Binary, initialize=0)
model.C = pyo.Var(model.J, model.M, within=pyo.NonNegativeReals, initialize=0)
model.C_global = pyo.Var(model.J, within=pyo.NonNegativeReals, initialize=0)
model.D = pyo.Var(model.J, within=pyo.NonNegativeReals, initialize=0)
model.Q0 = pyo.Var(model.J, model.M, domain=pyo.Binary)  # First job on machine

def objective_rule(model):
    return sum(model.D[j] for j in model.J)
model.obj = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

def constraint_1(model, j):
    return sum(model.q[j, m] for m in model.M) == P[j]
model.constraint_1 = pyo.Constraint(model.J, rule=constraint_1)

def constraint_2(model, j, m):
    return model.q[j, m] <= U[j, m] * P[j]
model.constraint_2 = pyo.Constraint(model.J, model.M, rule=constraint_2)

def constraint_3(model, m):
    return (sum(model.q[j, m] / R[m] for j in model.J) +
            sum(H[m] * model.Q[i, j, m] for i, j in model.JxJ if B[j] != B[i])) <= W[m]
model.constraint_3 = pyo.Constraint(model.M, rule=constraint_3)

def constraint_4(model, i, j, m):
    if B[i] != B[j]:
        return model.C[j, m] >= model.C[i, m] + model.q[j, m] / R[m] + H[m] - L * (1 - model.Q[i, j, m])
    return pyo.Constraint.Skip
model.constraint_4 = pyo.Constraint(model.JxJxM, rule=constraint_4)

def constraint_5(model, i, j, m):
    if B[i] == B[j]:
        return model.C[j, m] >= model.C[i, m] + model.q[j, m] / R[m] - L * (1 - model.Q[i, j, m])
    return pyo.Constraint.Skip
model.constraint_5 = pyo.Constraint(model.JxJxM, rule=constraint_5)

def constraint_6(model, j, m):
    return model.C_global[j] >= model.C[j, m]
model.constraint_6 = pyo.Constraint(model.J, model.M, rule=constraint_6)

def constraint_7(model, j):
    return model.D[j] >= model.C_global[j] - T[j]
model.constraint_7 = pyo.Constraint(model.J, rule=constraint_7)

def constraint_8(model, j, m):
    return model.y[j, m] <= model.q[j, m]
model.constraint_8 = pyo.Constraint(model.J, model.M, rule=constraint_8)

def constraint_9(model, j, m):
    return model.q[j, m] <= L * model.y[j, m]
model.constraint_9 = pyo.Constraint(model.J, model.M, rule=constraint_9)

def constraint_10(model, i, m):
    return sum(model.Q[i, j, m] for j in model.J if i != j) <= model.y[i, m]
model.constraint_10 = pyo.Constraint(model.J, model.M, rule=constraint_10)

def constraint_11(model, i, m):
    return sum(model.Q[j, i, m] for j in model.J if j != i) <= model.y[i, m]
model.constraint_11 = pyo.Constraint(model.J, model.M, rule=constraint_11)

def constraint_12(model, i, m):
    return sum(model.Q[i,j,m] for j in model.J if j != i) + model.Q0[i,m] == model.y[i,m]
model.constraint_12 = pyo.Constraint(model.J, model.M, rule=constraint_12)

def constraint_13(model, m):
    return sum(model.Q0[i, m] for i in J) == 1
model.constraint_13 = pyo.Constraint(model.M, rule=constraint_13)

solver = pyo.SolverFactory('glpk')
results = solver.solve(model)

if results.solver.status == pyo.SolverStatus.ok and results.solver.termination_condition == pyo.TerminationCondition.optimal:
    print("Optimal çözüm bulundu!")
    print(f"Toplam gecikme: {pyo.value(model.obj):.2f}")
    for j in model.J:
        print(f"Sipariş {j}:")
        print(f"  Global bitiş zamanı: {pyo.value(model.C_global[j]):.2f} dakika")
        print(f"  Gecikme: {pyo.value(model.D[j]):.2f} dakika")
        for m in model.M:
            if pyo.value(model.q[j, m]) > 0:
                for i in model.J:
                    if i != j:
                        if pyo.value(model.Q[i,j,m]) > 0:
                            print(f"-----{j} Siparişinden hemen önce başlayan iş: {i}")
                print(f"  Makine {m}: {pyo.value(model.q[j, m]):.2f} adet, Bitiş: {pyo.value(model.C[j, m]):.2f}")
else:
    print(f"Optimal çözüm bulunamadı. Durum: {results.solver.termination_condition}")