from pyomo.environ import *
from pyomo.opt import SolverFactory

# Initialize model
model = ConcreteModel()

# Sets
model.J = Set(initialize=[1, 2, 3, 4, 5])  # Jobs
model.I = Set(initialize=[1, 2])  # Machines

# Parameters
model.w = Param(model.J, initialize={1: 1, 2: 3, 3: 4, 4: 5, 5: 6})  # Job weights
model.oncelik = Param(model.J, model.J, initialize={
    (1,1): 0, (1,2): 0, (1,3): 0, (1,4): 0, (1,5): 1,
    (2,1): 0, (2,2): 0, (2,3): 0, (2,4): 0, (2,5): 0,
    (3,1): 1, (3,2): 0, (3,3): 0, (3,4): 0, (3,5): 0,
    (4,1): 0, (4,2): 0, (4,3): 1, (4,4): 0, (4,5): 0,
    (5,1): 0, (5,2): 0, (5,3): 0, (5,4): 0, (5,5): 0
})  # Precedence constraints
model.p = Param(model.J, model.I, initialize={
    (1,1): 2, (1,2): 5,
    (2,1): 5, (2,2): 3,
    (3,1): 4, (3,2): 5,
    (4,1): 3, (4,2): 2,
    (5,1): 8, (5,2): 2
})  # Processing times
model.d = Param(model.J, initialize={1: 20, 2: 30, 3: 18, 4: 22, 5: 50})  # Due dates
model.S = Param(model.J, model.J, model.I, initialize={
    (j,k,i): 2 if i == 1 else 3 for j in model.J for k in model.J for i in model.I
})  # Setup times
model.S0 = Param(model.J, model.I, initialize={
    (j,i): 2 if i == 1 else 3 for j in model.J for i in model.I
})  # Initial setup times
model.u = Param(model.J, model.I, initialize={
    (1,1): 1, (1,2): 1,
    (2,1): 1, (2,2): 0,
    (3,1): 1, (3,2): 1,
    (4,1): 0, (4,2): 1,
    (5,1): 1, (5,2): 1
})  # Machine eligibility
model.R = Param(model.I, initialize={1: 2, 2: 4})  # Machine ready times
model.SM = Param(model.I, initialize={1: 18, 2: 27})  # Machine unavailable start
model.FM = Param(model.I, initialize={1: 22, 2: 33})  # Machine unavailable end
model.BigM = Param(initialize=10000000)  # Big M value

# Variables
model.x = Var(model.J, domain=NonNegativeReals)  # Job start times
model.C = Var(model.J, domain=NonNegativeReals)  # Job completion times
model.E = Var(model.J, domain=NonNegativeReals)  # Earliness
model.T = Var(model.J, domain=NonNegativeReals)  # Tardiness
model.A = Var(model.J, model.I, domain=Binary)  # Job-machine assignment
model.Z = Var(model.J, model.J, model.I, domain=Binary)  # Job sequence on machine
model.Z0 = Var(model.J, model.I, domain=Binary)  # First job on machine
model.y = Var(model.J, domain=Binary)  # Job after unavailable period

# Objective function
model.obj = Objective(expr=sum(model.E[j] + model.T[j] for j in model.J), sense=minimize)

# Constraints
def job_assignment_rule(model, j):
    return sum(model.u[j,i] * model.A[j,i] for i in model.I) == 1
model.job_assignment = Constraint(model.J, rule=job_assignment_rule)

def machine_eligibility_rule(model, j, i):
    return model.A[j,i] <= model.u[j,i]
model.machine_eligibility = Constraint(model.J, model.I, rule=machine_eligibility_rule)

def predecessor_rule(model, i, k):
    return sum(model.Z[j,k,i] for j in model.J if j != k) + model.Z0[k,i] == model.A[k,i]
model.predecessor = Constraint(model.I, model.J, rule=predecessor_rule)

def successor_rule(model, i, k):
    return sum(model.Z[k,j,i] for j in model.J if j != k) <= model.A[k,i]
model.successor = Constraint(model.I, model.J, rule=successor_rule)

def first_job_rule(model, i):
    return sum(model.Z0[j,i] for j in model.J) <= 1
model.first_job = Constraint(model.I, rule=first_job_rule)

def linearization1_rule(model, i, j, k):
    if j != k:
        return model.Z[j,k,i] <= model.A[j,i]
    return Constraint.Skip
model.linearization1 = Constraint(model.I, model.J, model.J, rule=linearization1_rule)

def linearization2_rule(model, i, j, k):
    if j != k:
        return model.Z[j,k,i] <= model.A[k,i]
    return Constraint.Skip
model.linearization2 = Constraint(model.I, model.J, model.J, rule=linearization2_rule)

def completion_time_rule(model, j):
    return model.C[j] >= model.x[j] + sum(model.Z[k,j,i] * model.S[k,j,i] for i in model.I for k in model.J if k != j) + \
           sum(model.Z0[j,i] * model.S0[j,i] for i in model.I) + sum(model.A[j,i] * model.p[j,i] for i in model.I)
model.completion_time = Constraint(model.J, rule=completion_time_rule)

def start_time_rule(model, j):
    return model.x[j] >= sum(model.A[j,i] * model.R[i] for i in model.I)
model.start_time = Constraint(model.J, rule=start_time_rule)

def unavailable_period1_rule(model, j):
    return model.C[j] <= sum(model.A[j,i] * model.SM[i] + model.BigM * model.y[j] for i in model.I)
model.unavailable_period1 = Constraint(model.J, rule=unavailable_period1_rule)

def unavailable_period2_rule(model, j):
    return model.x[j] >= sum(model.A[j,i] * model.FM[i] - model.BigM * (1 - model.y[j]) for i in model.I)
model.unavailable_period2 = Constraint(model.J, rule=unavailable_period2_rule)

def sequencing_rule(model, j, k):
    if j != k:
        return model.C[j] <= model.x[k] + model.BigM * (1 - sum(model.Z[j,k,i] for i in model.I))
    return Constraint.Skip
model.sequencing = Constraint(model.J, model.J, rule=sequencing_rule)

def earliness_rule(model, j):
    return model.E[j] >= model.d[j] - model.C[j]
model.earliness = Constraint(model.J, rule=earliness_rule)

def tardiness_rule(model, j):
    return model.T[j] >= model.C[j] - model.d[j]
model.tardiness = Constraint(model.J, rule=tardiness_rule)


solver = SolverFactory('glpk')
results = solver.solve(model)
model.display()

