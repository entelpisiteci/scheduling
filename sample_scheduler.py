from pyomo.environ import *

# Create a model
model = ConcreteModel()

students = ['s1', 's2', 's3']  # Student groups
lecturers = ['l1', 'l2', 'l3']  # Lecturers
courses = ['c1', 'c2', 'c3']  # Courses
classrooms = ['r1', 'r2', 'r3']  # Classrooms
periods = range(1, 10)  # 9 periods in a day (1-9)

# Decision Variables
model.x = Var(courses, periods, classrooms, domain=Binary)  # If course 'c' is scheduled in period 'p' in classroom 'r'

# Constraints

# 3.1 No Overlapping (Students)
def no_overlap_students(model, student, period):
    return sum(model.x[c, period, r] for c in courses for r in classrooms) <= 1  # At most one course in a period for each student
model.no_overlap_students = Constraint(students, periods, rule=no_overlap_students)

# 3.1 No Overlapping (Lecturers)
def no_overlap_lecturers(model, lecturer, period):
    return sum(model.x[c, period, r] for c in courses for r in classrooms if lecturer in c) <= 1  # Lecturer should not have overlapping classes
model.no_overlap_lecturers = Constraint(lecturers, periods, rule=no_overlap_lecturers)

# 3.1 No Overlapping (Classrooms)
def no_overlap_classrooms(model, period, classroom):
    return sum(model.x[c, period, classroom] for c in courses) <= 1  # At most one course in a classroom at any given period
model.no_overlap_classrooms = Constraint(periods, classrooms, rule=no_overlap_classrooms)

# 3.3 Consecutiveness (Simplified)
def consecutiveness(model, course):
    return sum(model.x[course, period, r] for period in periods for r in classrooms) == len(periods)
model.consecutiveness = Constraint(courses, rule=consecutiveness)

# 3.4 Classroom Consistency
def classroom_consistency(model, course):
    # All periods of the course should be in the same classroom
    return sum(model.x[course, period, r] for period in periods for r in classrooms) == len(periods)
model.classroom_consistency = Constraint(courses, rule=classroom_consistency)

# Objective: (Minimizing the total number of scheduled periods)
def objective_rule(model):
    return sum(model.x[c, period, r] for c in courses for period in periods for r in classrooms)
model.obj = Objective(rule=objective_rule, sense=minimize)

# Solver
solver = SolverFactory('glpk')
results = solver.solve(model, tee=True)

# Output
for c in courses:
    for p in periods:
        for r in classrooms:
            if model.x[c, p, r].value > 0.5:
                print(f"Course {c} is scheduled in period {p} in classroom {r}")
