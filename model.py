import pandas as pd
from pyomo import environ as pe
from pyomo.util.model_size import build_model_size_report
from pyomo.environ import value
import time
import os

def load_data(file_path):
    speakers_df = pd.read_excel(file_path, sheet_name="speakers")
    schools_df = pd.read_excel(file_path, sheet_name="schools")
    province_groups = pd.read_excel(file_path, sheet_name="province groups")
    return speakers_df, schools_df, province_groups

def process_target_groups(speakers_df, schools_df):
    speakers_df["Target Group"] = speakers_df["Target Group"].replace({
        "Primary school students": 1,
        "Middle school students": 2,
        "High school and equivalent students": 3
    })

    schools_df["Target Group"] = schools_df["Target Group"].replace({
        "Primary school students": 1,
        "Middle school students": 2,
        "High school and equivalent students": 3
    })

    return speakers_df, schools_df

def build_model(K, O, S, A, H, T, Tk, stage):
    model = pe.ConcreteModel()
    model.k = pe.Set(initialize=K)
    model.o = pe.Set(initialize=O)
    model.s = pe.Set(initialize=S)
    model.a = pe.Set(initialize=A)
    model.h = pe.Set(initialize=H)
    model.t = pe.Set(initialize=T)

    model.x_ind = pe.Set(initialize=list((k, o, t) for k in K for o in O for t in list(range(1, Tk.loc[k, "Preference Count"] + 1))))
    model.y_ind = pe.Set(initialize=list((k, t) for k in K for t in list(range(1, Tk.loc[k, "Preference Count"] + 1))))

    model.x = pe.Var(model.x_ind, domain=pe.Binary)

    if stage >= 1:
        model.z = pe.Var(model.o, domain=pe.Binary)
    if stage >= 2:
        model.v = pe.Var(model.y_ind, domain=pe.Binary)
    if stage >= 3:
        model.w = pe.Var(model.k, domain=pe.Binary)
    if stage >= 4:
        model.y = pe.Var(model.y_ind, domain=pe.Binary)

    return model

def add_constraints_and_objective(model, K, O, Tk, stage, obj1, obj2, obj3, obj4, obj5):
    # Objective Functions
    if stage == 0:
        obj = sum(model.x[k, o, t] for k in K for o in O for t in list(range(1, Tk.loc[k, "Preference Count"] + 1)))
    elif stage == 1:
        obj = sum(model.z[o] for o in O)
    elif stage == 2:
        obj = sum(model.v[k, t] for k in K for t in list(range(1, Tk.loc[k, "Preference Count"] + 1)))
    elif stage == 3:
        obj = sum(model.w[k] for k in K)
    elif stage == 4:
        obj = sum(model.y[k, t] for k in K for t in list(range(1, Tk.loc[k, "Preference Count"] + 1)))
    if stage == 5:
        model.obj = pe.Objective(sense=pe.minimize, expr=obj)
    else:
        model.obj = pe.Objective(sense=pe.maximize, expr=obj)

    # Constraints
    model.c = pe.ConstraintList()
    return model

# Solve the model
def solve_model(model):
    solver_manager = pe.SolverManagerFactory('neos')
    results = solver_manager.solve(model, solver="cplex", load_solutions=True)
    return results

# Save results
def save_results(filtered_df, province_group_no):
    filtered_df.to_excel(f"{province_group_no}_results.xlsx")

# Save solution report
def save_solution_report(school_df, speaker_df, obj1, province_group_no):
    solution_report = {
        "school application count": len(school_df),
        "unique school count": len(school_df["Application ID"].unique()),
        "speaker application count": len(speaker_df),
        "unique speaker count": len(speaker_df["Project Id"].unique()),
        "matched discussion count": obj1
    }
    solution_report = pd.DataFrame(solution_report, index=[0]).T
    solution_report.to_excel(f"{province_group_no}_solution_report.xlsx")

# Main program
def main():
    os.environ['NEOS_EMAIL'] = '<VALID_EMAIL_ADDRESS>'
    data_location = "data.xlsx"

    speakers_df, schools_df, province_groups = load_data(data_location)
    speakers_df, schools_df = process_target_groups(speakers_df, schools_df)

    BigM = 100000
    for province_group_no in range(1, 7):
        print(f"Solutions for province group {province_group_no} are being generated")

        provinces = list(province_groups[province_groups["Group No"] == province_group_no]["Province"].values)
        province_schools = schools_df[schools_df["Province"].isin(provinces)]
        province_speakers = speakers_df[speakers_df["Province"].isin(provinces)]
        
        K = list(province_speakers["Project Id"].unique())
        O = list(province_schools["Application ID"].unique())
        S = list(set(list(province_speakers["Province"].unique()) + list(province_schools["Province"].unique())))
        A = list(set(list(province_speakers["Field"].unique()) + list(province_schools["Field"].unique())))
        H = list(set(list(province_speakers["Target Group"].unique()) + list(province_schools["Target Group"].unique())))
        T = list(range(1, 6))
        Tk = pd.DataFrame({"Preference Count": [3] * len(K)}, index=K)  # Example data, can be replaced with real data

        for stage in range(2):
            print(f"Stage {stage} solution is being generated")
            start_time = time.time()
            
            model = build_model(K, O, S, A, H, T, Tk, stage)
            model = add_constraints_and_objective(model, K, O, Tk, stage, 0, 0, 0, 0, 0)
            
            print(f"Model building started, elapsed time: {time.time() - start_time}")

            results = solve_model(model)
            print(f"Time taken to solve the model: {time.time() - start_time}")
            
            solution = model.x.extract_values()
            df = pd.DataFrame(solution, index=[0]).T
            filtered_df = df[df[0] == 1]

            save_results(filtered_df, province_group_no)
            save_solution_report(province_schools, province_speakers, 0, province_group_no)

            time.sleep(600)

if __name__ == "__main__":
    main()
