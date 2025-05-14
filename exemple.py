import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cvxpy as cp

# ----- Interface Moderne -----
st.set_page_config(page_title="Optimisation Multi-Objectif", layout="wide")

with st.sidebar:
    selected = option_menu(
        menu_title="Navigation",
        options=["Formulaire", "Résultats", "Front de Pareto"],
        icons=["pencil", "bar-chart", "graph-up"],
        menu_icon="cast",
        default_index=0
    )

st.title("🎯 Optimisation Multi-Objectif: Branch and Bound + Contrainte ε")

if selected == "Formulaire":
    st.header("📝 Données du Problème")
    n = st.number_input("Nombre d'objets", min_value=1, value=4)

    with st.form("formulaire"):
        poids = st.text_area("Poids des objets (séparés par des virgules)", "2, 3, 4, 5")
        util1 = st.text_area("Utilité fonction 1", "3, 4, 5, 8")
        util2 = st.text_area("Utilité fonction 2", "4, 5, 6, 7")
        capacite = st.number_input("Poids maximal autorisé", min_value=1, value=10)
        epsilon = st.number_input("Valeur de ε (contrainte sur f2)", min_value=0, value=10)
        submitted = st.form_submit_button("Valider")

    if submitted:
        poids = np.array(list(map(float, poids.split(','))))
        util1 = np.array(list(map(float, util1.split(','))))
        util2 = np.array(list(map(float, util2.split(','))))
        st.session_state['data'] = {
            'poids': poids,
            'util1': util1,
            'util2': util2,
            'capacite': capacite,
            'epsilon': epsilon
        }
        st.success("✅ Données enregistrées")

if selected == "Résultats":
    if 'data' in st.session_state:
        data = st.session_state['data']
        poids, util1, util2, capacite, epsilon = data.values()
        st.subheader("📊 Résultat de l'optimisation")

        x = cp.Variable(len(poids), boolean=True)
        contraintes = [poids @ x <= capacite, util2 @ x >= epsilon]
        objectif = cp.Maximize(util1 @ x)
        prob = cp.Problem(objectif, contraintes)
        prob.solve(solver=cp.GLPK_MI)

        st.write("Valeur optimale de f1:", prob.value)
        st.write("Solution optimale:", x.value)

        df = pd.DataFrame({
            "Objet": list(range(1, len(poids)+1)),
            "Sélectionné": np.round(x.value).astype(int),
            "Poids": poids,
            "Utilité f1": util1,
            "Utilité f2": util2
        })
        st.dataframe(df)
    else:
        st.warning("Veuillez d'abord remplir le formulaire.")

if selected == "Front de Pareto":
    if 'data' in st.session_state:
        data = st.session_state['data']
        poids, util1, util2, capacite, _ = data.values()

        st.subheader("📈 Front de Pareto approximatif")
        epsilons = np.linspace(0, sum(util2), 20)
        f1_vals = []
        f2_vals = []

        for eps in epsilons:
            x = cp.Variable(len(poids), boolean=True)
            contraintes = [poids @ x <= capacite, util2 @ x >= eps]
            objectif = cp.Maximize(util1 @ x)
            prob = cp.Problem(objectif, contraintes)
            try:
                prob.solve(solver=cp.GLPK_MI)
                f1_vals.append(prob.value)
                f2_vals.append((util2 @ x).value)
            except:
                pass

        plt.figure(figsize=(8, 5))
        plt.plot(f2_vals, f1_vals, marker='o', color='blue')
        plt.xlabel("Utilité f2")
        plt.ylabel("Utilité f1")
        plt.title("Front de Pareto")
        st.pyplot(plt.gcf())
    else:
        st.warning("Veuillez d'abord remplir le formulaire.")
