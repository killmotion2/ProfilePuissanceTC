import sklearn.linear_model
import streamlit as st
import pandas as pd
from pandas.errors import EmptyDataError
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from datetime import datetime, date
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
from PIL import Image


##TODO:Créer la foncitonnalité pour avoir plusieurs graphique sur une même page (posssibilité d'ajouter valeurs statistiques? correlation, )
#Pour partir l'app: streamlit run "C:\Users\User\Desktop\ProfilePuissance Python\StreamlitV2\main.py"

def find_data_from_dates(flt_data, slt_title):
    grouped_data = flt_data.groupby('Date')[slt_title].mean().reset_index()
    grouped_data['Improvement'] = grouped_data[slt_title].pct_change() * 100
    grouped_data['Improvement'].fillna(0, inplace=True)

    fig_combined = go.Figure()

    fig_combined.add_trace(go.Scatter(x=grouped_data['Date'], y=grouped_data[slt_title], line=dict(dash='dot')))

    fig_combined.update_layout(xaxis=dict(title='Date des séances'), yaxis=dict(title=slt_title))

    improvement_table = go.Table(header=dict(values=['Date', 'Changement dans la performance (%)']),
                                cells=dict(values=[grouped_data['Date'], grouped_data['Improvement']]))

    fig_table = go.Figure(data=[improvement_table])

    return fig_combined, fig_table



def find_data_from_single_date(flt_data, selected_title):
    each_set = flt_data['Set order'].unique()
    mean_of_variable_per_set = flt_data.groupby('Set order')[selected_title].mean().tolist()
    mean_of_variable_per_set.reverse()

    improvements = [(mean_of_variable_per_set[i] - mean_of_variable_per_set[i - 1]) / mean_of_variable_per_set[i - 1] * 100 if i > 0 else 0 for i in range(len(mean_of_variable_per_set))]
    improvements = [0 if abs(imp) < 0.001 else imp for imp in improvements]
    improvements = [round(imp, 2) for imp in improvements]

    improvement = pd.DataFrame({'Set order': each_set, 'Improvement': improvements})

    fig_combined = go.Figure()
    fig_combined.add_trace(go.Scatter(x=each_set, y=mean_of_variable_per_set, line=dict(dash='dot')))
    fig_combined.update_layout(xaxis=dict(title='# de séries'), yaxis=dict(title=selected_title))

    improvement_table = go.Table(header=dict(values=['Set order', 'Changement dans la performance (%)']),
                                 cells=dict(values=[improvement['Set order'], improvement['Improvement']]))

    fig_table = go.Figure(data=[improvement_table])
    fig_table.update_layout(margin=dict(t=0, b=0))

    return fig_combined, fig_table



def equation(x, a, b, c):
    return a * x**2 + b * x + c

def estimer_charge(vitesse_moyenne_x, f0):
    return (-5.961 * vitesse_moyenne_x**2 - 50.71 * vitesse_moyenne_x + 117.0)/100*f0

def afficher_graphique_FV():
    flt_data = df[(df['User'] == selected_user) & (df['Exercise'] == selected_exercice)]
    F0 = flt_data['Estimated 1RM [lb]'].unique().max()
    each_set = flt_data['Set order'].unique()
    vit_moyenne_x = [0] + [flt_data[flt_data['Set order'] == set]["Avg. velocity [m/s]"].mean() for set in each_set]
    force_moyenne_y = [F0] + [flt_data[flt_data['Set order'] == set]["Load [lb]"].mean() for set in each_set]

    fig = go.Figure(data=go.Scatter(x=vit_moyenne_x, y=force_moyenne_y, mode='markers', name="Valeurs existantes"))

    nouvelles_vitesses = [0.3, 0.5, 0.7]
    nouvelles_charges = [estimer_charge(vitesse, F0) for vitesse in nouvelles_vitesses]
    if st.checkbox("Ajouter les valeurs estimées dans la prédiction de la courbe"):
        vit_moyenne_x.extend(nouvelles_vitesses)
        force_moyenne_y.extend(nouvelles_charges)
        fig.add_trace(go.Scatter(x=nouvelles_vitesses, y=nouvelles_charges, mode='markers', name="Valeurs estimées"))

    popt, pcov = curve_fit(equation, vit_moyenne_x, force_moyenne_y)
    a_opt, b_opt, c_opt = popt

    trend_x = np.linspace(min(vit_moyenne_x), max(vit_moyenne_x), 100)
    trend_y = equation(trend_x, *popt)

    fig.add_trace(go.Scatter(x=trend_x, y=trend_y, mode='lines', name='Courbe de tendance'))

    vit_moyenne_x = np.array(
        [0] + [flt_data[flt_data['Set order'] == set]["Avg. velocity [m/s]"].mean() for set in each_set])
    force_moyenne_y = np.array([F0] + [flt_data[flt_data['Set order'] == set]["Load [lb]"].mean() for set in each_set])

    equation_text = f"Eq : {a_opt:.2f}x^2 + {b_opt:.2f}x + {c_opt:.2f}"
    fig.add_annotation(x=0.85, y=0.9, xref='paper', yref='paper', text=equation_text, showarrow=False)

    residuals = force_moyenne_y - equation(vit_moyenne_x, a_opt, b_opt, c_opt)
    ss_residual = np.sum(residuals ** 2)
    ss_total = np.sum((force_moyenne_y - np.mean(force_moyenne_y)) ** 2)
    r_squared = 1 - (ss_residual / ss_total)

    # Ajouter le coefficient de détermination comme annotation
    r_squared_text = f"R2 : {r_squared:.2f}"
    fig.add_annotation(x=0.85, y=0.85, xref='paper', yref='paper', text=r_squared_text, showarrow=False)

    fig.update_layout(
        xaxis_title='Vitesse moyenne (m/s)',
        yaxis_title='Force moyenne (lb)',
        title='Relation Force-Vitesse',
        showlegend=True,
        xaxis=dict(gridcolor='lightgray'),
        yaxis=dict(gridcolor='lightgray')
    )

    st.plotly_chart(fig)

    exercice_UB = ["Bench", "Pull", "Row", "Press"]
    exercice_LB = ["Squat", "Deadlift", "Lunge", "Fente"]

    is_UB_exercice = any(exo in selected_exercice for exo in exercice_UB)
    is_LB_exercice = any(exo in selected_exercice for exo in exercice_LB)

    training_type = st.selectbox("Type d'entraînement",
                                 ["Force absolue", "Force accélération", "Force-vitesse", "Vitesse-force",
                                  "Vitesse absolue"])

    v_min, v_max = 0, 0
    if is_UB_exercice:
        if training_type == "Force absolue":
            v_min, v_max = 0.15, 0.5
        elif training_type == "Force accélération":
            v_min, v_max = 0.5, 0.75
        elif training_type == "Force-vitesse":
            v_min, v_max = 0.75, 1
        elif training_type == "Vitesse-force":
            v_min, v_max = 1, 1.3
        elif training_type == "Vitesse absolue":
            v_min, v_max = 1.3, 1.5
    elif is_LB_exercice:
        if training_type == "Force absolue":
            v_min, v_max = 0.3, 0.5
        elif training_type == "Force accélération":
            v_min, v_max = 0.5, 0.75
        elif training_type == "Force-vitesse":
            v_min, v_max = 0.75, 1
        elif training_type == "Vitesse-force":
            v_min, v_max = 1, 1.5
        elif training_type == "Vitesse absolue":
            v_min, v_max = 1.3, 1.8
    else:
        if training_type == "Force absolue":
            v_min, v_max = 0.3, 0.5
        elif training_type == "Force accélération":
            v_min, v_max = 0.5, 0.75
        elif training_type == "Force-vitesse":
            v_min, v_max = 0.75, 1
        elif training_type == "Vitesse-force":
            v_min, v_max = 1, 1.5
        elif training_type == "Vitesse absolue":
            v_min, v_max = 1.3, 1.8

    estimated_load_max = int(round(equation(v_min, a_opt, b_opt, c_opt)))
    estimated_load_min = int(round(equation(v_max, a_opt, b_opt, c_opt)))

    estimated_load_min = max(0, estimated_load_min)
    estimated_load_max = max(0, estimated_load_max)

    st.write("Intervalles de vitesses", v_min, "-", v_max, "m/s")
    st.write("Intervalles de charges:", estimated_load_max, "-", estimated_load_min, "lbs")



def analyze_data(user,exercise, start_date, end_date):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    filtered_data = df[(df['User'] == user) & (df['Date'].between(start_date, end_date))& (df['Exercise'] == exercise)]
    if analyzing_multiple_date:
        fig, tabl = find_data_from_dates(filtered_data,selected_title)
        st.header(f"- {selected_title} dans le temps")
    else:
        fig, tabl = find_data_from_single_date(filtered_data, selected_title)
        st.header(f"- {selected_title} à chaque série")
    st.plotly_chart(fig)
    st.plotly_chart(tabl)








##VARIABLES IMPORTANTES
if 'lecture_courbe_FV' not in st.session_state:
        st.session_state.lecture_courbe_FV = False

analyzing = False
analyzing_multiple_date = False


##ILLUSTRATION INITIALE
st.set_page_config(page_title="Analyse profile athlètes de Tennis Canada", page_icon= ":bar_chart:", layout="wide")
emplacement_logo = st.empty()
main_title = st.empty()






##IMPORTATION DES DONNÉES/MAIN LOOP

upload_file = st.sidebar.file_uploader("Pick a TSV file", type='tsv')


if upload_file is not None:
    emplacement_logo.empty()
    main_title.empty()
    try:
        df = pd.read_table(upload_file)
        df['Date'] = pd.to_datetime(df['Date'])

        if st.sidebar.checkbox("Tableau origine"):
            st.dataframe(df)
        titres = df.columns.tolist()


        selected_user = st.sidebar.selectbox('Sélectionner un utilisateur', df['User'].unique())

        exp_analyse_generale = st.sidebar.button("Analyse générale")
        if exp_analyse_generale.__bool__():

            col1, col2, col3 = st.columns(3)
            start_date = datetime.combine(pd.to_datetime(df[(df['User'] == selected_user)]['Date'].unique().min()),datetime.min.time()).date()
            end_date = datetime.combine(pd.to_datetime(df[(df['User'] == selected_user)]['Date'].unique().max()),datetime.max.time()).date()
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)

            filtered_data_Hexbar = df[(df['User'] == selected_user) & (df['Date'].between(start_date, end_date)) & (df['Exercise'] == "Deadlift Hex bar")]
            #filtered_data_Db_press = df[(df['User'] == selected_user) & (df['Date'].between(start_date, end_date)) & (df['Exercise'] == "Flywheel row")]
            filtered_data_Flywheel_row_R = df[(df['User'] == selected_user) & (df['Date'].between(start_date, end_date)) & (df['Exercise'] == "Flywheel row (R)")]
            filtered_data_Flywheel_row_L = df[(df['User'] == selected_user) & (df['Date'].between(start_date, end_date)) & (df['Exercise'] == "Flywheel row (L)")]

            with col1:
                st.header("RFD (N/s) Hexbar DL")
                fig1 = find_data_from_dates(filtered_data_Hexbar, "Peak RFD [N/s]")
                st.plotly_chart(fig1)
            with col2:
                st.header("RFD (N/s) Flywheel (R&L)")
                fig2R = find_data_from_dates(filtered_data_Flywheel_row_R, "Peak RFD [N/s]")
                fig2L = find_data_from_dates(filtered_data_Flywheel_row_L, "Peak RFD [N/s]")
                fig_combined = go.Figure(fig2R.data[0])

                # Add the traces from fig2L to the combined figure
                for trace in fig2L.data:
                    fig_combined.add_trace(trace)

                st.plotly_chart(fig_combined)


        exp_analyse_detaille = st.sidebar.expander("Analyse détaillé",expanded=False)

        if exp_analyse_detaille.expanded:
            selected_exercice = exp_analyse_detaille.selectbox('Sélectionner un exercice',
                                                               df[(df['User'] == selected_user)]['Exercise'].unique())
            sorted(titres)

            # Le reste du contenu à l'intérieur de l'expander
            excluded_titles = ['Date', 'Time', 'User', 'Exercise', 'Set order', 'Rep order']
            selected_title = exp_analyse_detaille.selectbox("Sélectionnez une variable à analyser",[title for title in titres if title not in excluded_titles and df[(df['User'] == selected_user) & ( df['Exercise'] == selected_exercice)][title].notnull().any()])

            if exp_analyse_detaille.button("Courbe force-vitesse", key="courbe_fv"):
                st.session_state.lecture_courbe_FV = not st.session_state.lecture_courbe_FV

            if st.session_state.lecture_courbe_FV:
                afficher_graphique_FV()


            if exp_analyse_detaille.checkbox("Analyse de plusieurs séances?"):
                analyzing_multiple_date = True
                selected_start_date = exp_analyse_detaille.selectbox('Date de début',options=[date.strftime('%Y-%m-%d') for date in df[(df['User'] == selected_user) & (df['Exercise'] == selected_exercice)]['Date'].unique()])
                min_end_date = df[(df['User'] == selected_user) & (df['Exercise'] == selected_exercice) & (df['Date'] >= pd.to_datetime(selected_start_date))]['Date'].min()
                selected_end_date = exp_analyse_detaille.selectbox('Date de fin',options=[date.strftime('%Y-%m-%d') for date in df[(df['User'] == selected_user) & (df['Exercise'] == selected_exercice) & (df['Date'] >= min_end_date)]['Date'].unique()])
                start_date = datetime.combine(pd.to_datetime(selected_start_date), datetime.min.time()).date()
                end_date = datetime.combine(pd.to_datetime(selected_end_date), datetime.max.time()).date()
                analyze_data(selected_user, selected_exercice, start_date, end_date)
            else:
                analyzing_multiple_date = False
                selected_date_single = exp_analyse_detaille.selectbox('Date de début', options=[date.strftime('%Y-%m-%d') for date in df[(df['User'] == selected_user) & (df['Exercise'] == selected_exercice)]['Date'].unique()])
                end_date = start_date = datetime.combine(pd.to_datetime(selected_date_single),
                                                         datetime.max.time()).date()
                analyze_data(selected_user, selected_exercice, start_date, end_date)




    except pd.errors.EmptyDataError:
        st.error("Le fichier est vide ou ne contient pas de colonnes.")

else:
    logo = Image.open("C:/Users/User/Desktop/ProfilePuissance Python/Images Streamlit/Logo_Tennis_Canada.png")
    emplacement_logo.image(logo, width=100)
    main_title.header("Analyse du profile de puissance des athlètes de tennis", )
    st.sidebar.warning("Veuillez télécharger un fichier TSV valide.")
