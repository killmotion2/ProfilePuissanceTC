import string

import streamlit as st
import pandas as pd
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
from PIL import Image


##TODO: ajouter RFD dans graphique comparatif, changer graphique évolution avec une présentation de plusieurs données dans le temps
# Pour partir l'app: streamlit run "C:\Users\User\Desktop\ProfilePuissance Python\StreamlitV2\main.py"

def find_data_from_dates(flt_data, slt_title):
    grouped_data_raw = flt_data.groupby('Date')[slt_title]
    grouped_data = flt_data.groupby('Date')[slt_title].mean().reset_index()
    grouped_data['Improvement'] = grouped_data[slt_title].pct_change() * 100
    grouped_data['Improvement'].fillna(0, inplace=True)
    grouped_data['Improvement'] = grouped_data['Improvement'].round(2)

    effect_sizes = []
    effect_size_titles = []
    effect_sizes.append(0)
    effect_size_titles.append("")
    for i in range(1, len(grouped_data)):
        m1 = grouped_data[slt_title][i - 1]
        m2 = grouped_data[slt_title][i]
        s = grouped_data_raw.get_group(grouped_data['Date'][i]).std()
        if s != 0:
            effect_size = abs(m1 - m2) / s
        else:
            effect_size = 0  # Valeur par défaut lorsque l'écart-type est nul

        if effect_size < 0.2 * s:
            effect_size_title = "Sans intérêt"
        elif 0.2 * s <= effect_size < 0.5 * s:
            effect_size_title = "Faible"
        elif 0.5 * s <= effect_size < 0.8 * s:
            effect_size_title = "Modéré"
        elif effect_size > 0.8 * s:
            effect_size_title = "Grand"
        else:
            effect_size_title = ""

        effect_sizes.append(effect_size)
        effect_size_titles.append(effect_size_title)

    fig_combined = go.Figure(data=go.Scatter(x=grouped_data['Date'], y=grouped_data[slt_title], line=dict(dash='dot'),
                                             name=f"{selected_title}"))
    fig_combined.update_layout(title=f"{selected_exercice}, {slt_title} dans le temps",
                               xaxis=dict(title='Date des séances'), yaxis=dict(title=slt_title))

    improvement_table = go.Figure(data=go.Table(
        header=dict(values=['Date', 'Changement dans la performance (%)', 'Taille de Cohen']),
        cells=dict(values=[grouped_data['Date'].dt.strftime('%Y-%m-%d'), grouped_data['Improvement'],
                           [f"{np.round(size, 2)} ({title})" for size, title in
                            zip(effect_sizes, effect_size_titles)]]),
    ))
    improvement_table.update_layout(margin=dict(t=0, b=0), width=200)

    return fig_combined, improvement_table


def find_data_from_single_date(flt_data, selected_title):
    each_set = flt_data['Set order'].unique()
    mean_of_variable_per_set = flt_data.groupby('Set order')[selected_title].mean().tolist()
    mean_of_variable_per_set.reverse()

    improvements = [(mean_of_variable_per_set[i] - mean_of_variable_per_set[i - 1]) / mean_of_variable_per_set[
        i - 1] * 100 if i > 0 else 0 for i in range(len(mean_of_variable_per_set))]
    improvements = [0 if abs(imp) < 0.001 else imp for imp in improvements]
    improvements = [round(imp, 2) for imp in improvements]

    improvement = pd.DataFrame({'Set order': each_set, 'Improvement': improvements})

    fig_combined = go.Figure(
        data=go.Scatter(x=each_set, y=mean_of_variable_per_set, line=dict(dash='dot'), name=f"{selected_title}"))
    fig_combined.update_layout(title=f"{selected_exercice}, {selected_title} à chaque série ",
                               xaxis=dict(title='# de séries'), yaxis=dict(title=selected_title))
    fig_combined.update_layout(legend=dict(title=f"{selected_title}"))

    improvement_table = go.Figure(data=go.Table(header=dict(values=['Set order', 'Changement dans la performance (%)']),
                                                cells=dict(
                                                    values=[improvement['Set order'], improvement['Improvement']])))
    improvement_table.update_layout(margin=dict(t=0, b=0), width=150)

    return fig_combined, improvement_table


class CourbeForceVitesse:
    def __init__(self, selected_user, selected_exercice, start_date, end_date):
        self.df = df
        self.selected_user = selected_user
        self.selected_exercice = selected_exercice
        self.date_debut = pd.to_datetime(start_date)
        self.date_fin = pd.to_datetime(end_date)
        self.flt_data = df[(df['User'] == selected_user) & (df['Exercise'] == selected_exercice) & (
            df['Date'].between(self.date_debut, self.date_fin))]
        self.figure = None
        self.popt = None
        self.F0 = None
        self.is_type_of_exercice = None
        self.selectbox = "Force absolue"
        self.string_vitesse_maxmin = None
        self.string_repetition_maxmin = None
        self.string_load_maxmin = None

    def show_graph(self):
        with col1:
            st.plotly_chart(self.figure)

    def show_graphs_info(self):
        with col2:
            st.write(self.string_load_maxmin, self.string_vitesse_maxmin, self.string_repetition_maxmin)

    def equation(self, x, a, b, c):
        return a * x ** 2 + b * x + c

    def estimer_charge_squat(self, vitesse_moyenne_x, f0):
        return (-12.87 * vitesse_moyenne_x ** 2 - 46.31 * vitesse_moyenne_x + 116.3) / 100 * f0

    def estimer_charge_benchpress(self, vitesse_moyenne, f0):
        return (11.4196 * vitesse_moyenne ** 2 - 81.904 * vitesse_moyenne + 114.03) / 100 * f0

    def estimer_charge_tirade(self, vitesse_moyenne, f0):
        return (18.5797 * vitesse_moyenne ** 2 - 104.182 * vitesse_moyenne + 147.94) / 100 * f0

    def create_fv_graph(self, checkbox_show_estimated_data):
        self.F0 = self.flt_data['Estimated 1RM [lb]'].unique().max()
        each_set = self.flt_data['Set order'].unique()
        force_moyenne_y = [self.F0] + [self.flt_data[self.flt_data['Set order'] == set]["Load [lb]"].max() for set in each_set]

        exercice_UB_pull = ["Pull", "Row", "Tirade"]
        exercice_UB_push = ["Bench", "Press"]
        exercice_LB_pull = ["Deadlift"]
        exercice_LB_push = ["squat", "lunge", "fente"]

        is_UB_exercice_pull = any(exo in self.selected_exercice.lower() for exo in exercice_UB_pull)
        is_UB_exercice_push = any(exo in self.selected_exercice.lower() for exo in exercice_UB_push)
        is_LB_exercice_pull = any(exo in self.selected_exercice.lower() for exo in exercice_LB_pull)
        is_LB_exercice_push = any(exo in self.selected_exercice.lower() for exo in exercice_LB_push)
        self.is_type_of_exercice = [is_UB_exercice_pull, is_UB_exercice_push, is_LB_exercice_pull, is_LB_exercice_push]

        vit_moyenne_x = [0.0] + [self.flt_data[self.flt_data['Set order'] == set]["Avg. velocity [m/s]"].mean() for set in
                                 each_set]

        figFV = go.Figure(data=go.Scatter(x=vit_moyenne_x, y=force_moyenne_y, mode='markers', name="Valeurs existantes"))

        nouvelles_vitesses = [0.3, 0.5, 0.7]
        nouvelles_charges = [0.0]

        if checkbox_show_estimated_data:
            if is_UB_exercice_push:
                nouvelles_charges = [self.estimer_charge_benchpress(vitesse, self.F0) for vitesse in nouvelles_vitesses]
            elif is_UB_exercice_pull:
                nouvelles_charges = [self.estimer_charge_tirade(vitesse, self.F0) for vitesse in nouvelles_vitesses]
            elif is_LB_exercice_push or is_LB_exercice_pull:
                nouvelles_charges = [self.estimer_charge_squat(vitesse, self.F0) for vitesse in nouvelles_vitesses]
            for nv in nouvelles_vitesses:
                vit_moyenne_x.append(nv)
            for nc in nouvelles_charges:
                force_moyenne_y.append(nc)
            figFV.add_trace(
                go.Scatter(x=nouvelles_vitesses, y=nouvelles_charges, mode='markers', name="Valeurs estimées"))

        popt = curve_fit(self.equation, vit_moyenne_x, force_moyenne_y)
        self.popt = popt[0]
        a_opt, b_opt, c_opt = popt[0]

        trend_x = np.linspace(min(vit_moyenne_x), max(vit_moyenne_x), 100)
        trend_y = self.equation(trend_x, *popt[0])

        figFV.add_trace(go.Scatter(x=trend_x, y=trend_y, mode='lines', name='Courbe de tendance'))


        vit_moyenne_x = np.array(vit_moyenne_x)
        force_moyenne_y = np.array(force_moyenne_y)
        #vit_moyenne_x = np.array(
        #    [0] + [flt_data[flt_data['Set order'] == set]["Avg. velocity [m/s]"].mean() for set in each_set])
        #force_moyenne_y = np.array(
        #    [self.F0] + [flt_data[flt_data['Set order'] == set]["Load [lb]"].mean() for set in each_set])

        equation_text = f"Eq : {a_opt:.2f}x^2 + {b_opt:.2f}x + {c_opt:.2f}"
        figFV.add_annotation(x=0.85, y=0.9, xref='paper', yref='paper', text=equation_text, showarrow=False)

        residuals = force_moyenne_y - self.equation(vit_moyenne_x, a_opt, b_opt, c_opt)
        ss_residual = np.sum(residuals ** 2)
        ss_total = np.sum((force_moyenne_y - np.mean(force_moyenne_y)) ** 2)
        r_squared = 1 - (ss_residual / ss_total)

        r_squared_text = f"R2 : {r_squared:.2f}"
        figFV.add_annotation(x=0.85, y=0.85, xref='paper', yref='paper', text=r_squared_text, showarrow=False)

        figFV.update_layout(
            xaxis_title='Vitesse moyenne (m/s)',
            yaxis_title='Force moyenne (lb)',
            title='Relation Force-Vitesse',
            showlegend=True,
            xaxis=dict(gridcolor='lightgray'),
            yaxis=dict(gridcolor='lightgray')
        )
        self.figure = figFV

    def create_fv_zone_infos(self):
        a_opt, b_opt, c_opt = self.popt

        v_min, v_max = 0, 0
        if self.is_type_of_exercice[0] or self.is_type_of_exercice[1]:
            if fv_selectbox_choice == "Force absolue":
                v_min, v_max = 0.15, 0.5
            elif fv_selectbox_choice == "Force accélération":
                v_min, v_max = 0.5, 0.75
            elif fv_selectbox_choice == "Force-vitesse":
                v_min, v_max = 0.75, 1
            elif fv_selectbox_choice == "Vitesse-force":
                v_min, v_max = 1, 1.3
            elif fv_selectbox_choice == "Vitesse absolue":
                v_min, v_max = 1.3, 1.5
        elif self.is_type_of_exercice[2] or self.is_type_of_exercice[3]:
            if fv_selectbox_choice == "Force absolue":
                v_min, v_max = 0.3, 0.5
            elif fv_selectbox_choice == "Force accélération":
                v_min, v_max = 0.5, 0.75
            elif fv_selectbox_choice == "Force-vitesse":
                v_min, v_max = 0.75, 1
            elif fv_selectbox_choice == "Vitesse-force":
                v_min, v_max = 1, 1.5
            elif fv_selectbox_choice == "Vitesse absolue":
                v_min, v_max = 1.3, 1.8
        else:
            if fv_selectbox_choice == "Force absolue":
                v_min, v_max = 0.3, 0.5
            elif fv_selectbox_choice == "Force accélération":
                v_min, v_max = 0.5, 0.75
            elif fv_selectbox_choice == "Force-vitesse":
                v_min, v_max = 0.75, 1
            elif fv_selectbox_choice == "Vitesse-force":
                v_min, v_max = 1, 1.5
            elif fv_selectbox_choice == "Vitesse absolue":
                v_min, v_max = 1.3, 1.8

        estimated_load_max = int(round(self.equation(v_min, a_opt, b_opt, c_opt)))
        estimated_load_min = int(round(self.equation(v_max, a_opt, b_opt, c_opt)))

        estimated_load_min = max(0, estimated_load_min)
        estimated_load_max = max(0, estimated_load_max)

        reps_min = int(round((1.0278 * self.F0 - estimated_load_min) / (0.0278 * self.F0)))
        if reps_min < 0: reps_min = 0
        reps_max = int(round((1.0278 * self.F0 - estimated_load_max) / (0.0278 * self.F0)))
        if reps_max < 0: reps_max = 0

        self.string_vitesse_maxmin = f"Intervalles de vitesses : {v_min} - {v_max} m/s"
        self.string_load_maxmin = f"Intervalles de charges : {estimated_load_max} - {estimated_load_min} lbs"
        self.string_repetition_maxmin = f"Nombre de répétitions recommandé : {reps_max} - {reps_min}"



def analyse_data(user, exercise, start_date, end_date):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    filtered_data = df[(df['User'] == user) & (df['Date'].between(start_date, end_date)) & (df['Exercise'] == exercise)]

    if analyzing_multiple_date:
        fig, tabl = find_data_from_dates(filtered_data, selected_title)
    else:
        fig, tabl = find_data_from_single_date(filtered_data, selected_title)

    return fig, tabl


def add_figure_and_update_table():
    lgr_graphs = len(st.session_state.graphs_of_variables)
    st.session_state.graphs_of_variables = [[item[0], ] for item in st.session_state.graphs_of_variables]
    if lgr_graphs > 1:
        old_figure = st.session_state.graphs_of_variables[- 2][0]
        new_figure = st.session_state.graphs_of_variables[-1][0]
        old_figure.add_trace(new_figure.data[0])
        st.session_state.graphs_of_variables.pop()

        old_title = old_figure['layout']['title']['text']
        new_title = new_figure['layout']['title']['text']

        if selected_exercice in old_title and selected_exercice in new_title:
            new_title = new_title.replace(f"{selected_exercice},", "")
        end_title = ["dans le temps", "à chaque série"]
        for t in end_title:
            if t in old_title and t in new_title:
                old_title = old_title.replace(t, "")

        # Créer un nouveau graphique avec sous-tracés
        fig = make_subplots(rows=len(old_figure['data']), cols=1, shared_xaxes=True, vertical_spacing=0.1)

        # Ajouter les traces de l'ancienne figure dans les sous-tracés
        for row, trace in enumerate(old_figure['data'], start=1):
            fig.add_trace(trace, row=row, col=1)
            fig.update_xaxes(title_text=f'', row=row, col=1)
            fig.update_yaxes(title_text=f'', row=row, col=1)

        fig.update_layout(title=f"{old_title}, {new_title}",
                          xaxis=dict(title=""), yaxis=dict(title=""))

        # Mettre à jour la nouvelle figure
        new_figure = go.Figure(fig)

        # Mettre à jour la liste des figures dans la session
        st.session_state.graphs_of_variables[-1][0] = new_figure
    # return original_fig


def graph_manager(graphs_list, expander):
    for i, fig in enumerate(graphs_list):
        with expander:
            graph_title = f"Graphique {i + 1}"
            if st.button(f"{graph_title}     X"):
                graphs_list.pop(i)
                break


##VARIABLES IMPORTANTES
if 'lecture_courbe_FV' not in st.session_state:
    st.session_state.lecture_courbe_FV = False
if 'graphs_of_variables' not in st.session_state:
    st.session_state.graphs_of_variables = []
if 'graphs_of_fv' not in st.session_state:
    st.session_state.graphs_of_fv = []

analyzing = False
analyzing_multiple_date = False
checkbox_show_estimated_data = False

##ILLUSTRATION INITIALE
st.set_page_config(page_title="Analyse profile athlètes de Tennis Canada", page_icon=":bar_chart:", layout="wide")
emplacement_logo = st.empty()
main_title = st.empty()

##IMPORTATION DES DONNÉES/MAIN LOOP

upload_file = st.sidebar.file_uploader("Choisir un fichier TSV", type='tsv')

if upload_file is not None:
    emplacement_logo.empty()
    main_title.empty()
    try:
        df = pd.read_table(upload_file)
        df['Date'] = pd.to_datetime(df['Date'])

        if st.sidebar.checkbox("Tableau origine"):
            st.dataframe(df)
        titres = df.columns.tolist()
        col1, col2 = st.columns([3, 1])

        # Fonctionnalité d'analyse de la courbe force-vitesse
        exp_courbe_FV = st.sidebar.expander("Analyse courbe force-vitesse")
        if exp_courbe_FV.expanded:
            selected_user_fv = exp_courbe_FV.selectbox('Sélectionner un utilisateur', df['User'].unique(),
                                                       key="User_FV")
            selected_exercice_fv = exp_courbe_FV.selectbox('Sélectionner un exercice',
                                                           df[(df['User'] == selected_user_fv)]['Exercise'].unique(),
                                                           key="Exo_FV")

            selected_start_date = exp_courbe_FV.selectbox('Date de début',
                                                          options=[date.strftime('%Y-%m-%d') for date in df[
                                                              (df['User'] == selected_user_fv) & (df[
                                                                                                      'Exercise'] == selected_exercice_fv)][
                                                              'Date'].unique()], key="Start_date_FV")
            min_end_date = df[(df['User'] == selected_user_fv) & (df['Exercise'] == selected_exercice_fv) & (
                    df['Date'] >= pd.to_datetime(selected_start_date))]['Date'].min()
            selected_end_date = exp_courbe_FV.selectbox('Date de fin',
                                                        options=[date.strftime('%Y-%m-%d') for date in df[
                                                            (df['User'] == selected_user_fv) & (df[
                                                                                                    'Exercise'] == selected_exercice_fv) & (
                                                                    df['Date'] >= min_end_date)][
                                                            'Date'].unique()], key="End_date_FV")
            start_date_fv = datetime.combine(pd.to_datetime(selected_start_date), datetime.min.time()).date()
            end_date_fv = datetime.combine(pd.to_datetime(selected_end_date), datetime.max.time()).date()

            if "Flywheel" in selected_exercice_fv or "jump" in selected_exercice_fv:
                exp_courbe_FV.button("Ajouter courbe force-vitesse", disabled=True)
            elif exp_courbe_FV.button("Ajouter courbe force-vitesse", disabled=False):
                graph = CourbeForceVitesse(selected_user_fv, selected_exercice_fv, start_date_fv, end_date_fv)
                graph.create_fv_graph(checkbox_show_estimated_data)
                st.session_state.graphs_of_fv.append(graph)

            checkbox_show_estimated_data = exp_courbe_FV.checkbox(
                "Ajouter les valeurs estimées dans la prédiction de la courbe")
            fv_selectbox_choice = exp_courbe_FV.selectbox("Type d'entraînement",
                                                          ["Force absolue", "Force accélération", "Force-vitesse",
                                                           "Vitesse-force",
                                                           "Vitesse absolue"])

            graph_manager(st.session_state.graphs_of_fv, exp_courbe_FV)
            for gfv in st.session_state.graphs_of_fv:
                gfv.create_fv_graph(checkbox_show_estimated_data)
                gfv.create_fv_zone_infos()
                with col1:
                    gfv.show_graph()
                with col2:
                    gfv.show_graphs_info()

        # Fonctionnalité d'analyse de variables
        exp_analyse_detaille = st.sidebar.expander("Analyse détaillée", expanded=False)

        if exp_analyse_detaille.expanded:

            selected_user = exp_analyse_detaille.selectbox('Sélectionner un utilisateur', df['User'].unique(),
                                                           key="User_AD")
            selected_exercice = exp_analyse_detaille.selectbox('Sélectionner un exercice',
                                                               df[(df['User'] == selected_user)]['Exercise'].unique(),
                                                               key="Exo_AD")
            sorted(titres)

            excluded_titles = ['Date', 'Time', 'User', 'Exercise', 'Set order', 'Rep order']
            selected_title = exp_analyse_detaille.selectbox("Sélectionnez une variable à analyser",
                                                            [title for title in titres if
                                                             title not in excluded_titles and df[
                                                                 (df['User'] == selected_user) & (
                                                                         df['Exercise'] == selected_exercice)][
                                                                 title].notnull().any()], key="Title_AD")

            if exp_analyse_detaille.checkbox("Analyse de plusieurs séances?"):
                analyzing_multiple_date = True
                selected_start_date = exp_analyse_detaille.selectbox('Date de début',
                                                                     options=[date.strftime('%Y-%m-%d') for date in df[
                                                                         (df['User'] == selected_user) & (df[
                                                                                                              'Exercise'] == selected_exercice)][
                                                                         'Date'].unique()], key="Start_date_AD")
                min_end_date = df[(df['User'] == selected_user) & (df['Exercise'] == selected_exercice) & (
                        df['Date'] >= pd.to_datetime(selected_start_date))]['Date'].min()
                selected_end_date = exp_analyse_detaille.selectbox('Date de fin',
                                                                   options=[date.strftime('%Y-%m-%d') for date in df[
                                                                       (df['User'] == selected_user) & (df[
                                                                                                            'Exercise'] == selected_exercice) & (
                                                                               df['Date'] >= min_end_date)][
                                                                       'Date'].unique()], key="End_date_AD")
                start_date = datetime.combine(pd.to_datetime(selected_start_date), datetime.min.time()).date()
                end_date = datetime.combine(pd.to_datetime(selected_end_date), datetime.max.time()).date()
            else:
                analyzing_multiple_date = False
                selected_date_single = exp_analyse_detaille.selectbox('Date de début',
                                                                      options=[date.strftime('%Y-%m-%d') for date in df[
                                                                          (df['User'] == selected_user) & (df[
                                                                                                               'Exercise'] == selected_exercice)][
                                                                          'Date'].unique()], key="Single_date_AD")
                end_date = start_date = datetime.combine(pd.to_datetime(selected_date_single),
                                                         datetime.max.time()).date()

            if exp_analyse_detaille.button("Ajouter un graphique"):
                fig, tabl = analyse_data(selected_user, selected_exercice, start_date, end_date)
                st.session_state.graphs_of_variables.append([fig, tabl])

            if exp_analyse_detaille.button("Ajouter une variable au dernier graphique"):
                add_figure_and_update_table()

            # Dirige les boutons permettant d'effacer les graphiques illustrés
            graph_manager(st.session_state.graphs_of_variables, exp_analyse_detaille)

            for g in st.session_state.graphs_of_variables:
                with col1:
                    st.plotly_chart(g[0])
                with col2:
                    if len(g) > 1:
                        st.plotly_chart(g[1])


    except pd.errors.EmptyDataError:
        st.error("Le fichier est vide ou ne contient pas de colonnes.")

else:
    logo = Image.open("C:/Users/User/Desktop/ProfilePuissance Python/Images Streamlit/Logo_Tennis_Canada.png")
    emplacement_logo.image(logo, width=100)
    main_title.header("Analyse du profile de puissance des athlètes de tennis")
    st.sidebar.warning("Veuillez télécharger un fichier TSV valide.")
