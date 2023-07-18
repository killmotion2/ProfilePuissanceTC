import string

import plotly.tools
import re
import streamlit as st
import pandas as pd
from plotly import graph_objects as go
from plotly.subplots import make_subplots
import plotly.colors as pc
from datetime import datetime
import numpy as np
from scipy import stats

from scipy.optimize import curve_fit
from PIL import Image


##TODO: Comment puis-je rendre l'analyse d'une séance plus intéressante?, est-ce que c'est possible d'avoir statistqiue pour comparer les données (courbe une et l'autre)??, régler problème de template pour courbe FV simple
# Pour partir l'app: streamlit run "C:\Users\User\Desktop\ProfilePuissance Python\StreamlitV2\main.py"

def create_acronym(name):
    words = name.split()  # Divise le nom en mots
    acronym = words[0][0].upper()  # Première lettre du prénom en majuscule
    for word in words[1:]:
        acronym += f". {word[0].upper()}"  # Ajout de la première lettre des autres mots en majuscule
    return acronym


def find_unit_of_title(slt_title):
    # Recherche le motif '[x]' dans le titre
    match = re.search(r'\[(.*?)]', slt_title)
    if match:
        # Si un motif est trouvé, renvoie le texte entre crochets (l'unité)
        unit = match.group(1)
        return unit
    else:
        # Si aucun motif n'est trouvé, renvoie une valeur par défaut ou None
        return None
def find_Cohen_interpretation(effect_size, ET):
    if effect_size < 0.2 * ET:
        effect_size_title = "Sans intérêt"
    elif 0.2 * ET <= effect_size < 0.5 * ET:
        effect_size_title = "Faible"
    elif 0.5 * ET <= effect_size < 0.8 * ET:
        effect_size_title = "Modéré"
    elif effect_size > 0.8 * ET:
        effect_size_title = "Grand"
    else:
        effect_size_title = ""
    return effect_size_title

def find_data_from_dates(flt_data, slt_title, user_name):
    unit_of_title = find_unit_of_title(slt_title)
    #Valeurs de la moyenne de la variable choisi, regroupé par chaque date et chaque série
    grouped_data = flt_data.groupby(['Set order', 'Date'])[slt_title].mean().reset_index()
    # Valeurs de chaque charge maximale utilisée pour chaque date et chaque série
    load_date_set = flt_data.groupby(['Date', 'Set order'])['Load [lb]'].max().reset_index()

    load_dict = load_date_set.set_index(['Date', 'Set order'])['Load [lb]'].to_dict()
    grouped_data['Load [lb]'] = grouped_data.apply(lambda row: load_dict.get((row['Date'], row['Set order'])), axis=1)
    grouped_data = grouped_data.sort_values(by='Load [lb]', ascending=True)

    #Valeurs des améliorations de chaque charge (Différence entre les deux meilleurs performance de cette charge)
    grouped_by_charge = grouped_data.groupby('Load [lb]')
    improvement_values = []
    perf_differences = []
    effect_sizes = []
    SWC = []
    effect_size_titles = []


    for _, group in grouped_by_charge:
        if len(group) > 1:
            # Utilisez nlargest(2) pour obtenir les deux meilleures performances pour ce groupe
            top_2_performances = group.nlargest(2, slt_title)[slt_title].values
            # Calculez la différence entre les deux meilleures performances
            perf_diff = top_2_performances[0] - top_2_performances[1]
            improvement = round(perf_diff/top_2_performances[1] *100, 2)

            s = group[slt_title].std()
            SWC.append(round(s*0.2,2))
            effect_size = perf_diff
            effect_sizes.append(round(effect_size,2))
            effect_size_titles.append(find_Cohen_interpretation(effect_size,s))
        else:
            improvement = 'Pas assez de valeurs'
            perf_diff = 0
            effect_sizes.append("-")
            effect_size_titles.append("-")
            SWC.append("-")
        improvement_values.append(improvement)
        perf_differences.append(round(perf_diff,2))

    perf_changes_combined = [f"{pourcentage} ({original_unit})" for pourcentage, original_unit in
                            zip(improvement_values, perf_differences)]

    hovertemplate = 'Date: %{customdata|%Y-%m-%d}<br>#séries: %{text}<br>Charge: %{x}<br>' + slt_title + ': %{y}<extra></extra>'

    fig_combined = go.Figure()
    fig_combined.add_trace(go.Scatter(
        x=grouped_data['Load [lb]'],
        y=grouped_data[slt_title],
        mode='markers',
        hovertemplate=hovertemplate,
        text=grouped_data['Set order'],
        customdata=grouped_data['Date'],
    ))
    fig_combined.update_layout(
        title=f"{user_name}, {selected_exercice}, {slt_title}",
        xaxis_title='Charge [lb]',
        yaxis_title=slt_title,
    )


    improvement_table = go.Figure(data=go.Table(
        header=dict(values=['Charge (lb)', f"Changement dans la performance (%,({unit_of_title}))", f"Le plut petit changement significatif (SWC) ({unit_of_title})", 'Taille de Cohen']),
        cells=dict(values=[
            grouped_data['Load [lb]'].unique(),
            perf_changes_combined,
            SWC,
            effect_size_titles
        ])
    ))

    improvement_table.update_layout(margin=dict(t=0, b=0),width=300)

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
        self.trend_x = None
        self.trend_y = None
        self.is_type_of_exercice = None
        self.selectbox = "Force absolue"
        self.string_vitesse_maxmin = None
        self.string_repetition_maxmin = None
        self.string_load_maxmin = None
        self.area_chart = None
        self.is_comparaison_figure = None

    def show_graph(self):
        for trace in self.figure.data:
            trace.visible = True
        st.plotly_chart(self.figure)

    def show_graphs_info(self):
        if self.string_load_maxmin or self.string_vitesse_maxmin or self.string_repetition_maxmin != None:
            st.write(self.string_load_maxmin)
            st.write(self.string_vitesse_maxmin)
            st.write(self.string_repetition_maxmin)

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
        force_moyenne_y = [self.F0] + [self.flt_data[self.flt_data['Set order'] == set]["Load [lb]"].max() for set in
                                       each_set]

        exercice_UB_pull = ["Pull", "Row", "Tirade"]
        exercice_UB_push = ["Bench", "Press"]
        exercice_LB_pull = ["Deadlift"]
        exercice_LB_push = ["squat", "lunge", "fente"]

        is_UB_exercice_pull = any(exo in self.selected_exercice.lower() for exo in exercice_UB_pull)
        is_UB_exercice_push = any(exo in self.selected_exercice.lower() for exo in exercice_UB_push)
        is_LB_exercice_pull = any(exo in self.selected_exercice.lower() for exo in exercice_LB_pull)
        is_LB_exercice_push = any(exo in self.selected_exercice.lower() for exo in exercice_LB_push)
        self.is_type_of_exercice = [is_UB_exercice_pull, is_UB_exercice_push, is_LB_exercice_pull, is_LB_exercice_push]

        vit_moyenne_x = [0.0] + [self.flt_data[self.flt_data['Set order'] == set]["Avg. velocity [m/s]"].mean() for set
                                 in
                                 each_set]

        figFV = go.Figure(
            data=go.Scatter(x=vit_moyenne_x, y=force_moyenne_y, mode='markers',
                            name=f"Val.exist., {create_acronym(self.selected_user)}, {self.selected_exercice}"))

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
                go.Scatter(x=nouvelles_vitesses, y=nouvelles_charges, mode='markers',
                           name=f"Val.estim., {create_acronym(self.selected_user)}, {self.selected_exercice}"))

        popt = curve_fit(self.equation, vit_moyenne_x, force_moyenne_y)
        self.popt = popt[0]
        a_opt, b_opt, c_opt = popt[0]

        self.trend_x = np.linspace(min(vit_moyenne_x), 2, 100)
        self.trend_y = self.equation(self.trend_x, *popt[0])

        positive_indices = []
        for i, y in enumerate(self.trend_y):
            if y >= 0:
                positive_indices.append(i)
        self.trend_x = [self.trend_x[i] for i in positive_indices]
        self.trend_y = [self.trend_y[i] for i in positive_indices]

        figFV.add_trace(go.Scatter(x=self.trend_x, y=self.trend_y, mode='lines',
                                   name=f"Courbe FV ({self.date_debut.date()}/{self.date_fin.date()})"))


        vit_moyenne_x = np.array(vit_moyenne_x)
        force_moyenne_y = np.array(force_moyenne_y)

        equation_text = f"Eq : {a_opt:.2f}x^2 + {b_opt:.2f}x + {c_opt:.2f}"
        figFV.add_annotation(x=0.85, y=0.9, xref='paper', yref='paper', text=equation_text, showarrow=False)

        residuals = force_moyenne_y - self.equation(vit_moyenne_x, a_opt, b_opt, c_opt)
        ss_residual = np.sum(residuals ** 2)
        ss_total = np.sum((force_moyenne_y - np.mean(force_moyenne_y)) ** 2)
        r_squared = 1 - (ss_residual / ss_total)

        r_squared_text = f"R2 : {r_squared:.2f}"
        figFV.add_annotation(x=0.85, y=0.85, xref='paper', yref='paper', text=r_squared_text, showarrow=False)
        figFV.update_traces(hovertemplate='Vitesse: %{x:.2f} m/s <br> Charges: %{y:.2f} lbs')

        figFV.update_layout(
            xaxis_title='Vitesse moyenne (m/s)',
            yaxis_title='Charge (lb)',
            title='Relation Force-Vitesse',
            showlegend=True,
            xaxis=dict(gridcolor='lightgray'),
            yaxis=dict(gridcolor='lightgray', rangemode='nonnegative'),
            hovermode="x unified"
        )
        self.figure = figFV

    def create_fv_zone_infos(self, fv_selectbox_choice):
        if not self.is_comparaison_figure:
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
            self.string_repetition_maxmin = f"Répétitions avant l'échec : {reps_max} - {reps_min}"
        else:
            self.string_vitesse_maxmin = self.string_repetition_maxmin = self.string_load_maxmin = None

    def add_curve(self, graphs2):
        for trace in graphs2.figure.data:
            new_trace = go.Scatter(
                x=trace.x,
                y=trace.y,
                mode=trace.mode,
                name=trace.name,
                hovertemplate=f'Vitesse: %{{x:.2f}} m/s <br> Charges: %{{y:.2f}} lbs'
            )
            self.figure.add_trace(new_trace)

        if len(self.figure.data) >= 2:
            self.figure.layout.annotations = []
            x_values = self.trend_x
            y1_values = self.trend_y
            y2_values = graphs2.trend_y

            area_trace = go.Scatter(
                x=x_values,
                y=[abs(y1 - y2) for y1, y2 in zip(y1_values, y2_values)],
                mode='lines',
                fill='tozeroy',
                name='Différence',
                hovertemplate= 'Différence: %{y:.2f} lbs <br> Vitesse: %{x:.2f} m/s'
            )

            self.area_chart = area_trace

            # Créer un subplot avec deux lignes et une colonne
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)

            # Ajouter les courbes dans le premier graphique
            for trace in self.figure.data:
                fig.add_trace(trace, row=1, col=1)

            # Ajouter l'area_chart dans le deuxième graphique
            fig.add_trace(area_trace, row=2, col=1)

            # Définir les titres et les étiquettes des axes pour chaque graphique
            fig.update_layout(
                title='Relation Force-vitesse',
                xaxis=dict(title='Vitesse (m/s)', title_standoff=10,side='bottom'),
                yaxis=dict(title='Charges (lbs)', domain=[0.16, 1.0], rangemode='nonnegative'),
                yaxis2=dict(title='Diff. de perf. (lbs)', domain=[0.0, 0.15], rangemode='nonnegative')
            )
            self.figure = fig

        graphs2.string_vitesse_maxmin = graphs2.string_load_maxmin = graphs2.string_repetition_maxmin = None
        self.string_vitesse_maxmin = self.string_load_maxmin = self.string_repetition_maxmin = None


def analyse_data(user, exercise, start_date, end_date):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    filtered_data = df[(df['User'] == user) & (df['Date'].between(start_date, end_date)) & (df['Exercise'] == exercise)]

    if analyzing_multiple_date:
        fig, tabl = find_data_from_dates(filtered_data, selected_title,user)
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


def graph_manager(graphs_list, expander):
    for i, fig in enumerate(graphs_list):
        with expander:
            graph_title = f"Graphique {i + 1}"
            if len(graphs_list) < 1:
                bt = expander.empty()
            else:
                bt = expander.button(f"{graph_title}     X")
            if bt.__bool__():
                graphs_list.pop(i)
                st.experimental_rerun()


def show_all_graphs_in_loop(selectbox_choice):
    for gfv in st.session_state.graphs_of_fv:
        if selectbox_choice != st.session_state.previous_selectbox_choice:
            gfv.create_fv_zone_infos(selectbox_choice)
            st.session_state.previous_selectbox_choice = selectbox_choice
        with col1:
            gfv.show_graph()
        with col2:
            gfv.show_graphs_info()


##VARIABLES IMPORTANTES
if 'lecture_courbe_FV' not in st.session_state:
    st.session_state.lecture_courbe_FV = False
if 'graphs_of_variables' not in st.session_state:
    st.session_state.graphs_of_variables = []
if 'graphs_of_fv' not in st.session_state:
    st.session_state.graphs_of_fv = []
if 'previous_selectbox_choice' not in st.session_state:
    st.session_state.previous_selectbox_choice = ""

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
        col1, col2 = st.columns([2, 1])

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

            if exp_courbe_FV.button("Comparer les courbes force-vitesse"):
                lgr_graph_Fv = len(st.session_state.graphs_of_fv)
                st.session_state.graphs_of_fv[lgr_graph_Fv - 2].add_curve(
                    st.session_state.graphs_of_fv[lgr_graph_Fv - 1])
                st.session_state.graphs_of_fv.pop()

            if len(st.session_state.graphs_of_fv) >= 1:
                checkbox_show_estimated_data = exp_courbe_FV.checkbox(
                    "Ajouter les valeurs estimées dans la prédiction de la courbe",
                    on_change=lambda: st.session_state.graphs_of_fv[
                        len(st.session_state.graphs_of_fv) - 1].create_fv_graph(checkbox_show_estimated_data)
                )

            fv_selectbox_choice = exp_courbe_FV.selectbox("Type d'entraînement",
                                                          ["Force absolue", "Force accélération", "Force-vitesse",
                                                           "Vitesse-force",
                                                           "Vitesse absolue"])
            graph_manager(st.session_state.graphs_of_fv, exp_courbe_FV)
            show_all_graphs_in_loop(fv_selectbox_choice)

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