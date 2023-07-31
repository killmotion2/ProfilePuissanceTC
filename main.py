import re

import requests
import streamlit as st
import pandas as pd
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import numpy as np
from scipy.optimize import curve_fit
import itertools

##TODO: Réorganiser la fonction analyse détaillé de plusieurs séance afin d'avoir une classe
# Pour partir l'app: streamlit run "C:\Users\User\Desktop\ProfilePuissance Python\StreamlitV2\main.py"

#Variables à initialiser
if 'lecture_courbe_FV' not in st.session_state:
    st.session_state.lecture_courbe_FV = False
if 'graphs_of_variables' not in st.session_state:
    st.session_state.graphs_of_variables = []
if 'graphs_of_fv' not in st.session_state:
    st.session_state.graphs_of_fv = []
if 'dataframe_of_session' not in st.session_state:
    st.session_state.dataframe_of_session = []
if 'previous_selectbox_choice' not in st.session_state:
    st.session_state.previous_selectbox_choice = ""
checkbox_show_estimated_data = False

##ILLUSTRATION INITIALE
st.set_page_config(page_title="Analyse profile athlètes de Tennis Canada", page_icon=":bar_chart:", layout="wide")
emplacement_logo = st.empty()
main_title = st.empty()

#Texte pour les capsules informatives de chaque fonction de l'application
txt_info_FV = """
Cet onglet a pour but d’afficher et/ou de comparer des relations force-vitesse d’un athlète\n
-Comment l’onglet fonctionne ? :\n
-Pour ajouter une relation force-vitesse :\n
1. Choisir l’athlète, l’exercice ainsi que l’intervalle de dates à analyser
2. Peser le bouton « Ajouter relation force-vitesse »\n
NB : Il est possible d’ajouter des valeurs estimées à la relation force-vitesse, dans le cas où l’intervenant juge qu’il n’existe pas assez de valeurs existantes pour avoir une bonne validité dans la relation.\n
-Pour comparer deux relations :\n
1. Répéter les étapes « Pour ajouter une relation force-vitesse » pour les deux relations force-vitesse
2. Peser le bouton « Comparer relation force-vitesse »\n
-Aide mémoire en lien avec l'analyse des données:
1. L'équation de la relation force-vitesse peut être utile pour comparer un athlète à lui-même. Dans l'équation "Ax + B", plus le B est grand, plus le 1RM de l'athlète est élevé, tandis que plus le A est grand, moins l'atlète à de la faciliter à bouger des charges légère à de haute vitesse. À titre indicatif, au tennis, il est préférable d'avoir un B élevé, et un A petit.
2. Plus le R2 (coefficient de détermination) est grand, plus les points existants se collent à la droite. Grossièrement, plus le R2 se rapproche de 1, plus les points sont proches l'un de l'autre.
3. Lorsqu'on compare deux relations, la partie "Diff. (lbs)" du graphique correspond à la différence EN ABSOLUE entre les deux relations choisies.
"""
txt_info_DA = """Cet onglet a pour but d'afficher et d'analyser toutes les performances de l'athlète en fonction de la charge manipulée.\n
- Aide mémoire en lien avec l'analyse de données:\n
1. Plus le graphique a de valeurs, plus les données présenter dans le tableau (SWC et taille de Cohen) sont valide.
2. Le plus petit changement significatif (SWC) représente le gain minimal à acquérir afin de considérer une amélioration significative dans la performance.
3. La taille de Cohen est une façon de catégoriser l'effet de l'intervention. Les catégories sont les suivantes : Faible = 0.2* Écart-type, Modéré = 0.5 * ÉT, Grand = 0.8 * ÉT)."""

txt_info_session = """Cet onglet a pour but d'afficher toutes les données recueillies par l'accéléromètre Enode à 
l'aide d'un tableau."""

#Fonctions générales
def create_acronym(name):
    words = name.split()  # Divise le nom en mots
    acronym = words[0][0].upper()  # Première lettre du prénom en majuscule
    for word in words[1:]:
        acronym += f". {word[0].upper()}"  # Ajout de la première lettre des autres mots en majuscule
    return acronym

def find_unit_of_title(slt_title):
    match = re.search(r'\[(.*?)]', slt_title)
    if match:
        unit = match.group(1)
        return unit
    else:
        return None

def lbs_to_kg(weight_lbs):
    return weight_lbs * 0.453592

#Fonctions d'affichages
def graph_manager(graphs_list, expander):
    for i, fig in enumerate(graphs_list):
        with expander:
            graph_title = f"Graphique {i + 1}"
            if len(graphs_list) < 1:
                bt = expander.empty()
            else:
                bt = expander.button(f"{graph_title}     X", key={expander})
            if bt.__bool__():
                graphs_list.pop(i)
                st.experimental_rerun()


def show_all_FV_graphs_in_loop(selectbox_choice):
    for gfv in st.session_state.graphs_of_fv:
        gfv.create_fv_zone_infos(selectbox_choice)
        gfv.show_graph()

#Classes spécifiques à chaque fonctionnalité de l'application (Courbe FV, Analyse détaillée, Analyse d'une séance)
class CourbeForceVitesse:
    def __init__(self, selected_user, selected_exercice, start_date, end_date):
        self.selected_user = selected_user
        self.selected_exercice = selected_exercice
        self.date_debut = pd.to_datetime(start_date)
        self.date_fin = pd.to_datetime(end_date)
        self.flt_data = df[(df['User'] == selected_user) & (df['Exercise'] == selected_exercice) & (
            df['Date'].between(self.date_debut, self.date_fin))]
        self.figure = None
        self.table_data = None
        self.popt = None
        self.R2 = None
        self.trend_x = None
        self.trend_y = None
        self.is_type_of_exercice = None
        self.selectbox = "Force absolue"
        self.speed_minMax = None
        self.repetition_maxMin = None
        self.load_maxMin = None
        self.area_chart = None
        self.is_comparaison_figure = False
        self.info_table = None

    def show_graph(self):
        col1, col2 = st.columns([2, 1])
        with col1:
            st.plotly_chart(self.figure, use_container_width=True)
        with col2:
            st.plotly_chart(self.table_data, theme="streamlit", use_container_width=True)

    def equation(self, x, a, b):
        return a * x + b

    def estimer_charge_squat(self, vitesse_moyenne_x, f0):
        return (-12.87 * vitesse_moyenne_x ** 2 - 46.31 * vitesse_moyenne_x + 116.3) / 100 * f0

    def estimer_charge_benchpress(self, vitesse_moyenne, f0):
        return (11.4196 * vitesse_moyenne ** 2 - 81.904 * vitesse_moyenne + 114.03) / 100 * f0

    def estimer_charge_tirade(self, vitesse_moyenne, f0):
        return (18.5797 * vitesse_moyenne ** 2 - 104.182 * vitesse_moyenne + 147.94) / 100 * f0

    def create_fv_graph(self, checkbox_show_estimated_data):
        self.F0 = self.flt_data['Estimated 1RM [lb]'].unique().max()
        each_set = self.flt_data['Set order'].unique()

        force_moyenne_y = [
            self.flt_data[self.flt_data['Set order'] == set]["Load [lb]"].unique().tolist()
            for set in each_set
        ]

        force_moyenne_y = list(itertools.chain(*force_moyenne_y))
        force_moyenne_y.insert(0, self.F0)

        grouped_data = self.flt_data.groupby(['Load [lb]', 'Date'])['Avg. velocity [m/s]'].mean().reset_index()

        vit_moyenne_x = [0.0] + [
            grouped_data[grouped_data['Load [lb]'] == load_val]['Avg. velocity [m/s]'].mean().tolist()
            for load_val in force_moyenne_y[1:]]

        exercice_UB_pull = ["Pull", "Row", "Tirade"]
        exercice_UB_push = ["Bench", "Press"]
        exercice_LB_pull = ["Deadlift"]
        exercice_LB_push = ["squat", "lunge", "fente"]

        is_UB_exercice_pull = any(exo in self.selected_exercice.lower() for exo in exercice_UB_pull)
        is_UB_exercice_push = any(exo in self.selected_exercice.lower() for exo in exercice_UB_push)
        is_LB_exercice_pull = any(exo in self.selected_exercice.lower() for exo in exercice_LB_pull)
        is_LB_exercice_push = any(exo in self.selected_exercice.lower() for exo in exercice_LB_push)
        self.is_type_of_exercice = [is_UB_exercice_pull, is_UB_exercice_push, is_LB_exercice_pull, is_LB_exercice_push]

        hovertemplate = 'Vitesse: %{x:.2f} m/s <br> Charges: %{y:.2f} lbs (%{customdata:.1f} kg)'
        figFV = go.Figure(
            data=go.Scatter(x=vit_moyenne_x, y=force_moyenne_y, mode='markers',
                            name=f"Val.exist., {create_acronym(self.selected_user)} \n, {self.selected_exercice}",
                            hovertemplate=hovertemplate,
                            customdata=lbs_to_kg(np.array(force_moyenne_y)))
        )

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
                           name=f"Val.estim., {create_acronym(self.selected_user)}, \n {self.selected_exercice}",
                           hovertemplate=hovertemplate,
                           customdata=lbs_to_kg(np.array(force_moyenne_y)))
            )

        popt = curve_fit(self.equation, vit_moyenne_x, force_moyenne_y)
        self.popt = popt[0]
        a_opt, b_opt = popt[0]

        self.trend_x = np.linspace(min(vit_moyenne_x), 2.5, 100)
        self.trend_y = self.equation(self.trend_x, *popt[0])

        positive_indices = []
        for i, y in enumerate(self.trend_y):
            if y >= 0:
                positive_indices.append(i)
        self.trend_x = [self.trend_x[i] for i in positive_indices]
        self.trend_y = [self.trend_y[i] for i in positive_indices]


        figFV.add_trace(go.Scatter(x=self.trend_x, y=self.trend_y, mode='lines',
                                   name=f"Courbe FV ({self.date_debut.date()}/{self.date_fin.date()})",
                                   hovertemplate=hovertemplate,
                                   customdata=lbs_to_kg(np.array(self.trend_y)))
                        )

        vit_moyenne_x = np.array(vit_moyenne_x)
        force_moyenne_y = np.array(force_moyenne_y)

        residuals = force_moyenne_y - self.equation(vit_moyenne_x, a_opt, b_opt)
        ss_residual = np.sum(residuals ** 2)
        ss_total = np.sum((force_moyenne_y - np.mean(force_moyenne_y)) ** 2)
        self.R2 = 1 - (ss_residual / ss_total)

        figFV.update_layout(
            xaxis_title='Vitesse moyenne (m/s)',
            yaxis_title='Charge (lb)',
            title='Relation Force-Vitesse',
            showlegend=True,
            xaxis=dict(gridcolor='lightgray'),
            yaxis=dict(gridcolor='lightgray', rangemode='nonnegative'),
            hovermode="x unified",
            margin=dict(l=50, r=50, b=50, t=50),
            legend=dict(orientation="h", x=0, y=-0.15)
        )
        self.figure = figFV

    def create_fv_zone_infos(self, fv_selectbox_choice):
        if not self.is_comparaison_figure:
            a_opt, b_opt = self.popt

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

            estimated_load_max = int(round(self.equation(v_min, a_opt, b_opt)))
            estimated_load_min = int(round(self.equation(v_max, a_opt, b_opt)))

            estimated_load_min = max(0, estimated_load_min)
            estimated_load_max = max(0, estimated_load_max)

            reps_min = int(round((1.0278 * self.F0 - estimated_load_min) / (0.0278 * self.F0)))
            if reps_min < 0: reps_min = 0
            reps_max = int(round((1.0278 * self.F0 - estimated_load_max) / (0.0278 * self.F0)))
            if reps_max < 0: reps_max = 0

            self.speed_minMax = [v_min, v_max]
            self.load_maxMin = [estimated_load_max, estimated_load_min]
            self.repetition_maxMin = [reps_max, reps_min]

            info_table = [
                ["Équation", "R2", "Intervalles de vitesses", "Intervalles de charges", "Répétitions avant l'échec"],
                [f"{self.popt[0]:.2f}x + {self.popt[1]:.2f}", f"{self.R2:.2f}",
                 f"{self.speed_minMax[0]} - {self.speed_minMax[1]} m/s",
                 f"{self.load_maxMin[0]}({lbs_to_kg(self.load_maxMin[0]):.1f}) - {self.load_maxMin[1]}({lbs_to_kg(self.load_maxMin[1]):.1f}) lbs (kg)",
                 f"{self.repetition_maxMin[0]} - {self.repetition_maxMin[1]}"]]

            self.table_data = go.Figure(data=go.Table(
                header=dict(
                    values=["",
                            f"{create_acronym(self.selected_user)} ({self.date_debut.date()}/{self.date_fin.date()})"]),
                cells=dict(values=info_table)))
            self.table_data.update_layout(autosize=True)

    def add_curve(self, graphs2):
        if not self.is_comparaison_figure:
            self.is_comparaison_figure = True
            for trace in graphs2.figure.data:
                new_trace = go.Scatter(
                    x=trace.x,
                    y=trace.y,
                    mode=trace.mode,
                    name=trace.name,
                    hovertemplate=f'Vitesse: %{{x:.2f}} m/s <br> Charges: %{{y:.2f}} lbs'
                )
                self.figure.add_trace(new_trace)

            if self.is_comparaison_figure:
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
                    hovertemplate='Différence: %{y:.2f} lbs <br> Vitesse: %{x:.2f} m/s'

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
                    yaxis=dict(title='Charges (lbs)', domain=[0.21, 1.0], rangemode='nonnegative'),
                    yaxis2=dict(title='Diff.(lbs)', domain=[0.05, 0.20], rangemode='nonnegative'),
                    xaxis=dict(title='Vitesse (m/s)'),
                    legend = dict(orientation="h", x=0, y=-0.15)
                )
                info_table = [
                    ["Équation", "R2"],
                    [f"{self.popt[0]:.2f}x + {self.popt[1]:.2f}", f"{self.R2:.2f}"],
                    [f"{graphs2.popt[0]:.2f}x + {graphs2.popt[1]:.2f}", f"{graphs2.R2:.2f}"]
                ]
                # Create the table figure
                table = go.Figure(data=go.Table(
                    header=dict(
                        values=["",
                                f"{create_acronym(self.selected_user)} ({self.date_debut.date()}/{self.date_fin.date()})",
                                f"{create_acronym(graphs2.selected_user)} ({graphs2.date_debut.date()}/{graphs2.date_fin.date()})"]
                    ),
                    cells=dict(values=info_table)))
                # Update the table layout
                self.table_data = table
                self.figure = fig
            else:
                st.warning('Vous ne pouvez pas comparer plus que 2 courbes dans le même graphique')

class Multiple_Dates_Analysis:
    def __init__(self, flt_data, user, exercise, title):
        self.flt_data = flt_data
        self.user = user
        self.exercise = exercise
        self.slt_title = title
        self.fig = None
        self.tabl = None
    def show_graph_DA(self):
        col1, col2 = st.columns([2, 1])
        with col1:
            st.plotly_chart(self.fig, use_container_width=True)
        with col2:
            st.plotly_chart(self.tabl, use_container_width=True)
    def find_Cohen_interpretation(self, effect_size, ET):
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

    def find_data_from_dates(self):
        unit_of_title = find_unit_of_title(self.slt_title)
        # Valeurs de la moyenne de la variable choisi, regroupé par chaque date et chaque série
        grouped_data = self.flt_data.groupby(['Set order', 'Date'])[self.slt_title].mean().reset_index()
        # Valeurs de chaque charge maximale utilisée pour chaque date et chaque série
        load_date_set = self.flt_data.groupby(['Date', 'Set order'])['Load [lb]'].max().reset_index()

        load_dict = load_date_set.set_index(['Date', 'Set order'])['Load [lb]'].to_dict()
        grouped_data['Load [lb]'] = grouped_data.apply(lambda row: load_dict.get((row['Date'], row['Set order'])),
                                                       axis=1)
        grouped_data = grouped_data.sort_values(by='Load [lb]', ascending=True)

        unique_dates = grouped_data['Date'].unique()

        # Trois teintes de couleur pour le bleu, vert et rouge
        colors = ['#0072BD', '#4DBEEE', '#B0E0E6', '#A2142F', '#ED553B', '#FFAC9F', '#007F3F', '#2ECC40', '#9ACD32']
        date_to_color = {date: colors[i % len(colors)] for i, date in enumerate(unique_dates)}

        grouped_data['Color'] = grouped_data['Date'].map(date_to_color)

        # Valeurs des améliorations de chaque charge (Différence entre les deux meilleurs performance de cette charge)
        grouped_by_charge = grouped_data.groupby('Load [lb]')
        improvement_values = []
        perf_differences = []
        effect_sizes = []
        SWC = []
        effect_size_titles = []

        # trouver la différence entre les meilleurs performance et trouver l'effet de Cohen
        for _, group in grouped_by_charge:
            if len(group) > 1:

                top_2_performances = group.nlargest(2, self.slt_title)[self.slt_title].values
                perf_diff = top_2_performances[0] - top_2_performances[1]
                improvement = round(perf_diff / top_2_performances[1] * 100, 2)

                s = group[self.slt_title].std()
                SWC.append(round(s * 0.2, 2))
                effect_size = perf_diff
                effect_sizes.append(round(effect_size, 2))
                effect_size_titles.append(self.find_Cohen_interpretation(effect_size, s))
            else:
                improvement = 'Pas assez de valeurs'
                perf_diff = 0
                effect_sizes.append("-")
                effect_size_titles.append("-")
                SWC.append("-")
            improvement_values.append(improvement)
            perf_differences.append(round(perf_diff, 2))

        perf_changes_combined = [f"{pourcentage} ({original_unit})" for pourcentage, original_unit in
                                 zip(improvement_values, perf_differences)]

        hovertemplate = 'Date: %{customdata|%Y-%m-%d}<br>#séries: %{text}<br>Charge: %{x}<br>' + self.slt_title + ': %{y}<extra></extra>'

        fig_combined = go.Figure()
        fig_combined.add_trace(go.Scatter(
            x=grouped_data['Load [lb]'],
            y=grouped_data[self.slt_title],
            mode='markers',
            marker=dict(
                color=grouped_data['Color'],  # Set the color based on the 'Color' column
                size=8,  # Adjust the size of the markers as needed
                line=dict(width=1, color='black'),  # Set marker outline properties
            ),
            hovertemplate=hovertemplate,
            text=grouped_data['Set order'],
            customdata=grouped_data['Date'],
        ))
        fig_combined.update_layout(
            title=f"{self.user}, {selected_exercice_DA}, {self.slt_title}",
            xaxis_title='Charge [lb]',
            yaxis_title=self.slt_title,
        )

        improvement_table = go.Figure(data=go.Table(
            header=dict(values=['Charge (lb)', f"Changement dans la performance (%,({unit_of_title}))",
                                f"Le plut petit changement significatif (SWC) ({unit_of_title})", 'Taille de Cohen']),
            cells=dict(values=[
                grouped_data['Load [lb]'].unique(),
                perf_changes_combined,
                SWC,
                effect_size_titles
            ])
        ))

        improvement_table.update_layout(margin=dict(t=0, b=0))
        self.fig = fig_combined
        self.tabl = improvement_table


class Session_dataframe:
    def __init__(self, flt_data, user_name, exercice, date):
        self.data = self.reorganise_data(flt_data)
        self.name = user_name
        self.exercice = exercice
        self.date = date

    def reorganise_data(self, data_to_organise):
        first_columns = ['Load [lb]', 'Set order', 'Rep order']
        other_columns = sorted(list(data_to_organise.columns.drop(first_columns)))
        return data_to_organise.reindex(columns=first_columns + other_columns).reset_index()

    def show_dataframe(self):
        st.dataframe(self.data, use_container_width=True)


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

        if selected_exercice_DA in old_title and selected_exercice_DA in new_title:
            new_title = new_title.replace(f"{selected_exercice_DA},", "")
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


##MAIN LOOP

#Importation des données
upload_file = st.sidebar.file_uploader("Choisir un fichier TSV", type='tsv')

if upload_file is not None:
    emplacement_logo.empty()
    main_title.empty()
    try:
        df = pd.read_table(upload_file)
        df['Date'] = pd.to_datetime(df['Date'])

        titres = df.columns.tolist()


        # Fonctionnalité #1. Analyse de la courbe force-vitesse
        exp_courbe_FV = st.sidebar.expander("Analyse courbe force-vitesse")
        if exp_courbe_FV.expanded:
            if exp_courbe_FV.checkbox("Info", key="Info_FV"):
                st.info(txt_info_FV)


            selected_user_fv = exp_courbe_FV.selectbox('Sélectionner un utilisateur', sorted(df['User'].unique()),
                                                       key="User_FV")
            selected_exercice_fv = exp_courbe_FV.selectbox('Sélectionner un exercice',
                                                           sorted(df[(df['User'] == selected_user_fv)]['Exercise'].unique()),
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
                exp_courbe_FV.button("Ajouter relation force-vitesse", disabled=True)
            elif exp_courbe_FV.button("Ajouter relation force-vitesse", disabled=False):
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
            show_all_FV_graphs_in_loop(fv_selectbox_choice)

        # Fonctionnalité #2. Analyse de variables
        exp_detail_analysis = st.sidebar.expander("Analyse détaillée dans le temps", expanded=False)

        if exp_detail_analysis.expanded:

            selected_user_DA = exp_detail_analysis.selectbox('Sélectionner un utilisateur', sorted(df['User'].unique()),
                                                             key="User_AD")
            selected_exercice_DA = exp_detail_analysis.selectbox('Sélectionner un exercice',
                                                                 sorted(df[(df['User'] == selected_user_DA)][
                                                                     'Exercise'].unique()),
                                                                 key="Exo_AD")
            sorted(titres)
            custom_order_titles = ['Time to peak velocity [ms]', 'Peak velocity [m/s]', 'Peak power [W]',
                                   'Peak RFD [N/s]']
            sorted_remaining_titles = sorted([title for title in titres if title not in custom_order_titles])
            excluded_titles_DA = ['Date', 'Time', 'User', 'Exercise', 'Set order', 'Rep order', 'Load [lb]',
                                  'Estimated 1RM [lb]', 'Total volume [lb]', 'Maximum load [lb]']

            sorted_titles = custom_order_titles + sorted_remaining_titles
            selected_title_DA = exp_detail_analysis.selectbox(
                "Sélectionnez une variable à analyser",
                [title for title in sorted_titles if
                 title not in excluded_titles_DA and df[
                     (df['User'] == selected_user_DA) & (df['Exercise'] == selected_exercice_DA)][
                     title].notnull().any()],
                key="Title_AD"
            )
            selected_start_date = exp_detail_analysis.selectbox('Date de début',
                                                                options=[date.strftime('%Y-%m-%d') for date in df[
                                                                    (df['User'] == selected_user_DA) & (df[
                                                                                                            'Exercise'] == selected_exercice_DA)][
                                                                    'Date'].unique()], key="Start_date_AD")
            min_end_date = df[(df['User'] == selected_user_DA) & (df['Exercise'] == selected_exercice_DA) & (
                    df['Date'] >= pd.to_datetime(selected_start_date))]['Date'].min()
            selected_end_date = exp_detail_analysis.selectbox('Date de fin',
                                                              options=[date.strftime('%Y-%m-%d') for date in df[
                                                                  (df['User'] == selected_user_DA) & (df[
                                                                                                          'Exercise'] == selected_exercice_DA) & (
                                                                          df['Date'] >= min_end_date)][
                                                                  'Date'].unique()], key="End_date_AD")
            start_date = datetime.combine(pd.to_datetime(selected_start_date), datetime.min.time()).date()
            end_date = datetime.combine(pd.to_datetime(selected_end_date), datetime.max.time()).date()

            if exp_detail_analysis.button("Ajouter un graphique", key='bt_add_graph_DA'):
                filtrd_data_DA = df[(df['User'] == selected_user_DA) & (
                    df['Date'].between(pd.to_datetime(start_date), pd.to_datetime(end_date))) & (
                                                df['Exercise'] == selected_exercice_DA)]
                graph_DA = Multiple_Dates_Analysis(filtrd_data_DA, selected_user_DA, selected_exercice_DA,
                                                   selected_title_DA)
                graph_DA.find_data_from_dates()
                st.session_state.graphs_of_variables.append(graph_DA)

            # Dirige les boutons permettant d'effacer les graphiques illustrés
            graph_manager(st.session_state.graphs_of_variables, exp_detail_analysis)
            for g in st.session_state.graphs_of_variables:
                g.show_graph_DA()

        #Fonctionnalité #3. Analyse d'une session
        exp_detail_session = st.sidebar.expander("Analyse d'une séance", expanded=False)
        if exp_detail_session.expanded:
            selected_user_session = exp_detail_session.selectbox('Sélectionner un utilisateur', sorted(df['User'].unique()),
                                                                 key="User_session")
            selected_exercice_session = exp_detail_session.selectbox('Sélectionner un exercice',
                                                                     sorted(df[(df['User'] == selected_user_session)][
                                                                         'Exercise'].unique()),
                                                                     key="Exo_session")

            selected_date_single_session = exp_detail_session.selectbox('Date de début',
                                                                        options=[date.strftime('%Y-%m-%d') for date in
                                                                                 df[
                                                                                     (df[
                                                                                          'User'] == selected_user_session) & (
                                                                                                 df[
                                                                                                     'Exercise'] == selected_exercice_session)][
                                                                                     'Date'].unique()],
                                                                        key="Single_date_session")

            if exp_detail_session.button("Ajouter un graphique", key='bt_add_graph_session'):
                filtered_df = df[
                    (df['User'] == selected_user_session) & (df['Exercise'] == selected_exercice_session) & (
                            df['Date'] == selected_date_single_session)].dropna(axis=1, how='all').drop(
                    columns=['Date', 'Time', 'User', 'Total volume [lb]', 'Maximum load [lb]'])

                st.session_state.dataframe_of_session.append(
                    Session_dataframe(filtered_df, selected_user_session, selected_exercice_session,
                                      selected_date_single_session))

            # Dirige les boutons permettant d'effacer les graphiques illustrés
            graph_manager(st.session_state.dataframe_of_session, exp_detail_session)

            for g in st.session_state.dataframe_of_session:
                g.show_dataframe()

    except pd.errors.EmptyDataError:
        st.error("Le fichier est vide ou ne contient pas de colonnes.")

else:

    emplacement_logo.image(
        "https://github.com/killmotion2/ProfilePuissanceTC/blob/main/Logo_Tennis_Canada.png?raw=true", width=100)
    main_title.header("Analyse du profile de puissance des athlètes de tennis")
    st.sidebar.warning("Veuillez télécharger un fichier TSV valide.")
