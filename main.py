import re
import streamlit as st
import pandas as pd
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import numpy as np
import math
from scipy.optimize import curve_fit
import itertools
from itertools import cycle


# Pour partir l'app: streamlit run "C:\Users\User\Desktop\ProfilePuissance Python\StreamlitV2\main.py"

# Initializing variables
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

load_title = None
estimated_1RM_title = None

##ILLUSTRATION INITIALE
st.set_page_config(page_title="Analyse du profil des athlètes de Tennis Canada", page_icon=":bar_chart:",
                   layout="wide")
emplacement_logo = st.empty()
main_title = st.empty()

# Texts for info segment
txt_info_FV = """
Cet onglet a pour but d’afficher et/ou de comparer des relations force-vitesse d’un athlète\n
-Comment l’onglet fonctionne ? :\n
-Pour ajouter une relation force-vitesse :\n
1. Choisir l’athlète, l’exercice ainsi que l’intervalle de dates à analyser
2. Peser le bouton « Ajouter relation force-vitesse »\n
-Pour comparer deux relations :\n
1. Répéter les étapes « Pour ajouter une relation force-vitesse » pour les deux relations force-vitesse
2. Peser le bouton « Comparer relation force-vitesse »\n
Définitions des zones de vitesses:\n
1. Force absolue: Capacité à générer une force musculaire maximale ou à bouger des charges très lourdes.
2. Force-vitesse: Capacité à bouger des charges élevées a une vitesse modérée. EX: Au football, un joueur de la ligne défensive doit rapidement bouger une masse devant lui (son adversaire).
3. Vitesse-force: Capacité à bouger des charges modérées à haute vitesse ou capacité à exploser. EX: Au football, un joueur de la ligne défensive sort rapidement de sa position initiale avant de faire contact avec son adversaire.
4. Vitesse absolue: Capacité à surmonter l'inertie rapidement. Ex: Un receveur au football doit pouvoir rapidement surmonter l'inertie lors de ses premiers pas pour atteindre sa vitesse maximale de course. \n 
Référence: Mann, J. B. et al.(2015). Velocity-based training in football. Strength & Conditioning Journal, 37(6), 52-57. \n
-Aide mémoire en lien avec l'analyse des données:
1. L'équation de la relation force-vitesse peut être utile pour comparer un athlète à lui-même. Dans l'équation "Ax + B", plus le B est grand, plus le 1RM de l'athlète est élevé, tandis que plus le A est grand, moins l'athlète à de la faciliter à bouger des charges légères à de hautes vitesses. À titre indicatif, au tennis, il est préférable d'avoir un B élevé, et un A petit.
2. Plus il existe de points et plus il existe des performances sur tout le continuum de l'axe de vitesse, plus la validité des prédictions va être élevée.
3. Plus le R2 (coefficient de détermination) se rapproche de 1, plus les points existants se collent à la droite. 
4. L'estimation de l'erreur correspond à la moyenne de l'erreur sur toute la relation charge-vitesse. 
"""
txt_info_DA = """Cet onglet a pour but d'analyser l'évolution de l'athlète en fonction de la charge manipulée.\n
- Aide mémoire en lien avec l'analyse de données:\n
1. Chaque point présenté dans le graphique équivault à la moyenne des trois meilleurs performances de chaque séance d'entrainement.
2. La "différence de performance" correspond à la différence des deux meilleurs performances exécutées par l'athlète.
3. La taille de Cohen est une façon de catégoriser l'effet de l'intervention. Les catégories sont les suivantes : Faible = 0.2* Écart-type, Modéré = 0.5 * ÉT, Grand = 0.8 * ÉT). Par exemple, après un cycle de puissance, mon athlète a amélioré sa vitesse maximale au squat avec une charge de 50lbs, passant de 1.3 à 1.4 m/s. Selon la taille de Cohen, cette amélioration est considéré comme 'faible' puisqu'il a eu seulement une augmentation de la vitesse de 0.1m/s."""

txt_info_session = """Cet onglet a pour but d'afficher toutes les données brutes recueillies par l'accéléromètre Enode, soit après le filtrage des données nulles et abérrantes.\n
NB : Chaque ligne affiché représente les informations pour chaque répétition effectuée. Ainsi, il suffit de lire le tableau de gauche à droite pour suivre la progression de l'athlète dans son entraînement. """



# General fonctions
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
    kg = weight_lbs * 0.453592
    return kg

def kg_to_lbs(kg_values):
    lbs_values = kg_values * 2.20462
    return round(lbs_values,2)

def find_difference_of_2_variables(value1, value2):
    difference_percentage = (value2 - value1) * 100 / min(abs(value2), abs(value1))
    rounded_percentage = round(difference_percentage, 2)
    return rounded_percentage

def remove_outliers(column):
    q1 = column.quantile(0.25)
    q3 = column.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.70 * iqr
    upper_bound = q3 + 1.70 * iqr
    return column.apply(lambda x: None if x < lower_bound or x > upper_bound else x)


# Illustration fonctions
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

def graph_manager_tableau(graphs_list, expander):
    for i, fig in enumerate(graphs_list):
        with expander:
            graph_title = f"Tableau {i + 1}"
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


# Class for each fonctionnality of the application (Courbe FV, Analyse détaillée de plusieurs séances, Analyse d'une séance)
class ForceVelocityCurve:
    def __init__(self, selected_user, selected_exercice, start_date, end_date):
        self.selected_user = selected_user
        self.selected_exercice = selected_exercice
        self.strt_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.flt_data = df[(df['User'] == selected_user) & (df['Exercise'] == selected_exercice) & (
            df['Date'].between(self.strt_date, self.end_date))]
        self.figure = None
        self.table_data_stats = None
        self.table_data_recommandation = None

        self.popt = None
        self.R2 = None
        self.SEE = None
        self.FV_trend_x = None
        self.FV_trend_y = None
        self.F1RM_x = None
        self.F1RM_y = None
        self.Pmax_y = None
        self.Pmax_x = None
        self.Fmax = None
        self.Vmax = None

        self.is_type_of_exercice = None
        self.selectbox = "Force absolue"
        self.speed_minMax = None
        self.repetition_maxMin = None
        self.load_maxMin = None
        self.area_chart = None

        self.is_comparaison_figure = False

    def show_graph(self):
        col1, col2 = st.columns([2, 1])
        with col1:
            st.plotly_chart(self.figure, use_container_width=True)
        with col2:
            if not self.is_comparaison_figure:
                fig = make_subplots(rows=2, cols=1, vertical_spacing=0.1,specs=[[{"type": "table"}],
               [{"type": "table"}]])
                fig.add_trace(self.table_data_stats, row=1,col=1)
                fig.add_trace(self.table_data_recommandation,row=2,col=1)

                fig.update_layout(
                    margin=dict(l=0, r=0, t=0, b=0),
                    autosize=True
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.plotly_chart(go.Figure(self.table_data_stats), use_container_width=True,)

    def equation(self, x, a, b):
        return a * x + b
    def find_V1RM_of_exercice(self):
        exo = self.selected_exercice
        V1RM = None
        if "Pull" in exo or "Row" in exo or "Tirade" in exo: V1RM = 0.4
        elif "Squat" in exo: V1RM = 0.30
        elif "Deadlift" in exo: V1RM = 0.15
        elif exo == "Bench press": V1RM = 0.17
        elif exo == "Prone bench pull": V1RM = 0.50
        elif exo == "Pull-up": V1RM = 0.23
        elif exo == "Seated military press": V1RM = 0.19
        elif exo == "Lat pulldown" : V1RM = 0.47
        elif exo == "Seated cable row": V1RM = 0.40
        elif exo == "Hip thrust" : V1RM = 0.25
        elif exo == "Leg press" : V1RM = 0.21
        else: V1RM = 0.3
        return V1RM
    def create_fv_graph(self):
        # Get lists of infos for FV graph
        each_set = self.flt_data['Set order'].unique()

        mean_force_y = [
            self.flt_data[self.flt_data['Set order'] == set][load_title].unique().tolist()
            for set in each_set
        ]
            #Transform a list of lists in a simple list
        mean_force_y = list(itertools.chain(*mean_force_y))

        # Calculez la moyenne de chaque groupe
        grouped_data = self.flt_data.groupby([load_title, 'Date', 'Set order'])['Avg. velocity [m/s]'].nlargest(3).reset_index()
        std_deviation_x = grouped_data.groupby([load_title, 'Date', 'Set order'])[
            'Avg. velocity [m/s]'].std().reset_index()
        grouped_data = grouped_data.groupby([load_title, 'Date', 'Set order'])['Avg. velocity [m/s]'].mean().reset_index()

        std_deviation_x = std_deviation_x['Avg. velocity [m/s]'].tolist()
        mean_velocity_x = grouped_data['Avg. velocity [m/s]'].tolist()


        mean_velocity_x = [v for v in mean_velocity_x if v is not None and not math.isnan(v)]
        mean_force_y = mean_force_y[:len(mean_velocity_x)]
        std_deviation_x = [v for v in std_deviation_x if v is not None and not math.isnan(v)]

        st.write(grouped_data)
        st.write(mean_force_y)
        st.write(mean_velocity_x)
        st.write(std_deviation_x)




            #Check wich body part is working on this exercise
        exercice_UB_pull = ["Pull", "Row", "Tirade"]
        exercice_UB_push = ["Bench", "Press", "Push"]
        exercice_LB_pull = ["Deadlift", "Hip"]
        exercice_LB_push = ["squat", "lunge", "fente"]

        is_UB_exercice_pull = any(exo in self.selected_exercice.lower() for exo in exercice_UB_pull)
        is_UB_exercice_push = any(exo in self.selected_exercice.lower() for exo in exercice_UB_push)
        is_LB_exercice_pull = any(exo in self.selected_exercice.lower() for exo in exercice_LB_pull)
        is_LB_exercice_push = any(exo in self.selected_exercice.lower() for exo in exercice_LB_push)
        self.is_type_of_exercice = [is_UB_exercice_pull, is_UB_exercice_push, is_LB_exercice_pull, is_LB_exercice_push]
        self.F1RM_x = self.find_V1RM_of_exercice()

        hovertemplate = f'Vitesse: %{{x:.2f}} m/s <br> Charges: %{{y:.2f}} {find_unit_of_title(load_title)}'
        figFV = go.Figure(
            data=go.Scatter(x=mean_velocity_x, y=mean_force_y, mode='markers',
                            name=f" {create_acronym(self.selected_user)} \n, {self.selected_exercice}",
                            hovertemplate=hovertemplate)
        )


        # Code for estimation of equation and R2 of the FV relation
        popt = curve_fit(self.equation, mean_velocity_x, mean_force_y)
        self.popt = popt[0]
        a_opt, b_opt = popt[0]

        self.FV_trend_x = np.linspace(0, 2.5, 100)
        self.FV_trend_y = self.equation(self.FV_trend_x, *popt[0])


        positive_numbers_of_y = []
        for i, y in enumerate(self.FV_trend_y):
            if y >= 0:
                positive_numbers_of_y.append(i)

        self.FV_trend_x = [self.FV_trend_x[i] for i in positive_numbers_of_y]
        self.FV_trend_y = [self.FV_trend_y[i] for i in positive_numbers_of_y]

        # Trouver l'indice du maximum de vitesse et maximum de force de la courbe FV
        self.Vmax =  max(self.FV_trend_x)
        self.Fmax = max(self.FV_trend_y)
        self.F1RM_y = self.equation(self.F1RM_x, *popt[0])



        fv_trace = go.Scatter(x=self.FV_trend_x, y=self.FV_trend_y, mode='lines',
                              name=f"Courbe FV ({self.strt_date.date()}/{self.end_date.date()})",
                              hovertemplate=hovertemplate)

        figFV.add_trace(fv_trace)

        mean_velocity_x = np.array(mean_velocity_x)
        mean_force_y = np.array(mean_force_y)

        #Finding R2
        residuals = mean_force_y - self.equation(mean_velocity_x, a_opt, b_opt)
        ss_residual = np.sum(residuals ** 2)
        ss_total = np.sum((mean_force_y - np.mean(mean_force_y)) ** 2)
        self.R2 = 1 - (ss_residual / ss_total)
        #Finding Standard Error of the Estimate)
        df_residual = len(mean_force_y) - 2
        variance_residual = ss_residual/df_residual
        self.SEE = round(np.sqrt(variance_residual),2)


        #Code for power curve
        if parameter_chkbox_is_in_kg:
            power_curve_y = pd.Series(self.FV_trend_y) * 9.81 * pd.Series(self.FV_trend_x)
        else:
            power_curve_y = pd.Series(self.FV_trend_y) * 4.44822 * pd.Series(self.FV_trend_x)

        figFV.add_trace(go.Scatter(x=self.FV_trend_x, y=power_curve_y, mode='lines',
                                   name=f"Courbe Puissance ({self.strt_date.date()}/{self.end_date.date()})",
                                   hovertemplate=f'Puissance: %{{y:.2f}} W <br> Vitesse: %{{x:.2f}} m/s',
                                   opacity=0.5,
                                   line=dict(color=fv_trace.line.color)))


        # Trouver l'indice du maximum de la courbe de puissance
        max_power_index = power_curve_y.idxmax()
        self.Pmax_x = self.FV_trend_x[max_power_index]
        self.Pmax_y = power_curve_y[max_power_index]


        #Ajout du 1RM et Pmax dans le graphique
        figFV.add_trace(go.Scatter(x=[0, self.F1RM_x,self.Pmax_x, self.Vmax], y=[self.Fmax,self.F1RM_y, self.Pmax_y, 0],
                                   mode='markers',
                                   marker=dict(symbol=1, size=7),
                                   name="Fmax/Pmax/Vmax"))

        figFV.update_layout(
            template="plotly",
            xaxis_title='Vitesse moyenne (m/s)',
            yaxis_title=f'Charge ({find_unit_of_title(load_title)})/Puissance (W)',
            title='Relation Force-Vitesse',
            showlegend=True,
            yaxis=dict( rangemode='nonnegative'),
            hovermode="x unified",
            margin=dict(l=50, r=50, b=50, t=50),
            legend=dict(orientation="h", x=0, y=-0.15)
        )
        #Warning if data is minimal
        if len(mean_velocity_x) < 10:
            st.warning(f"Attention : Le nombre de données utilisées pour créer le graphique de {self.selected_user} ({self.strt_date.date()}/{self.end_date.date()}) est limité. Les résultats pourraient être imprécis.", icon="⚠️")
        self.figure = figFV

    def create_fv_zone_infos(self, fv_selectbox_choice):
        if not self.is_comparaison_figure:
            a_opt, b_opt = self.popt

            # Checking wich infos to display based on Fv_selectbox_choice
            v_min, v_max = 0, 0
            if self.is_type_of_exercice[0] or self.is_type_of_exercice[1]:
                if fv_selectbox_choice == "Force absolue":
                    v_min, v_max = self.F1RM_x, 0.75
                elif fv_selectbox_choice == "Force-vitesse":
                    v_min, v_max = 0.75, 1
                elif fv_selectbox_choice == "Vitesse-force":
                    v_min, v_max = 1, 1.3
                elif fv_selectbox_choice == "Vitesse absolue":
                    v_min, v_max = 1.3, 1.5
            elif self.is_type_of_exercice[2] or self.is_type_of_exercice[3]:
                if fv_selectbox_choice == "Force absolue":
                    v_min, v_max = self.F1RM_x, 0.75
                elif fv_selectbox_choice == "Force-vitesse":
                    v_min, v_max = 0.75, 1
                elif fv_selectbox_choice == "Vitesse-force":
                    v_min, v_max = 1, 1.5
                elif fv_selectbox_choice == "Vitesse absolue":
                    v_min, v_max = 1.3, 1.8
            else:
                if fv_selectbox_choice == "Force absolue":
                    v_min, v_max = self.F1RM_x, 0.75
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

            reps_min = int(round((1.0278 * self.Fmax - estimated_load_min) / (0.0278 * self.Fmax)))
            if reps_min < 0: reps_min = 0
            reps_max = int(round((1.0278 * self.Fmax - estimated_load_max) / (0.0278 * self.Fmax)))
            if reps_max < 0: reps_max = 0

            self.speed_minMax = [v_min, v_max]
            self.load_maxMin = [estimated_load_max, estimated_load_min]
            self.repetition_maxMin = [reps_max, reps_min]

            # Construction of the infos table
            info_table_stats = [
                ["Équation", "R<sup>2</sup>","Erreur de l'estimation", "F-zéro", "1RM","Pmax","V-zéro"],
                [f"{self.popt[0]:.2f}x + {self.popt[1]:.2f}",
                 f"{self.R2:.2f}",
                 f"\u00B1{self.SEE:.2f} ({find_unit_of_title(load_title)})",
                 f"{round(self.Fmax)} ({find_unit_of_title(load_title)})",
                 f"{round(self.F1RM_y)} ({find_unit_of_title(load_title)})",
                 f"{round(self.Pmax_y)} (W) (à {round(self.Pmax_x,2)} m/s)",
                 f"{round(self.Vmax,2)} (m/s)"]]

            self.table_data_stats = go.Table(
                header=dict(
                    values=["Infos statistiques",
                            f"{create_acronym(self.selected_user)} \n ({self.strt_date.date()} \n/ {self.end_date.date()})"]),
                cells=dict(values=info_table_stats))

            info_table_recommandations = [["Intervalles de vitesses", "Intervalles de charges",
                 "Répétitions avant l'échec"],[f"{self.speed_minMax[0]} - {self.speed_minMax[1]} m/s",
                 f"{self.load_maxMin[0]} - {self.load_maxMin[1]} {find_unit_of_title(load_title)}",
                 f"{self.repetition_maxMin[0]} - {self.repetition_maxMin[1]}"]]

            self.table_data_recommandation = go.Table(
                header=dict(
                    values=["Recommandations sur le type d'entraînement choisi",
                            f"{create_acronym(self.selected_user)} ({self.strt_date.date()}/{self.end_date.date()})"]),
                cells=dict(values=info_table_recommandations))


    def add_curve(self, graphs2):
        if not self.is_comparaison_figure:
            self.is_comparaison_figure = True
            # Add all the info of the graphs#2 to the graph#1

            # Thème de couleur bleu pour self.figure
            theme_color_self = cycle(["#6baed6","#08519c","#c6dbef"])
            # Thème de couleur orange pour graph2
            theme_color_graph2 = cycle(["#fd8d3c","#e6550d","#fdd0a2"])

            temporFig = go.Figure()
            # Ajouter les lignes de self.figure avec le thème de couleur bleu
            for trace in self.figure.data:
                new_trace = go.Scatter(
                    x=trace.x,
                    y=trace.y,
                    mode=trace.mode,
                    name=trace.name,
                    marker=dict(symbol=trace.marker.symbol, size=trace.marker.size),
                    line=dict(color=next(theme_color_self)),
                    hovertemplate=trace.hovertemplate
                )
                temporFig.add_trace(new_trace)
            self.figure = temporFig
            # Ajouter les lignes de graph2 avec le thème de couleur orange
            for trace in graphs2.figure.data:
                new_trace = go.Scatter(
                    x=trace.x,
                    y=trace.y,
                    mode=trace.mode,
                    name=trace.name,
                    marker=dict(symbol=trace.marker.symbol, size=trace.marker.size),
                    line=dict(color=next(theme_color_graph2)),
                    hovertemplate=trace.hovertemplate
                )
                self.figure.add_trace(new_trace)

            self.figure.update_layout(
                title='Relation Force-vitesse',
                yaxis=dict(title=f'Charges {find_unit_of_title(load_title)}/Puissance (W)', domain=[0.21, 1.0], rangemode='nonnegative'),
                yaxis2=dict(title=f'Diff.{find_unit_of_title(load_title)}', domain=[0.05, 0.20], rangemode='nonnegative'),
                xaxis=dict(title='Vitesse (m/s)'),
                legend=dict(orientation="h", x=0, y=-0.15)
            )
            if self.end_date > graphs2.end_date:
                recent_graph = self
                old_graph = graphs2
            else:
                recent_graph = graphs2
                old_graph = self
            # Create the statistic infos table
            info_table = [
                ["Équation", "R2", "Erreur de l'estimation","F-zéro", "1RM","Pmax", "V-zéro"],
                [f"{old_graph.popt[0]:.2f}x + {old_graph.popt[1]:.2f}", f"{old_graph.R2:.2f}",f"\u00B1{old_graph.SEE:.2f} ({find_unit_of_title(load_title)})", f"{round(old_graph.Fmax,2)} ({find_unit_of_title(load_title)})",f"{round(old_graph.F1RM_y,2)} ({find_unit_of_title(load_title)})", f"{round(old_graph.Pmax_y)} (W) (à {round(old_graph.Pmax_x,2)} m/s)", f"{round(old_graph.Vmax, 2)} (m/s)"],
                [f"{recent_graph.popt[0]:.2f}x + {recent_graph.popt[1]:.2f}", f"{recent_graph.R2:.2f}",f"\u00B1{recent_graph.SEE:.2f} ({find_unit_of_title(load_title)})",f"{round(recent_graph.Fmax)} ({find_unit_of_title(load_title)})",f"{round(recent_graph.F1RM_y,2)} ({find_unit_of_title(load_title)})", f"{round(recent_graph.Pmax_y)} (W) (à {round(recent_graph.Pmax_x,2)} m/s)", f"{round(recent_graph.Vmax, 2)} (m/s)"],
                ["-","-","-", f"{find_difference_of_2_variables(old_graph.Fmax,recent_graph.Fmax)} %",f"{find_difference_of_2_variables(old_graph.F1RM_y,recent_graph.F1RM_y)} %",  f"{find_difference_of_2_variables(old_graph.Pmax_y, recent_graph.Pmax_y)} %", f"{find_difference_of_2_variables(old_graph.Vmax, recent_graph.Vmax)} %"]
            ]

            self.table_data_stats = go.Table(
                header=dict(
                    values=["Infos statistiques",
                            f"{create_acronym(old_graph.selected_user)} \n ({old_graph.strt_date.date()} \n/ {old_graph.end_date.date()})",
                            f"{create_acronym(recent_graph.selected_user)}\n ({recent_graph.strt_date.date()}\n/ {recent_graph.end_date.date()})","Amélioration (%)"]
                ),
                cells=dict(values=info_table))
            self.table_data_recommandation = None

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
        # Initializing data for the graph
        unit_of_title = find_unit_of_title(self.slt_title)
            #Filter the data to have a df of the date, load and the mean of the 3 best performances of this load and date
        graph_infos = self.flt_data.groupby([load_title, 'Date'])[self.slt_title].nlargest(3).reset_index()
        stats_infos = graph_infos
        graph_infos = graph_infos.groupby([load_title, 'Date']).mean().reset_index()

        # Initializing colors for distinct display of the data
        colors = [
    '#E0FFFF', '#008080', '#003333',  # Teintes de cyan
    '#87CEEB', '#00A9A9', '#006666',  # Teintes de bleu
    '#E6E6FA', '#9370DB', '#4B0082',  # Teintes de violet
    '#00FF7F', '#3CB371', '#008000',  # Teintes de vert
    '#FF6347', '#FF4500', '#8B0000',  # Teintes d'orange
    '#FFD700', '#FFB90F', '#DAA520',  # Teintes de jaune
    '#FFC0CB', '#FF69B4', '#DB7093',  # Teintes de rose
    '#F0E68C', '#DAA520', '#B8860B'   # Teintes d'or
]
        unique_loads = graph_infos[load_title].unique()
        load_colors = colors[:len(unique_loads)]  # Utiliser autant de couleurs que de charges uniques

        load_to_color = {load: color for load, color in zip(unique_loads, load_colors)}

        graph_infos['Color'] = graph_infos[load_title].map(load_to_color)
        graph_infos['LegendGroup'] = graph_infos['Color'].apply(lambda color: str(color))

        # Initializing data for the infos table
        improvement_values = []
        perf_differences = []
        effect_sizes = []
        SWC = []
        effect_size_titles = []
        obj_performance = []


        for _, group in stats_infos.groupby(load_title):
            s = group[self.slt_title].std()
            s = round(s * 0.2, 2)
            SWC.append(s)
        SWC = np.mean(SWC)

        # Analysing data from the two best performances/find the Cohen effect/SWC
        for _, group in stats_infos.groupby(load_title):
            if len(group.groupby(['Date'])) > 2:

                mean_of_session = group.groupby(['Date']).mean().reset_index()
                top_2_performances = mean_of_session.nlargest(2, self.slt_title)[self.slt_title].values
                perf_diff = top_2_performances.max() - top_2_performances.min()
                improvement = round(perf_diff / top_2_performances[1] * 100, 2)


                effect_size = perf_diff
                effect_sizes.append(round(effect_size, 2))
                effect_size_titles.append(self.find_Cohen_interpretation(effect_size, SWC))

                obj_performance.append(round(SWC + top_2_performances.max(), 2))
            else:
                improvement = 'Pas assez de valeurs'
                perf_diff = 0
                effect_sizes.append("-")
                effect_size_titles.append("-")
                obj_performance.append("-")
            improvement_values.append(improvement)
            perf_differences.append(round(perf_diff, 2))
        perf_changes_combined = [f"{pourcentage} ({original_unit})" for pourcentage, original_unit in
                                 zip(improvement_values, perf_differences)]

        hovertemplate = f'Date: %{{x|%Y-%m-%d}}<br>Charge [{find_unit_of_title(load_title)}]: %{{customdata}} <br>' + self.slt_title + ': %{y:.2f}<extra></extra>'

        fig_combined = go.Figure()
        unique_loads = graph_infos[load_title].unique()

        for load in unique_loads:
            load_data = graph_infos[graph_infos[load_title] == load]

            fig_combined.add_trace(go.Scatter(
                x=load_data['Date'],
                y=load_data[self.slt_title],
                mode='markers',
                marker=dict(
                    color=load_data['Color'],
                    size=8,
                    line=dict(width=1, color='black'),
                ),
                showlegend=True,
                legendgroup=str(load_data['LegendGroup'].iloc[0]),
                # Utiliser le premier élément de LegendGroup pour regrouper les traces
                name=f"Charge {round(load,2)} ({find_unit_of_title(load_title)})",
                hovertemplate=hovertemplate,
                customdata=load_data[load_title],
            ))

        fig_combined.update_layout(
            title=f"{self.user}, {selected_exercice_DA}, {self.slt_title}",
            xaxis_title=f'Date',
            yaxis_title=self.slt_title,
            legend=dict(title=load_title),
        )
        rounded_unique_loads = [round(value, 2) for value in graph_infos[load_title].unique()]
        improvement_table = go.Figure(data=go.Table(
            header=dict(values=[f'Charge ({find_unit_of_title(load_title)})', f"Changement dans la performance (%,({unit_of_title}))"
                                , "Taille d'effet",
                                f"Objectif de performance ({unit_of_title})"]),
            cells=dict(values=[
                rounded_unique_loads,
                perf_changes_combined,
                effect_size_titles,
                obj_performance
            ])
        ))

        improvement_table.update_layout(margin=dict(t=0, b=0))
        self.fig = fig_combined
        self.tabl = improvement_table


class Session_dataframe:
    def __init__(self, flt_data, user_name, exercice, date, is_filter):
        self.data = self.reorganise_data(flt_data)
        self.name = user_name
        self.exercice = exercice
        self.date = date
        self.is_filter = is_filter

    def reorganise_data(self, data_to_organise):
        first_columns = [load_title, 'Set order', 'Rep order']
        other_columns = sorted(list(data_to_organise.columns.drop(first_columns)))
        return data_to_organise.reindex(columns=first_columns + other_columns).reset_index()

    def show_dataframe(self):
        txt_is_filter = None
        if self.is_filter:
            txt_is_filter = "données filtrées"
        else:
            txt_is_filter = "données originales"
        st.subheader(f"{self.exercice}, {self.name} ({self.date}), {txt_is_filter}")
        st.dataframe(self.data, use_container_width=True)


##MAIN LOOP

# Importation des données
upload_file = st.sidebar.file_uploader("Choisir un fichier TSV", type='tsv')

if upload_file is not None:
    emplacement_logo.empty()
    main_title.empty()
    try:
        df = pd.read_table(upload_file)
            #Remove columns
        columns_to_drop = ['Total volume', 'Maximum load']
        for col in df.columns:
            if any(word in col for word in columns_to_drop):
                df = df.drop(columns=[col])
            #Find wich type of unit the df use
        load_title = df.filter(like='Load').columns[0]
        estimated_1RM_title = df.filter(like='Estimated 1RM').columns[0]
        if "lb" in find_unit_of_title(estimated_1RM_title):
            df_in_kg = False
        else :
            df_in_kg = True
        df['Date'] = pd.to_datetime(df['Date'])
        titres = df.columns.tolist()

            #Filter illogical data
        df_original = df
        df = df.replace(0, None)
        for col in df.select_dtypes(include=['number']).columns:
            if col not in ['Date', 'Time', 'User', 'Exercise', 'Load [lb]', 'Set order', 'Rep order']:
                df[col] = remove_outliers(df[col])





        exp_parametre = st.sidebar.expander("Paramètres")
        if exp_parametre.expanded:
            parameter_chkbox_is_in_kg = exp_parametre.checkbox("kg", value=df_in_kg)
            if df_in_kg and not parameter_chkbox_is_in_kg:

                # Convertir les colonnes estimate_title et load_title de kg à lb
                df["Estimated 1RM [lb]"] = kg_to_lbs(df["Estimated 1RM [lb]"])
                df["Load [lb]"] = kg_to_lbs(df["Load [lb]"])

                # Supprimer les colonnes existantes
                df.drop(columns=["Estimated 1RM [kg]", "Load [kg]"], inplace=True)

                # Changer les titres des colonnes
                estimated_1RM_title = "Estimated 1RM [lb]"
                load_title = "Load [lb]"
                df.rename(columns={"Estimated 1RM [kg]": estimated_1RM_title,
                                   "Load [kg]": load_title}, inplace=True)

            elif not df_in_kg and parameter_chkbox_is_in_kg:

                # Convertir les colonnes estimate_title et load_title de lb à kg
                df["Estimated 1RM [kg]"] = lbs_to_kg(df["Estimated 1RM [lb]"])
                df["Load [kg]"] = lbs_to_kg(df["Load [lb]"])

                # Supprimer les colonnes existantes
                df.drop(columns=["Estimated 1RM [lb]", "Load [lb]"], inplace=True)

                # Changer les titres des colonnes
                estimated_1RM_title = "Estimated 1RM [kg]"
                load_title = "Load [kg]"
                df.rename(columns={"Load [lb]": load_title,
                                   "Estimated 1RM [lb]": estimated_1RM_title}, inplace=True)


        # Fonctionnality #1. Analyse de la courbe force-vitesse
        exp_courbe_FV = st.sidebar.expander("Analyse courbe force-vitesse")
        if exp_courbe_FV.expanded:
            if exp_courbe_FV.checkbox("Info", key="Info_FV"):
                st.info(txt_info_FV)

            selected_user_fv = exp_courbe_FV.selectbox('Sélectionner un utilisateur', sorted(df['User'].unique()),
                                                       key="User_FV")
            selected_exercice_fv = exp_courbe_FV.selectbox('Sélectionner un exercice',
                                                           sorted(df[(df['User'] == selected_user_fv)][
                                                                      'Exercise'].unique()),
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

            # Remove "Ajout relation FV" if Flywheel or jump exercice is selected
            if "Flywheel" in selected_exercice_fv or "jump" in selected_exercice_fv:
                exp_courbe_FV.button("Ajouter relation force-vitesse", disabled=True)
            elif exp_courbe_FV.button("Ajouter relation force-vitesse", disabled=False):
                graph = ForceVelocityCurve(selected_user_fv, selected_exercice_fv, start_date_fv, end_date_fv)
                graph.create_fv_graph()
                st.session_state.graphs_of_fv.append(graph)

            # Button to create FV graph comparaison
            if exp_courbe_FV.button("Comparer les courbes force-vitesse"):
                lgr_graph_Fv = len(st.session_state.graphs_of_fv)
                st.session_state.graphs_of_fv[lgr_graph_Fv - 2].add_curve(
                    st.session_state.graphs_of_fv[lgr_graph_Fv - 1])
                st.session_state.graphs_of_fv.pop()

            fv_selectbox_choice = exp_courbe_FV.selectbox("Type d'entraînement",
                                                          ["Force absolue", "Force-vitesse",
                                                           "Vitesse-force",
                                                           "Vitesse absolue"])
            graph_manager(st.session_state.graphs_of_fv, exp_courbe_FV)
            show_all_FV_graphs_in_loop(fv_selectbox_choice)

        # Fonctionnality #2. Analyse de variables
        exp_detail_analysis = st.sidebar.expander("Analyse détaillée des performances", expanded=False)

        if exp_detail_analysis.expanded:
            if exp_detail_analysis.checkbox("Info", key="Info_DA"):
                st.info(txt_info_DA)
            selected_user_DA = exp_detail_analysis.selectbox('Sélectionner un utilisateur', sorted(df['User'].unique()),
                                                             key="User_AD")
            selected_exercice_DA = exp_detail_analysis.selectbox('Sélectionner un exercice',
                                                                 sorted(df[(df['User'] == selected_user_DA)][
                                                                            'Exercise'].unique()),
                                                                 key="Exo_AD")

            # Adjust the order of the titles (the available variables) by putting some at the start of the selection to be favored
            sorted(titres)
            custom_order_titles = ['Time to peak velocity [ms]', 'Peak velocity [m/s]', 'Peak power [W]',
                                   'Peak RFD [N/s]']
            sorted_remaining_titles = sorted([title for title in titres if title not in custom_order_titles])
            excluded_titles_DA = ['Date', 'Time', 'User', 'Exercise', 'Set order', 'Rep order', 'Load',
                                  'Estimated 1RM', 'Total volume', 'Maximum load']

            sorted_titles = custom_order_titles + sorted_remaining_titles
            selected_title_DA = exp_detail_analysis.selectbox(
                "Sélectionnez une variable à analyser",
                [title for title in sorted_titles if
                 title not in excluded_titles_DA and
                 all(keyword not in title for keyword in excluded_titles_DA) and
                 df[(df['User'] == selected_user_DA) & (df['Exercise'] == selected_exercice_DA)][
                     title].notnull().any()],
                key="Title_AD"
            )
            filtered_df_data_DA = df[(df['User'] == selected_user_DA) & (df['Exercise'] == selected_exercice_DA)]

            # Trouver la date minimale et la date maximale dans le dataframe filtré
            start_date = filtered_df_data_DA['Date'].min()
            end_date = filtered_df_data_DA['Date'].max()

            # Button to add the graph
            if exp_detail_analysis.button("Ajouter un graphique", key='bt_add_graph_DA'):
                filtrd_data_DA = df[(df['User'] == selected_user_DA) & (
                    df['Date'].between(pd.to_datetime(start_date), pd.to_datetime(end_date))) & (
                                            df['Exercise'] == selected_exercice_DA)]
                graph_DA = Multiple_Dates_Analysis(filtrd_data_DA, selected_user_DA, selected_exercice_DA,
                                                   selected_title_DA)
                graph_DA.find_data_from_dates()
                st.session_state.graphs_of_variables.append(graph_DA)

            graph_manager(st.session_state.graphs_of_variables, exp_detail_analysis)
            for g in st.session_state.graphs_of_variables:
                g.show_graph_DA()

        # Fonctionnality #3. Analyse d'une session
        exp_detail_session = st.sidebar.expander("Analyse d'une séance", expanded=False)
        if exp_detail_session.expanded:
            if exp_detail_session.checkbox("Info", key="Info_session"):
                st.info(txt_info_session)
            selected_user_session = exp_detail_session.selectbox('Sélectionner un utilisateur',
                                                                 sorted(df['User'].unique()),
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

            if exp_detail_session.button("Ajouter un tableau (avec filtrage)", key='bt_add_graph_session_with_filter'):
                filtered_df = df[
                    (df['User'] == selected_user_session) & (df['Exercise'] == selected_exercice_session) & (
                            df['Date'] == selected_date_single_session)].dropna(axis=1, how='all')
                filtered_df.reset_index()
                columns_to_drop = ['Date', 'Time', 'User', 'Exercise']
                for col in filtered_df.columns:
                    if any(word in col for word in columns_to_drop):
                        filtered_df = filtered_df.drop(columns=[col])
                st.session_state.dataframe_of_session.append(
                    Session_dataframe(filtered_df, selected_user_session, selected_exercice_session,
                                      selected_date_single_session, is_filter = True))
            if exp_detail_session.button("Ajouter un tableau (sans filtrage)", key='bt_add_graph_session_without_filter'):
                filtered_df = df_original[
                    (df['User'] == selected_user_session) & (df['Exercise'] == selected_exercice_session) & (
                            df['Date'] == selected_date_single_session)].dropna(axis=1, how='all')
                filtered_df.reset_index()
                columns_to_drop = ['Date', 'Time', 'User', 'Exercise']
                for col in filtered_df.columns:
                    if any(word in col for word in columns_to_drop):
                        filtered_df = filtered_df.drop(columns=[col])

                st.session_state.dataframe_of_session.append(
                    Session_dataframe(filtered_df, selected_user_session, selected_exercice_session,
                                      selected_date_single_session, is_filter = False))

            graph_manager_tableau(st.session_state.dataframe_of_session, exp_detail_session)
            for g in st.session_state.dataframe_of_session:
                g.show_dataframe()

    except pd.errors.EmptyDataError:
        st.error("Le fichier est vide ou ne contient pas de colonnes.")

else:
    emplacement_logo.image(
        "https://github.com/killmotion2/ProfilePuissanceTC/blob/main/Logo_Tennis_Canada.png?raw=true", width=100)
    main_title.header("Analyse du profil de puissance des athlètes de tennis")
    st.sidebar.warning("Veuillez télécharger un fichier TSV valide.")
