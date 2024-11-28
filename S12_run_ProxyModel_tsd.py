# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 13:40:44 2017

@author: Oliver Braganza
"""
import S5_ProxyModel1 as pm1
# import sys
from mesa.batchrunner import BatchRunner
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib as mpl
import numpy as np
import pandas as pd
import scipy.stats as stats

# make pdf text illustrator readable
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

""" Batch Runner
parameters:
data_collect_interval: interval between timesteps for data collection
width, height: size of toroidal grid (left/right and top/bottom rap around)
practice_mutation_rate: sd of gaussian theta change during inheritance
talend_sd: sd of talent distribution
competition: level of competition (0 to 1) = fraction that loses/dies
numAgents: number of agents in a population
selection_pressure: probability of loser death
survival_uncertainty: (currently not in Model) sd of erf prospect
goal_scale: scaling factor of psychological goal valuation
goal_angle: angle between proxy and goal (0° to 180°) defining practice space
"""

data_collect_interval = 1
finalStep = 100

variable_parameters = {"talent_sd": [0, 0.25, 0.5, 1, 2, 3]}
parameters = {"data_collect_interval": data_collect_interval,
              "width": 1, "height": 1,
              "competition": 0.9,
              "numAgents": 100,
              "goal_scale": 1,
              "goal_angle": np.pi/4,
              "selection_pressure": 0.1,
              "practice_mutation_rate": np.pi/90,
              "angle_agency": 0}

batch_run = BatchRunner(pm1.ProxyModel, variable_parameters, parameters, iterations=10, max_steps=finalStep+1,
                        model_reporters={"DataCollector": lambda ProxyModel: ProxyModel.datacollector})

batch_run.run_all()

''' Batch run data '''
data_pre = batch_run.get_model_vars_dataframe()

''' Data Collector data (all steps, model level) '''
collector_data = pd.DataFrame()

for run in data_pre.Run:
    current_run = data_pre.DataCollector[run].get_model_vars_dataframe()
    current_run["Run"] = run
    current_run["Step"] = current_run.index * data_collect_interval
    collector_data = collector_data.append(current_run)

''' Full data (model level)'''
modeldata = pd.merge(data_pre, collector_data, on="Run")
modeldata = modeldata.drop("DataCollector", 1)

''' Data Collector data (all steps, agent level) '''
collector_data = pd.DataFrame()
for run in data_pre.Run:
    current_run = data_pre.DataCollector[run].get_agent_vars_dataframe()
    current_run["Run"] = run
    current_run.reset_index(inplace=True)
    current_run["Step"] = current_run["Step"] * data_collect_interval
    collector_data = collector_data.append(current_run)

''' Full data (agent level)'''
agentdata = pd.merge(data_pre, collector_data, on="Run")
agentdata = agentdata.drop("DataCollector", 1)


def showModel(modeldata):
    """ model vizualizations (particular timestep)"""

    visualizeStep = max(modeldata.Step.unique())
    f, ((ax1, ax2, ax3)) = plt.subplots(1, 3, figsize=(8, 2))
    f.subplots_adjust(hspace=.3, wspace=.6, left=0.1, right=0.9)
    plt.suptitle('Model average, Step ' + str(visualizeStep))

    modeldata = modeldata[modeldata.Step == visualizeStep]
    # colormap
    cmap = plt.cm.jet
    cNorm = colors.Normalize(vmin=np.min(modeldata.talent_sd),
                             vmax=np.max(modeldata.talent_sd))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
    low_col = scalarMap.to_rgba(min(modeldata.talent_sd))
    high_col = scalarMap.to_rgba(max(modeldata.talent_sd))

    # compute means & std
    xVals = modeldata.talent_sd.unique()
    Gm = modeldata.groupby(['talent_sd'])['mean_goal_value'].mean()
    Gstd = modeldata.groupby(['talent_sd'])['mean_goal_value'].std()
    Pm = modeldata.groupby(['talent_sd'])['mean_proxy_value'].mean()
    Pstd = modeldata.groupby(['talent_sd'])['mean_proxy_value'].std()

    ax1.scatter(modeldata.talent_sd, modeldata.mean_goal_value,
                alpha=0.5, c=low_col, marker=".", label='_nolegend_')
    ax1.scatter(modeldata.talent_sd, modeldata.mean_proxy_value,
                alpha=0.5, c=high_col, marker=".", label='_nolegend_')
    ax1.plot(xVals, Pm, color=high_col, label='proxy')
    ax1.fill_between(xVals, Pm-Pstd, Pm+Pstd, color=high_col, alpha=0.2)
    ax1.plot(xVals, Gm, color=low_col, label='goal')
    ax1.fill_between(xVals, Gm-Gstd, Gm+Gstd, color=low_col, alpha=0.2)

    ax1.legend()
    ax1.set_ylabel('mean value')
    ax1.set_xlabel('talent_sd')

    utility_mean = modeldata.groupby(['talent_sd'])['mean_utility'].mean()
    utility_std = modeldata.groupby(['talent_sd'])['mean_utility'].std()
    ax2.scatter(modeldata.talent_sd, modeldata.mean_utility,
                alpha=0.2, c='g', marker=".", label='iterations')
    ax2.plot(xVals, utility_mean, c='g')
    ax2.fill_between(xVals, utility_mean-utility_std,
                     utility_mean+utility_std, alpha=0.2, color='g')
    ax2.set_ylabel('mean utility', color='g')
    ax2.set_xlabel('talent_sd')

    groups_m = modeldata.groupby(['talent_sd'])['mean_practice']
    practice_mean = np.zeros(len(parameters['talent_sd']))
    i = 0
    for name, group in groups_m:
        practice_mean[i] = stats.circmean(group, -np.pi, np.pi)
        i += 1
    groups_std = modeldata.groupby(['talent_sd'])['mean_practice']
    practice_std = np.zeros(len(parameters['talent_sd']))
    i = 0
    for name, group in groups_std:
        practice_std[i] = stats.circstd(group, -np.pi, np.pi)
        i += 1
    practice_mean = practice_mean/np.pi*180
    # practice_std = modeldata.groupby(['talent_sd'])['mean_practice'].std()
    practice_std = practice_std/np.pi*180
    ax2a = ax2.twinx()
    ax2a.scatter(modeldata.talent_sd,
                 modeldata.mean_practice/np.pi*180,
                 alpha=0.2, c='k', marker=".", label='iterations')
    ax2a.plot(xVals, practice_mean, c='k')
    ax2a.fill_between(xVals, practice_mean-practice_std,
                      practice_mean+practice_std, alpha=0.2, color='k')
    ax2a.set_ylim([0, 23])
    ax2a.set_ylabel('mean practice (°)')

    # vector plot
    G_oc_m = modeldata.groupby(['talent_sd'])['mean_goal_oc'].mean()

    # plot mean chosen practices as vectors
    ax3.set_ylabel('goal (oc)')
    ax3.set_xlabel('proxy')

    for i in range(len(xVals)):
        colorVal = scalarMap.to_rgba(xVals[i])
        ax3.arrow(0, 0, Pm.values[i], G_oc_m.values[i], color=colorVal)

    ax3.set_ylim([0, np.max(Pm)+1])
    ax3.set_xlim([0, np.max(Pm)+1])
    ax3.set_aspect('equal')
    cbar_ax = f.add_axes([0.92, 0.15, 0.01, 0.7])
    mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap, norm=cNorm,
                              orientation='vertical', label='talent_sd')

    f.savefig('Model.pdf')


def showAgentDynamics(agentdata):
    """ Agent Dynamics """
    columns = 4
    column_length = 15
    if finalStep > 100:
        f1, ax_array = plt.subplots(1, columns, figsize=(2, column_length))
    else:
        # column_length = int(len(agentdata.Step.unique())/10)
        f1, ax_array = plt.subplots(1, columns, figsize=(8, 2))
    f1.subplots_adjust(hspace=.3, wspace=.3, left=0.1, right=0.9)
    cbar_ax = f1.add_axes([0.92, 0.15, 0.03, 0.5])
    cNorm = colors.Normalize(vmin=np.min(agentdata.Effort),
                             vmax=np.max(agentdata.Effort))
    # cNorm = colors.Normalize(vmin=np.min(agentdata.Practice),
    #                         vmax=np.max(agentdata.Practice))

    for i, ax in enumerate(ax_array):
        if i == 0:
            ax.set_ylabel('step')
        else:
            ax.tick_params(labelleft='off')
        run = int(i * (max(agentdata.Run)+1)/(columns-0.9))
        run_data = agentdata[agentdata.Run == run]
        image = np.empty((len(run_data.Step.unique()),
                          len(run_data.AgentID.unique())))
        births = np.zeros((len(run_data.Step.unique()),
                          len(run_data.AgentID.unique())))
        for row, Step in enumerate(run_data.Step.unique()):
            stepdata = run_data[run_data.Step == Step]
            sortby = 'AgentID'
            image[row] = np.array(stepdata.sort_values(sortby)['Effort'])
            births[row] = np.array(stepdata.sort_values(sortby)['Genealogy'])
            if Step == max(run_data.Step.unique()):
                break
        births = np.gradient(births, axis=0)
        births = (births > 0)*1
        births = np.ma.masked_where(births < 0.9, births)

        ax.imshow(image, cmap="viridis", norm=cNorm,
                  extent=(0, max(run_data.AgentID.unique()),
                          max(run_data.Step.unique()), 0),
                  aspect=1/data_collect_interval)
        ax.imshow(births, cmap="Greys",
                  extent=(0, max(run_data.AgentID.unique()),
                          max(run_data.Step.unique()), 0),
                  aspect=1/data_collect_interval, alpha=0.95)

        ax.set_xlabel('agent #')
        sp = run_data.talent_sd.iloc[0]
        ax.set_title('tsd{:.2f} r{:02d}'.format(sp, run))

    mpl.colorbar.ColorbarBase(cbar_ax, norm=cNorm, orientation='vertical',
                              label='effort')

    f1.savefig('AgentDynamics.pdf')


def showAgents(agentdata):
    """ individual Agent outcomes """
    columns = 4
    f2, ax_array = plt.subplots(1, columns, figsize=(8, 2), sharey=True)
    f2.subplots_adjust(hspace=.3, wspace=.3, left=0.1, right=0.9)
    cmap = plt.cm.inferno
    cbar_ax = f2.add_axes([0.92, 0.15, 0.01, 0.7])
    cNorm = colors.Normalize(vmin=np.min(agentdata.Talent),
                             vmax=np.max(agentdata.Talent))

    # ga = agentdata.talent_sd.iloc[0]
    for i, ax in enumerate(ax_array):
        if i == 0:
            ax.set_ylabel('goal (oc)')
        else:
            ax.tick_params(labelleft='off')
        # get practice angles
        run = int(i * (max(agentdata.Run)+1)/(columns-0.9))
        goal_angle = agentdata[agentdata.Run == run].goal_angle.iloc[0]
        talent_sd = agentdata[agentdata.Run == run].talent_sd.iloc[0]
        talent_sd_data = agentdata[agentdata.talent_sd == talent_sd]
        proxy = np.zeros([len(talent_sd_data.AgentID.unique()),
                         len(talent_sd_data.Run.unique())])
        goal_oc = np.zeros([len(talent_sd_data.AgentID.unique()),
                            len(talent_sd_data.Run.unique())])
        talent = np.zeros([len(talent_sd_data.AgentID.unique()),
                           len(talent_sd_data.Run.unique())])
        runcolor = np.zeros([len(talent_sd_data.AgentID.unique()),
                             len(talent_sd_data.Run.unique())])
        for index, iteration in enumerate(talent_sd_data.Run.unique()):
            for Agent in talent_sd_data.AgentID.unique():
                proxy[Agent][index] = talent_sd_data[(talent_sd_data.Run == iteration) & (talent_sd_data.Step == finalStep)].Proxy.iloc[Agent]    
                goal_oc[Agent][index] = talent_sd_data[(talent_sd_data.Run == iteration) & (talent_sd_data.Step == finalStep)].Goal_oc.iloc[Agent]
                talent[Agent][index] = talent_sd_data[(talent_sd_data.Run == iteration) & (talent_sd_data.Step == finalStep)].Talent.iloc[Agent]
                runcolor[Agent][index] = index

        # determine plot range
        xub = np.max(agentdata.Proxy) + 0.5
        xlb = np.min(agentdata.Proxy) - 0.5
        yub = np.max(agentdata.Goal_oc) + 0.5
        ylb = np.min(agentdata.Goal_oc) - 0.5
        if xub-xlb > yub-ylb:
            yub = ylb + xub-xlb
        # plot goal angle
        ax.plot([0, np.cos(goal_angle)*np.max(agentdata.Goal)],
                [0, np.sin(goal_angle)*np.max(agentdata.Goal)],
                c='grey', ls='--', lw=0.5)
        ax.plot([0, np.max(agentdata.Proxy)],
                [0, 0],
                c='grey', ls='--', lw=0.5)
        ax.scatter(proxy, goal_oc, c=talent, cmap=cmap, norm=cNorm,
                   alpha=0.9, marker=".")
        # ax.set_xticks([25, 45, 65])
        ax.set_ylim([ylb, yub])
        ax.set_xlim([xlb, xub])
        # ax.set_aspect('equal')
        ax.set_xlabel('proxy')
    mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap, norm=cNorm,
                              orientation='vertical', label='talent')

    f2.savefig('Agents.pdf')


def showModelDynamics(modeldata):
    """ Model Dynamics for particular talent_sd """

    columns = 4
    f1, ax_array = plt.subplots(1, columns, figsize=(8, 2), sharey=True)
    f1.subplots_adjust(hspace=.3, wspace=.3, left=0.1, right=0.9)
    # colormap
    cmap = plt.cm.jet
    cNorm = colors.Normalize(vmin=np.min(modeldata.talent_sd),
                             vmax=np.max(modeldata.talent_sd))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
    low_col = scalarMap.to_rgba(np.min(modeldata.talent_sd))
    high_col = scalarMap.to_rgba(np.max(modeldata.talent_sd))

    for i, ax in enumerate(ax_array):
        if i == 0:
            ax.set_ylabel('value')
        else:
            ax.tick_params(labelleft='off')
        run = int(i * (max(modeldata.Run)+1)/(columns-0.9))
        talent_sd = modeldata[modeldata.Run == run].talent_sd.iloc[0]
        talent_sd_data = modeldata[modeldata.talent_sd == talent_sd]

        xVals = talent_sd_data.Step.unique()
        proxy_mean = talent_sd_data.groupby(['Step'])['mean_proxy_value'].mean()
        proxy_std = talent_sd_data.groupby(['Step'])['mean_proxy_value'].std()
        # ax.scatter(talent_sd_data.Step,
        #           talent_sd_data.mean_proxy_value,
        #           alpha=0.2, c=high_col, marker=".", label='_nolegend_')
        ax.plot(xVals, proxy_mean, c=high_col, label='proxy')
        ax.fill_between(xVals, proxy_mean - proxy_std,
                        proxy_mean + proxy_std, alpha=0.2, color=high_col)
        goal_mean = talent_sd_data.groupby(['Step'])['mean_goal_value'].mean()
        goal_std = talent_sd_data.groupby(['Step'])['mean_goal_value'].std()
        # ax.scatter(talent_sd_data.Step,
        #           talent_sd_data.mean_goal_value,
        #           alpha=0.2, c=low_col, marker=".", label='_nolegend_')
        ax.plot(xVals, goal_mean, c=low_col, label='goal')
        ax.fill_between(xVals, goal_mean - goal_std,
                        goal_mean + goal_std, alpha=0.2, color=low_col)
        ax.set_xlabel('step')
        # ax.set_title('c{:.1f} r{:02d}'.format(talent_sd, run))
        ax.set_ylim([np.min([modeldata.mean_proxy_value.min(),
                             modeldata.mean_goal_value.min()]),
                     np.max([modeldata.mean_proxy_value.max(),
                             modeldata.mean_goal_value.max()])])
        if i == 0:
            ax.legend()

    f1.savefig('Dynamics.pdf')

    f2, ax_array = plt.subplots(1, columns, figsize=(8, 1.2), sharey=True)
    f2.subplots_adjust(hspace=.3, wspace=.3, left=0.1, right=0.9)
    for i, ax in enumerate(ax_array):
        if i == 0:
            ax.set_ylabel('mean practice (°)')
        else:
            ax.tick_params(labelleft='off')
        run = int(i * (max(modeldata.Run)+1)/(columns-0.9))
        talent_sd = modeldata[modeldata.Run == run].talent_sd.iloc[0]
#        print(talent_sd/np.pi * 180)
        talent_sd_data = modeldata[modeldata.talent_sd == talent_sd]
        xVals = talent_sd_data.Step.unique()
        yVals_raw = talent_sd_data.groupby(['Step'])['mean_practice']
        practice_mean = np.empty(np.size(xVals))
        practice_std = np.empty(np.size(xVals))
        i = 0
        for name, group in yVals_raw:
            practice_mean[i] = stats.circmean(group, -np.pi, np.pi)
            practice_std[i] = stats.circstd(group, -np.pi, np.pi)
            i += 1

        practice_mean = practice_mean / np.pi * 180
        practice_std = practice_std / np.pi * 180
#        ax.scatter(talent_sd_data.Step,
#                   talent_sd_data.mean_practice,
#                   alpha=0.2, c="k", marker=".")
        ax.plot(xVals, practice_mean, c='k')
        ax.fill_between(xVals, practice_mean - practice_std,
                        practice_mean + practice_std, alpha=0.2, color="k")
        ax.plot([0, np.max(xVals)],
                [0, 0],
                c='grey', ls='--', lw=0.5)
        ax.set_xlabel('step')
        # ax.set_ylim([0,1])

    f2.savefig('PracticeDynamics.pdf')


def showAgentProxyDynamics(agentdata):
    """ Agent Dynamics """
    columns = 4
    # column_length = 15
    steps_to_average = 10
    # f1, ax_array = plt.subplots(1, columns, figsize=(2, column_length))

    # column_length = int(len(agentdata.Step.unique())/10)
    f1, ax_array = plt.subplots(1, columns, figsize=(8, 2))
    f1.subplots_adjust(hspace=.3, wspace=.1, left=0.1, right=0.9)
    # plt.suptitle('Agent Dynamics')
    cbar_ax = f1.add_axes([0.92, 0.15, 0.03, 0.5])
    cNorm = colors.Normalize(vmin=np.min(agentdata.Proxy),
                             vmax=np.max(agentdata.Proxy))
    # cNorm = colors.Normalize(vmin=np.min(agentdata.Practice),
    #                         vmax=np.max(agentdata.Practice))

    for i, ax in enumerate(ax_array):
        if i == 0:
            ax.set_ylabel('step')
        else:
            ax.tick_params(labelleft='off')
        run = int(i * (max(agentdata.Run)+1)/(columns-0.9))
        run_data = agentdata[agentdata.Run == run]
        # talent_sd = run_data.talent_sd.iloc[0]
        image = np.empty((len(run_data.Step.unique()),
                          len(run_data.AgentID.unique())))
        births = np.empty((len(run_data.Step.unique()),
                          len(run_data.AgentID.unique())))
        for row, Step in enumerate(run_data.Step.unique()):
            stepdata = run_data[run_data.Step == Step]
            sortby = 'Practice'
            image[row] = np.array(stepdata.sort_values(sortby)['Proxy'])
            births[row] = np.array(stepdata.sort_values(sortby)['Genealogy'])
            if Step == max(run_data.Step.unique()):
                break
        births = np.gradient(births, axis=0)
        births = (births > 0)*1
        births = np.ma.masked_where(births < 0.9, births)

        step_averaged = image.transpose().reshape(-1,steps_to_average).mean(1).reshape(20,-1).transpose()
        ax.imshow(step_averaged, cmap="inferno", norm=cNorm)
        # ax.imshow(births, cmap="Greys",
        #          extent=(0, max(run_data.AgentID.unique()),
        #                  max(run_data.Step.unique()),0),
        #                  aspect=1/data_collect_interval, alpha=0.95)

        ax.set_xlabel('agent #')
        # ax.set_title('c{:.1f} r{:02d}'.format(talent_sd, run))

    mpl.colorbar.ColorbarBase(cbar_ax, cmap='inferno', norm=cNorm,
                              orientation='vertical', label='proxy')

    f1.savefig('AgentProxyDynamics.pdf')


def showSortedAgents(agentdata):
    """ mean proxy and practice of agents sorted my practice in each run """
    columns = 4
    f2, ax_array = plt.subplots(1, columns, figsize=(8, 2))
    f2.subplots_adjust(hspace=.3, wspace=.3, left=0.1, right=0.9)

    StepsToAverage = 100
    Steps = range(finalStep-StepsToAverage, finalStep)
    # ga = agentdata.talent_sd.iloc[0]
    for i, ax in enumerate(ax_array):
        if i == 0:
            ax.set_ylabel('proxy')
        else:
            ax.tick_params(labelleft='off')
        # get practice angles
        run = int(i * (max(agentdata.Run)+1)/(columns-0.9))
        talent_sd = agentdata[agentdata.Run == run].talent_sd.iloc[0]
        talent_sd_data = agentdata[agentdata.talent_sd == talent_sd]
        proxy = np.zeros([len(talent_sd_data.AgentID.unique()),
                         len(talent_sd_data.Run.unique()),
                         StepsToAverage])
        practice = np.zeros([len(talent_sd_data.AgentID.unique()),
                             len(talent_sd_data.Run.unique()),
                             StepsToAverage])
        talent = np.zeros([len(talent_sd_data.AgentID.unique()),
                           len(talent_sd_data.Run.unique()),
                           StepsToAverage])

        for index, iteration in enumerate(talent_sd_data.Run.unique()):
            for sindex, step in enumerate(Steps):
                RunRowProxy = talent_sd_data[(talent_sd_data.Run == iteration) & (talent_sd_data.Step == step)].Proxy
                RunRowProxy = np.array(RunRowProxy)
                RunRowPractice = talent_sd_data[(talent_sd_data.Run == iteration) & (talent_sd_data.Step == step)].Practice
                RunRowPractice = np.array(RunRowPractice)
                PracticeRanks = RunRowPractice.argsort()
                RunRowProxy = RunRowProxy[PracticeRanks]
                RunRowPractice = RunRowPractice[PracticeRanks]
#                for Agent in talent_sd_data.AgentID.unique():   
                proxy[:, index, sindex] = RunRowProxy # talent_sd_data[(talent_sd_data.Run == iteration) & (talent_sd_data.Step == step)].Proxy.iloc[Agent]    
                practice[:, index, sindex] = RunRowPractice # talent_sd_data[(talent_sd_data.Run == iteration) & (talent_sd_data.Step == step)].Practice.iloc[Agent]
                    # talent[Agent][index][sindex] = talent_sd_data[(talent_sd_data.Run == iteration) & (talent_sd_data.Step == step)].Talent.iloc[Agent]

        # plot goal angle
        # ax.plot([0, 45], [0, 0], c='grey', ls='--', lw=0.5)
        xVals = talent_sd_data.AgentID.unique()
        proxyStepMean = np.mean(proxy, axis=2)
        means = np.mean(proxyStepMean, axis=1)
        stds = np.std(proxyStepMean, axis=1)
        ax.plot(xVals, means, color='k')
        ax.fill_between(xVals, means - stds, means + stds, alpha=0.2, color="k")
        # ax.set_xticks([25, 45, 65])

        ax.set_xlabel('practice rank')

    f2.savefig('SortedAgents.pdf')


modeldata.to_csv('model.csv')
agentdata.to_csv('agent.csv')

showAgentDynamics(agentdata)
showAgents(agentdata)
showModelDynamics(modeldata)
showModel(modeldata)
# showAgentProxyDynamics(agentdata)
# showSortedAgents(agentdata)
# for finalStep in range(35,45): showAgents(agentdata)
