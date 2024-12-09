# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 15:48:08 2017

@author: Oliver Braganza

proxyeconomics model
The model is based on mesa framework. Agent properties and computations are in
the Agent class, System level properties and computations in the Model class.
"""

from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import SingleGrid
import random
from mesa.datacollection import DataCollector
import numpy as np

''' Functions computing model level readouts for data collection '''


def compute_mean_proxy_value(model):
    """ Returns the mean total proxy value across agents """
    proxy_values = [agent.proxy for agent in model.agents]
    return sum(proxy_values)/model.num_agents


def compute_mean_goal_value(model):
    """ Returns the mean goal value across agents """
    goal_values = [agent.goal for agent in model.agents]
    return sum(goal_values)/model.num_agents


def compute_mean_goal_oc(model):
    """ Returns the mean independent goal component across agents """
    goal_oc = [agent.goal_oc for agent in model.agents]
    return sum(goal_oc)/model.num_agents


def compute_mean_effort(model):
    """ returns the mean effort across agents """
    effort_values = [agent.effort for agent in model.agents]
    return sum(effort_values)/model.num_agents


def compute_mean_utility(model):
    """ returns the mean utility across agents """
    utility_values = [agent.utility for agent in model.agents]
    return sum(utility_values)/model.num_agents


def compute_mean_practice(model):
    """ returns the mean practice angle across agents """
    pr_vals = [agent.practice for agent in model.agents]
    return np.arctan2(np.mean(np.sin(pr_vals)), np.mean(np.cos(pr_vals)))

def compute_mean_talent(model):
    """ returns the mean talent across agents """
    t_vals = [agent.talent for agent in model.agents]
    return sum(t_vals)/model.num_agents


class ProxyAgent(Agent):
    """
    Agent class
    - initialize agents (practice, effort,..
    - step agents (optimize effort/practice to maximize utility)
    """
    def __init__(self, unique_id, model):
        super().__init__(model)
        self.practice = np.random.uniform(0, self.model.goal_angle)
        # self.practice = self.model.goal_angle
        self.talent = np.random.normal(10, self.model.talent_sd)
        if self.talent < 0:
            self.talent = 0.01
        self.effort = 0
        self.proxy = np.cos(self.practice) * self.effort
        self.previous_step_proxy = 0
        self.goal = np.cos(self.model.goal_angle - self.practice) * self.effort
        self.goal_oc = np.sin(self.practice) * self.effort
        self.goal_scale = self.model.goal_scale
        self.utility = np.nan
        self.child_of = self.unique_id
        

    def step(self):
        """ Actions to perform on each time step """
        self.previous_step_proxy = self.proxy
        self.optimize_effort()

    def optimize_effort(self):
        """ Heuristic to optimize effort level (and potentially practice):
        Vary effort by test_list and check if utility increases.
        Utility has 3 components:
        1. proxy value (extrinsic)
        utility/disutility derived from prospect of surviving competition
        computed from own relative proxy-rank within population
        2. goal value (intrinsic)
        utility/disutility derived from contributing to the societal goal
        3. effort cost
        disutility due to effort expenditure
        effort cost = effort^2 /talent

        If agents have agency over the practice angle, they similarly optimize
        by going through test_list at every angle in angle_list.
        """
        test_list = [-10, -5, -1, -0.5, -0.1,
                     0, 0.1, 0.5, 1, 5, 10]

        ''' agency >0 introduces agency over the practice angle 
        0 means no agency; 1 means full agency '''
        agency = self.model.angle_agency
        angle_list = [self.practice]
        if np.random.rand() < agency:
            ''' social learning '''
#            angle_list = neighbor_practices
            ''' individual learning (gaming) '''
            own_practice = self.practice/np.pi*180
            change_angle = [-5, -1, 0, 1, 5]
            angle_list = [own_practice-x for x in change_angle]
            angle_list = np.deg2rad(angle_list)


        def get_prospect(self):
            """ calculates the utility/disutility from the prospect of winning/
            loosing competition """

            #''' list of neighbors proxy performances '''
            # neighbors = self.model.grid.get_neighbors(self.pos, moore=True, include_center=True, radius=2)
            # proxies = list(n.proxy for n in neighbors)
            agents = self.model.agents
            proxies = list(n.proxy for n in agents)
            self.proxy = np.cos(self.practice) * self.effort
            own_proxy = self.proxy

            rel_surv_thresh = self.model.competition
            ordered = np.sort(proxies)
            survival_threshold = ordered[int(rel_surv_thresh*len(proxies))-1]
            
            ''' survival threshold based on previous-step proxies '''
            # ps_proxies = list(n.previous_step_proxy for n in agents)
            # survival_threshold = ordered[int(rel_surv_thresh*len(ps_proxies))-1]
            
            if self.model.competition > 0:
                ''' McDermott Prospect '''
                # prospect = ps * ss.erf((own_proxy-survival_threshold)/survival_uncertainty)
                ''' Kahneman Tversky Prospect '''
                prospect = abs(own_proxy-survival_threshold)**0.88
                ''' Step Prospect '''
                # prospect = 1
            else:
                prospect = 0  # no competition
            if (own_proxy-survival_threshold) < 0:  # loss aversion
                prospect = -abs(prospect) * 2.25
                # prospect = -1
            return prospect

        def get_utility(self, prospect):
            ''' utility function '''
            self.goal = np.cos(self.model.goal_angle - self.practice) * self.effort
            gsc = self.goal_scale
            e = self.effort
            t = self.talent

            utility = prospect + gsc*self.goal - (e**2)/t

            return utility

        old_effort = self.effort
        new_effort = 0
        new_practice = self.practice
        max_utility = -1000
        for test_angle in angle_list:
            for test_effort in test_list:
                self.effort = old_effort + test_effort
                self.practice = test_angle
                if self.effort > 0:
                    prospect = get_prospect(self)
                    utility = get_utility(self, prospect)
                    if np.isnan(utility):
                        print('error: utility is nan')
                    if utility > max_utility:
                        max_utility = utility
                        new_effort = self.effort
                        new_practice = self.practice

        self.utility = max_utility
        self.effort = new_effort
        self.practice = new_practice
        self.oldproxy = self.proxy
        self.proxy = np.cos(self.practice) * self.effort 
        self.goal = np.cos(self.model.goal_angle - self.practice) * self.effort
        self.goal_oc = np.sin(self.practice) * self.effort


class ProxyModel(Model):
    """
    Model class
    - initialize model (agents on grid, time)
    - step model (implement selection/evolution, collect data)
    """
    def __init__(self,
                 data_collect_interval,
                 width, height,
                 competition,
                 numAgents, talent_sd,
                 goal_scale,
                 goal_angle,
                 selection_pressure,
                 practice_mutation_rate,
                 angle_agency,
                 max_steps,seed):
        
        self.data_collect_interval = data_collect_interval
        self.num_agents = numAgents
        self.selection_pressure = selection_pressure
        self.grid = SingleGrid(width, height, True)  # toroidal (all ends rap)
        self.talent_sd = talent_sd
        self.practice_mutation_rate = practice_mutation_rate
        self.competition = competition
        self.angle_agency = angle_agency
        self.goal_scale = goal_scale
        self.random = random.Random()
        self.running = True
        self.time = 0
        self.goal_angle = goal_angle
        self.max_steps = max_steps
        self.steps = 0
        #super().__init__(seed=seed)
        
        ''' Create agents on the grid '''
        for i in range(self.num_agents):
            A = ProxyAgent(i, self)
            ''' Add all agents row wise from top left to bottom right '''
            if self.grid.width > 1 and self.grid.height > 1:
                x = i % self.grid.width
                y = int(i/self.grid.height)
                self.grid.place_agent(A, (x, y))

        self.datacollector = DataCollector(
                model_reporters={"mean_proxy_value": compute_mean_proxy_value,
                                 "mean_goal_value": compute_mean_goal_value,
                                 "mean_goal_oc": compute_mean_goal_oc,
                                 "mean_effort": compute_mean_effort,
                                 "mean_utility": compute_mean_utility,
                                 "mean_practice": compute_mean_practice,
                                 "mean_talent": compute_mean_talent},
                agent_reporters={"Proxy": lambda A: A.proxy,
                                 "Goal": lambda A: A.goal,
                                 "Goal_oc": lambda A: A.goal_oc,
                                 "Utility": lambda A: A.utility,
                                 "Effort": lambda A: A.effort,
                                 "Practice": lambda A: A.practice,
                                 "Genealogy": lambda A: A.child_of,
                                 "Talent": lambda A: A.talent})

    def kill_and_replace(self):
        """ recompute rank with chosen effort levels
        randomly kill losers with probability = sp and
        replace with offspring from random winner
        (new agents 1. inherit their practice and effort from the parent,
        2. draw new random talent,
        3. take the location & ID from the dead agent to facilitate display.
        Deaths, births and genealogy are stored in "Genealogy")"""
        agents = self.agents
        proxies = list(n.proxy for n in agents)
        rel_surv_thresh = self.competition
        ordered = np.sort(proxies)
        survival_threshold = ordered[int(rel_surv_thresh*len(proxies))-1]
        # print(survival_threshold)
        potential_losers = list(losers for losers in agents if losers.proxy <= survival_threshold)
        potential_winners = list(losers for losers in agents if losers.proxy >= survival_threshold)
        for potential_loser in potential_losers:
            if np.random.rand() < self.selection_pressure:
                loser = potential_loser
                winner = random.choice(potential_winners)

                ''' offspring: here the loser becomes the offspring of the winner '''
                loser.effort = winner.effort
                loser.practice = np.random.normal(winner.practice,
                                                  self.practice_mutation_rate)
                loser.talent = np.random.normal(10, self.talent_sd)

                ''' practice within 360° '''
                if loser.practice > np.pi*2:
                    loser.practice = loser.practice - np.pi*2
    
                ''' no negative talent '''
                if loser.talent < 0:
                    loser.talent = 0.01
                loser.child_of = winner.unique_id
                
    def fitness_proportionate_selection(self):
        """ recompute rank (i.e. proxy values) with chosen effort levels
        randomly choose agent proportional to fitness (i.e. proxy-value)
        chosen agent reproduces, replacing a randomly drawn other agent
        2. draw new random talent,
        3. take the location & ID from the dead agent to facilitate display.
        Deaths, births and genealogy are stored in "Genealogy")"""
        agents = self.agents
        proxies = list(n.proxy for n in agents)
        rel_proxies = [p/sum(proxies) for p in proxies]
        inv_rel_proxies = [1-p for p in rel_proxies]
        inv_rel_proxies = [p/sum(inv_rel_proxies) for p in inv_rel_proxies]
        for potential_event in agents:
            if np.random.rand() < self.selection_pressure * self.competition:
                winner = np.random.choice(agents, p=rel_proxies)
                loser = np.random.choice(agents)#, p=inv_rel_proxies)
            
                ''' offspring: here the loser becomes the offspring of the winner '''
                loser.effort = winner.effort
                loser.practice = np.random.normal(winner.practice,
                                                  self.practice_mutation_rate)
                loser.talent = np.random.normal(10, self.talent_sd)

                ''' practice within 360° '''
                if loser.practice > np.pi*2:
                    loser.practice = loser.practice - np.pi*2
    
                ''' no negative talent '''
                if loser.talent < 0:
                    loser.talent = 0.01
                loser.child_of = winner.unique_id

    def step(self):
        ''' adjust effort levels in random order '''


        #print(len(self.agents),self.steps)
        self.datacollector.collect(self) #tried to make smart data collection but that would mean changing batchrunner
        self.agents.shuffle_do("step")
        self.kill_and_replace()
        self.steps += 1