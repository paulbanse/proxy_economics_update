# Proxyeconomics_original
Proxyeconomics Agent based model

The model is written in Python3.7 using the 'Mesa0.8.6' package. 
I used 'Anaconda' environment manager and 'spyder3.3.6' editor.
All simulations were run on a standard PC (20GB RAM).

To install Mesa:
in the Anaonda Prompt run: pip install Mesa
(note that this is case-sensitive)

You will also need the following packages, which can be installed in a similar way or using the anaconda package manager:
'matplotlib, numpy, pandas, scipy, random'

The model code is in: ProxyModel1.py

It contains two main classes (ProxyAgent, ProxyModel), defining the behavior of the agents and overall model, respectively.
Additionally it contains a number of data collector functions to compute model means used by the mesa batch_runner.

To run a family of models load and run eg.: S6_run_ProxyModel_competition.py.
Within you can set the parameters of interest:

The total number of modelsteps can be set under finalStep (line 36).
Each run_ProxyModel_....py file is made to display a family of models over one variable parameter (e.g. competition, line 37),
where the other parameters can be set within lines 38 to 47
The number of model repeats for each individual parameter constellation can be set under iterations (line50).

Default parameters in run_ProxyModel_competition are set to reproduce Fig.3,4.
Fig. 5 was produced by running S6_run_ProxyModel_competition.py or S7_run_ProxyModel_goal_angle.py with the indicated parameters.
Fig. 6 was produced by increasing the finalStep to 10000 and the data_collect_interval (line38) to 10. 
Fig. 7 was produced by running families with other variable parameters (e.g. S11_run_ProxyModel_selection_pressure.py). For each parameter, step size was increased until it was clear that equilibrium had been reached.
Fig. 8 was produced by modifying the get_prospect() function in the ProxyAgent class in ProxyModel1.py to ''' Step Prospect ''' (A-C).
  or by setting angle_agency (line 149,154 in run_ProxyModel_....py) to 1 (D-F).
Fig. S1 was produced by running S6_run_ProxyModel_competition.py with selection pressure (line 45) set to 0
Fig. S2 was produced by activating (uncommenting) self.fitness_proportionate_selection() in S5_ProxyModel1 (line 325) and deactivating (commenting) line 324
Fig. S3 was produced by returning own_proxy-survival_threshold as prospect (line 155) in addition the changes for S2
Fig. S4 was produced by running code as in S3 but for 1000 time steps and p=0.9


# proxy_economics_update
# proxy_economics_update
