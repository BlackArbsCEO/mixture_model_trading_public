# README #


### What is this repository for? ###

Tutorial demonstrating trading algorithm development from theory to implementation. (IN-PROGESS)

Focuses on using mixture models as a method for market timing. Types of mixtures to be covered include:

* gaussian mixtures (sklearn, pymc3)
* bayesian mixtures (sklearn, pymc3)
* gaussian processes (sklearn, pymc3) (maybe?)

#### There are some challenges to implementation ####

* I prefer to use the Quantopian platform for several reasons however there are some major restrictions on the packages that you are allowed to import (no pymc3) AND on the classes within certain approved packages (no sklearn.mix). 
* This means I can try to implement a crude(r) version of a backtest locally using either a custom solution or an already developed package (open to suggestions)
* Or I code the required algorithm(s) from scratch within the constraints of Quantopian's allowed packages/classes.
 
### Chapter (Notebook) Outlines ###

1. Motivation
	- this chapter goes through a demo of challenges to time series prediction using real market data
	- what is stationarity?
	- why one distribution isn't enough?
	- how can multiple distributions help?
2. Gaussian mixtures
	- in this chapter we explore the underlying intuition behind gaussian mixtures
	- we apply a gaussian mixture model to predict security return distribution
	- example in sklearn
	- demo strengths and weaknesses
3. Nonparametric mixture models
	1. bayesian mixtures using sklearn and pymc3
		- demo strengths and weaknesses
	3. gaussian processes using sklearn and pymc3
		- demo strengths and weaknesses
4. Designing the strategy
	1. hypothesis/theory
	2. initial testing
    3. backtesting 
5. Operationalizing 
	1. ??


### Who do I talk to? ###

* blackarbsceo | [blackarbs.com](www.blackarbs.com)
* bcr@blackarbs.com
