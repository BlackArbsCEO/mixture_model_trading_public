Quantconnect Mixture Model Algorithm Outline

Resolution: Daily

**schedule tasks to better control the flow of information through the algorithm**

1. Check open orders:
	* confirm all orders are filled
	* track fill dates and construct liquidation dates

2. Check if any current holdings meet liquidation criteria:
	* check if today's date is greater than or equal to liquidation date
	* if so liquidate position

3. Run main algorithm:
	* lookback period: `252 days` or approximately `1 trading year`
     	* fit gmm using `N` components
     	* extract hidden states, parameters
     	* sample from chosen distribution using predicted state parameters
	* compute estimate of confidence interval boundaries for each symbol
	* compare boundaries with current return to identify outliers
	* assess direction of outliers e.g. `too_low` or `too_high`
	* assign securities to long or short list based on direction of outliers.

4. Use main algorithm results to send orders:
	* use long and/or short list to send orders
	* this implementation uses `MarketOnOpenOrders`.
 
