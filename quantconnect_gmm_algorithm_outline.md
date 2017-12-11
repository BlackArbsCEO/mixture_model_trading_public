Quantconnect Mixture Model Algorithm Outline

Resolution: Daily

**schedule tasks to better control the flow of information through the algorithm**

1. Check open orders:
	* confirm all orders are filled
	* track fill dates and construct liquidation dates

2. Check inventory:
	* check current assets held, get quantity and total value

3. Check if any current holdings meet liquidation criteria:
	* check if today's date is greater than or equal to liquidation date

4. Compute fixed trade size:
	* compute trade size in shares as percentage of portfolio value
	* by default I set to 5%

5. Run main algorithm:
	* lookback period: 252 days or 1 trading year
	* compute optimal number of components over lookback period using `bic`
     	* fit gmm using components
     	* extract hidden states, parameters
     	* sample from distr using those parameters
	* log confidence interval boundaries for each symbol
	* compare boundaries with current return to identify outliers
	* assess direction of outliers e.g. `too_low` or `too_high`
	* assign securities to long or short list based on direction of outliers.

6. Use main algorithm results to send orders:
	* use long and/or short list to send orders
	* this implementation attempts to use `MarketOnOpenOrders`
 
