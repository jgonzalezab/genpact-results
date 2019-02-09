# Genpact Machine Learning Hackathon
This repo contains all the code and solutions used in the [Genpact Machine Learning Hackathon](https://datahack.analyticsvidhya.com/contest/genpact-machine-learning-hackathon/) organized by Analytics Vidhya. It was a 48 hours hackathon based on food demand forecasting. This challenge was based on a real case, more formally the problem statement was the following:

*Your client is a meal delivery company which operates in multiple cities. They have various fulfillment centers in these cities for dispatching meal orders to their customers. The client wants you to help these centers with demand forecasting for upcoming weeks so that these centers will plan the stock of raw materials accordingly.*

*The replenishment of majority of raw materials is done on weekly basis and since the raw material is perishable, the procurement planning is of utmost importance. Secondly, staffing of the centers is also one area wherein accurate demand forecasts are really helpful. Given the following information, the task is to predict the demand for the next 10 weeks (Weeks: 146-155) for the center-meal combinations in the test set:*

* *Historical data of demand for a product-center combination (Weeks: 1 to 145)*
* *Product (Meal) features such as category, sub-category, current price and discount*
* *Information for fulfillment center like center area, city information etc*

As we can see the objective of this hackathon was to predict the demand of the different types of meat for the next 10 weeks  across the different centers.

The metric used to evaluate the results was the 100 * RSMLE:

<a href="https://www.codecogs.com/eqnedit.php?latex=RMSLE=&space;\frac{100}{n}&space;\sum_{i=1}^{n}&space;(log(\hat{y}&plus;1)&space;-&space;log(y&plus;1)))^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?RMSLE=&space;\frac{100}{n}&space;\sum_{i=1}^{n}&space;(log(\hat{y}&plus;1)&space;-&space;log(y&plus;1)))^2" title="RMSLE= \frac{100}{n} \sum_{i=1}^{n} (log(\hat{y}+1) - log(y+1)))^2" /></a>

Although the problem is based on a time series structure we approached in a ML way. The main steps of this approach were the following (they are simplified due to the complexity of the approach):
* Data engineering (cleaning, reordering, join of DFs...)
* Split training set by meal and train a ML model for each one
* Calculate reduced and augmented versions of these predictions
* Split training set by center and train a ML model for each one
* Calculate reduced and augmented versions of these predictions
* Ensemble all the predictions in an optimal way

All the details of this approach can be found on [final-approach.ipynb](https://github.com/jgonzalezab/genpact-results/blob/master/final-approach/final-approach.ipynb). We have also included other scripts with other approaches used in the hackathon but not submitted as final. With this approach we achivied a **53.79472 RSMLE (71/765) on the private leaderboard**. Unfortunately data is not freely available so the results here obtained can't be replicated without the pertinent permission.
