# Recommender System

### Stocks.csv:<br>
#### Features
* currency_code: **please, ignore**<br>
* country_code: **country where the sale is being made (the dataset is from a multinational company)**<br>
* month_code: **dates referred as "1900" are open orders, no sales date yet**
* inventory_cost: **represent production costs**
* volume_primary_units: **decimal numbers are correct since some products are sold in liters**

#### Other information
1. If sales “inventory_cost”, “volume_primary_units” and “invoiced_sales” are zero there is no information available 
yet about the orders. <br>
No information available yet about these orders, so they can be eliminated (off invoice adjustments) <br><br>
2. Negative invoiced_sales:<br><br>
2.1. Case 1: negative **invoiced_sales** with positive **volume_primary_units** <br> 
Comercial agreement with the client leading to a compensation due to delay or stock rupture.<br><br>
2.2. Case 2: negative **invoiced_sales** with negative **volume_primary_units** <br>
Product and credit return. <br><br>
2.3. Case 3: negative **invoiced_sales** with zero **volume_primary_units** <br>
Credit return without product return.<br><br>
3. inventory_costs<br>
inventory_costs are multiplied by the primary units so they are also affected by the decimals 
volume_primary_units<br><br>

