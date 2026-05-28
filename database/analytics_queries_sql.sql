SELECT Attrition_Flag, COUNT(*) AS customer_count
FROM raw_customers
GROUP BY Attrition_Flag;

SELECT Card_Category, Attrition_Flag, COUNT(*) AS customer_count
FROM raw_customers
GROUP BY Card_Category, Attrition_Flag
ORDER BY Card_Category;

SELECT Card_Category, Attrition_Flag, AVG(Contacts_Count_12_mon) AS avg_support_calls
FROM raw_customers
GROUP BY Card_Category, Attrition_Flag
ORDER BY Card_Category;

SELECT Card_Category, Attrition_Flag, AVG(Total_Trans_Ct) AS avg_transaction_count
FROM raw_customers
GROUP BY Card_Category, Attrition_Flag
ORDER BY Card_Category;

SELECT Income_Category, Attrition_Flag, COUNT(*) AS customer_count
FROM raw_customers
GROUP BY Income_Category, Attrition_Flag
ORDER BY Income_Category;

SELECT Attrition_Flag, AVG(Avg_Utilization_Ratio) AS avg_utilization
FROM raw_customers
GROUP BY Attrition_Flag;

SELECT Attrition_Flag, AVG(Total_Relationship_Count) AS avg_products_held
FROM raw_customers
GROUP BY Attrition_Flag;