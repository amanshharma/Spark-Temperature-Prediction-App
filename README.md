# Spark Temperature prediction application

For this project I used a dataset from UCI Machine Learning Repository. 

Dataset and relevant details : http://archive.ics.uci.edu/ml/datasets/Air+Quality

## Goal: 
To write a Spark App to predict "Temperature" - T field in the Dataset, 24hrs ahead. 

**To build**:

To run (Set spark to path variables to run anywhere in cmd or move to installation folder, in my case: C:\enterprise\spark-1.6.2-bin-hadoop2.6\bin): 

Before running please put the dataset files at C:\enterprise\spark-1.6.2-bin-hadoop2.6\bin so that spark can access while running. I have attached the dataset files in the project.

Use command mvn package to build maven jar.
Jar will be stored in 'Target' folder after completion.
Location in my case is C:\Users\Aman\Enterprise630\Workspace\temperature-prediction\target\temperature-prediction-0.0.1-SNAPSHOT.jar

**To run**

Use the below command on cmd to run:
spark-submit --class "app.SimpleApp" --master local[4] C:\Users\Aman\Enterprise630\Workspace\temperature-prediction\target\temperature-prediction-0.0.1-SNAPSHOT.jar


My spark app reads raw data from **AirQualityUCI.csv**. Then it makes data cleaning and writes clean data and labels to **AirQualityUCI_WithLabels.csv** Label is the temperature 24 hours late.

**AirQualityUCI_WithLabels.csv** does not include the rows form original file that has 200 in the T column since it looks like as a default value for missing data. Also it does not include the row that could have 200 as a label. All commas in the numbers are replaced by dots since then we will use Double.parseDouble(some_string) and it does not work with strings that contain commas.

-	I decided to use **LinearRegressionWithSGD** model since we have a regression problem. I did not do feature normalization, but I tried some different step size in range 0.01 - 0.0000001 and got the best result using 0.001. I also tried different number of iterations and some different sets of features.


-	To train my algorithm I used 70% of data. Then I tested on my testing set (30% of data). I used MSE as my metric.
I got the best result MSE = 8.0 on testing set (since I used random split the MSE is little bit different every run - about 8.0-8.6).


-	I used only two features - temperature and difference of temperatures for previous 24 hours. For example if we want to predict temperature on May, 8th, 6.00 am, we use temperature on May, 7th, 6.00 am and difference between the temperatures on May, 7th, 6.00 am and May, 6th, 6.00 am.


-	There is a lot of work that can be done to improve the result. For example, we can try to use the difference in Humidity - to create a new column as I did for the difference in temperature. 

Also, it could be useful to normalize all columns. In this case, it could be easy to try a lot of different feature combinations without changing step size (I learned that step size is very related with range of features values).

