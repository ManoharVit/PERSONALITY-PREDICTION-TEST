# PERSONALITY-PREDICTION-TEST
Determining which of the five primary personality types the respondent most closely aligns with in order to categorize them. Taking a personality test can assist you in selecting a career path by providing you with greater insight into how you respond to a variety of scenarios. The process of categorization becomes more difficult to do as the number of predictive features available for analysis grows. In order to simplify the dataset and take into account its suggested feature columns, we make use of a technique called Principal Component Analysis (PCA), which is a dimension reduction method. Following this, a solution should be proposed that models and classifies the data using machine learning techniques such as Logistic Regression, Decision Tree Classifier, Neural network, KNN Classifier, Naive Bayes, and Discriminant Analysis. This solution should integrate these machine learning approaches in both PCA and dataset prediction characteristics. A model that has been adequately trained and generalized will have the ability to predict the class of participants with a level of accuracy that is acceptable. Additionally, evaluation metrics such asError Analysis, Lift Charts and ROC curves will be used to test the accuracy and measure of the models. The purpose of this research is to determine whether or not a phishing website can be recognized by its description and the essential characteristics it possesses.

## Problem Definition/specification
The assessment consists of fifty items that you must score on a five-point scale (like 1=Disagree, 3=Neutral, and 5=Agree) according to how accurate one is. Typically, it takes between 3 and 8 minutes to finish. Five personality characteristics such as extraversion, neuroticism, agreeableness, conscientiousness, and openness to experience respond to the majority of a person's personality question
A personality test can help you choose an occupation by giving you more insight into how you react in different situations. Career professionals and psychologists use this information in personality career tests for recruitment and candidate assessment. Accurate personality estimation is delicate and can be falsified to some extent by a seeker.
The dataset has a target column, which reflects the characteristic that received the highest possible score on the test, which is our response/target variable
A model that has been adequately trained and generalized will have the ability to predict the class of participants with a level of accuracy that is acceptable. 

## Dataset description
The personality test was developed using the International Personality Item Pool (IPIP)‘s "Big-Five Factor Markers.  Overall, dataset contains 964573 records and 113 features.
There are four distinct data types, which are referred to as int64(3), float32(100), float64(4), datetime64[ns](1), and object(4), and the count of each type is mentioned in brackets.
Each column has certain amount of null values, in which “Countries”, "lat_appx_lots_of_err" ,"long_appx_lots_of_err“ columns top.
The right-top shows the simple statistics and distributions of dataset. The order is shown in alphabetical order.

## Data preprocessing/exploration/visualization
The variables which are related to question and answers are highly correlated whereas the participant information variables are low correlated. 
 We changed the “dataload” column to datetime format and stored it in the “date_column” column. Converted the "lat_appx_lots_of_err" and "long_appx_lots_of_err" to numeric data type, “country” and “target” to string datatype. Dropped the first two columns which have no order and necessary for validation and so on similar procedure in other columns
There are tail’s (outliers) on the right probably of time columns due to errors in the calculation of the timeframe. So, we handled it by changing the milliseconds into second's format.
We replaced a few missing values in the “Countries” column using "lat_appx_lots_of_err" ,"long_appx_lots_of_err" and replaced null with country names. Rest of the null and duplicate values are dropped, which is 1.4% of dataset size.

## Machine learning models
Final data set is typically subset of main data set, which are answer columns featured selected using PCA and SelectKbest methods.
We have 50 attributes and 1 target attribute “Target”, which has class of 5 categories ‘O’, ’C’, ’E’, ’A’, ’N’. So, validation data has numerical predictors and categorical response
After that, we apply the following algorithms to the data in order to classify it: 
- Logistic Regression
- Decision Tree Classifier
- K-Neighbors Classifier
- Naive Bayes
- Neural network
- Discriminant Analysis 

## Performance comparison
Out of all “Oversampled Neural Network” model shown the best accuracy

## Recommendation/Conclusion
We have used the saved model and evaluated prediction test on new dataset with out target class and we have successfully predicted target class

## Tableau Dashboard
https://public.tableau.com/app/profile/giri.manohar/viz/Giri_v_16819304576880/Dashboard1
![image](https://github.com/ManoharVit/PERSONALITY-PREDICTION-TEST/assets/50493896/715e223a-ab2c-4123-bca4-472c9d021454)
