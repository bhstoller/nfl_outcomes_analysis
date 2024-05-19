# From College to the NFL: A Data-Driven Approach to Predicting Wide Receiver Success

## Introduction and Executive Summary
This project aimed to predict the success of wide receivers in the NFL based on their college statistics and draft performance. Utilizing data from two open-source APIs covering the years 2004-2023, I tested this hypothesis by developing decision tree and random forest models, which were calibrated through RFECV to optimize feature selection and SMOTE to address the present class imbalances. Ultimately, the SMOTE-enhanced random forest model achieved the highest accuracy, exhibiting a 66% success rate in predicting whether wide receivers would become low, average, or high performers. Notably, ESPN's annual pre-draft grade and wide receiver positional ranking emerged as the primary drivers behind the model's predictive accuracy. Despite this moderate predictive success, the random forest and other models included were subject to a few limitations, such as class imbalance and sample bias, which underscore the importance of addressing statistical challenges to maximize the effectiveness of predictive modeling in analytics.

## Data Collection and Processing
Using the CFBD and NFL-data-py open-source APIs, I collected comprehensive datasets of college football, NFL, and draft statistics from 2004-2023, where I focused on the wide receiver positon due to its comparitevely greater abundance of data. I then cleaned, standardized, and ultimately merged the data using a name-year identifier that I created for each athlete, since the two independent APIs did not share a common ID.

## Methodology
After plotting the correlation between the sum of athletes' z-scores across key football metrics (receiving yards, receptions, and touchdowns) and determining there was no significant linear or polynomial relationship, I employed K-Means clustering to instead group athletes into three performance classes: low, average, and high. To deploy a classification model, I used decision trees and random forests, which were optimized through Recursive Feature Elimination with Cross Validation (RFECV) and hyperparameter tuning. 
</p>
Given the combination of sample bias from the open-source APIs inability to capture the entire population of wide recievers during the sourced timeframe, along with the class imbalances stemming from the limited number of high-performing players, I applied Synthetic Minority Oversampling Technqiue (SMOTE) to enhance the data. By synthetically generating samples of the minority classes, SMOTE helped address these statistical limitations and improved the overall accuracy of the decision tree and random forest models.

## Results
The most successful model, the SMOTE-enhanced random forest, achieved an accuracy of 66%, with the two primary predictors being ESPN's pre-draft grade and positional ranking. Enhancing the data with SMOTE significantly improved the model's performance by addressing the class imbalances. Below are the classification reports of the random forests with and without the SMOTE enhancement.
</p>
<p align="center">
  <img src="Figures/figure_24.png" title="Random Forest Classification Report Comparison">
</p>
As shown, the SMOTE-enhanced model exhibits improved precision, recall, and F1-scores across each class, particularly for the high-performers where the base random forest failed to correctly classify any athletes in this category. 
</p>
Regarding the key drivers, the following figure anlyzes the model's top five features by their Mean Decrease in Impurity (MDI) and Mean Decrease in Accuracy (MDA).
</p>
<p align="center">
  <img src="Figures/figure_23.png" title="SMOTE Random Forest: Top Five Features by MDI and MDA Importance">
</p>
As depicted in the chart, ESPN's pre-draft grade and positional ranking emerged as the features with the highest combined MDI and MDA in my model, suggesting their intriguingly accurate indications of wide receiver success in the NFL.

## Discussion
While the results of the models I developed indicate that certain college stats and draft information are in fact predictors of NFL success, they are subject to statistical caveats including limited generalizability and assumptions introduced from using SMOTE. Given that the initial data does not capture the entire population and also filters out any college athletes who did not perform well enough to be drafted, the model's insights and predictions may not fully represent the performance potential of all wide receivers.
</p>
Additionally, due to the severe class imbalance, my use of SMOTE introduced assumptions of independence among the synthetic samples and original data. This could lead to overfitting, which might not accurately capture the true variability of the minority class and could thereby bias the model's performance.

## Conclusion and Future Work
This project estalishes a comprehensive baseline for predicting NFL success from college and draft statistics. Future improvements could involve utilizing a larger, more representative sample size or enhancing the model's sophistication to incorporate complex features such as mental stamina. Together, these enhancements would help mitigate the existing statistical caveats and integrate more qualitative relationships into the conclusions, thereby further improving the model's predictive accuracy.

## Learn More and Connect
Explore the full report [here](full_project_report.md), which expands on the above analysis and provides a more detailed documentation my decisions.
</p>
Please contact me at bhs.stoller@gmail.com for more information.
