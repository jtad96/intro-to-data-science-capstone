import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import numpy as np
from matplotlib.ticker import PercentFormatter
import seaborn as sns
from scipy.stats import mannwhitneyu, kstest, ttest_ind
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import KNNImputer
from math import sqrt


def score_model(model, X_train, X_test, y_train, y_test):
    y_train_hat = model.predict(X_train)
    y_test_hat = model.predict(X_test)

    #r2_score
    train_r2 = r2_score(y_train, y_train_hat)
    test_r2 = r2_score(y_test, y_test_hat)

    #mse
    train_rmse = sqrt(mean_squared_error(y_train, y_train_hat))
    test_rmse = sqrt(mean_squared_error(y_test, y_test_hat))

    out = {
        "Training RMSE": train_rmse,
        "Training R2": train_r2,
        "Testing RMSE": test_rmse,
        "Testing R2": test_r2
    }

    return out


def alpha_search(alg, alphas):
    rmses = []
    r_squareds = []

    for alpha in alphas:
        # fit model
        model = alg(alpha=alpha, random_state=seed)
        model.fit(X_train_scaled, y_train)

        # Assess model
        scores = score_model(model, X_train_scaled, X_test_scaled, y_train, y_test)
        rmses.append(scores["Training RMSE"])
        r_squareds.append(scores["Training R2"])
    
    # Get minimum mean squared error
    minimum_mse = min(rmses)
    minimum_mse_index = rmses.index(minimum_mse)  # index of minimum mean squared error
    best_alpha = alphas[minimum_mse_index]

    # Return output as dictionary
    return {
        "alphas": alphas,
        "training_rmses": rmses,
        "training_r2s": r_squareds,
        "best_alpha": best_alpha,
        "minimum_rmse": minimum_mse
    }


# read and write locations
data_folder = os.path.join("..", "data")
raw_folder = os.path.join(data_folder, "raw")
processed_folder = os.path.join(data_folder, "processed")

# file names
rmp_num_filename = "rmpCapstoneNum.csv"
rmp_qual_filename = "rmpCapstoneQual.csv"
rmp_tags_filename = "rmpCapstoneTags.csv"

df_num = pd.read_csv(os.path.join(raw_folder, rmp_num_filename), header=None)
df_qual = pd.read_csv(os.path.join(raw_folder, rmp_qual_filename), header=None)
df_tags = pd.read_csv(os.path.join(raw_folder, rmp_tags_filename), header=None)

df_num_column_names = [
     "average_rating",
     "average_difficulty",
     "number_of_ratings",
     "received_a_pepper",
     "would_take_again",
     "number_of_ratings_online",
     "male_gender",
     "female_gender",
]


df_qual_column_names = [
     "major",
     "university",
     "state",
]

df_tags_column_names = [
    "tough_grader",
    "good_feedback",
    "respected",
    "lots_to_read",
    "participation_matters",
    "dont_skip_class_or_you_will_not_pass",
    "lots_of_homework",
    "inspirational",
    "pop_quizzes",
    "accessible",
    "so_many_papers",
    "clear_grading",
    "hilarious",
    "test_heavy",
    "graded_by_few_things",
    "amazing_lectures",
    "caring",
    "extra_credit",
    "group_projects",
    "lecture_heavy"
]

df_num.columns = df_num_column_names
df_qual.columns = df_qual_column_names
df_tags.columns = df_tags_column_names

# join together
df = df_num.join(df_qual).join(df_tags)

# drop na values based on average_rating column
df = df.dropna(subset="average_rating")

# filter out the professors with less than 5 ratings
df_filtered = df[df["number_of_ratings"] >= 5].copy()

# calculate total # of tags for each professor
total_tag_counts = df_filtered[df_tags_column_names].sum(axis = 1)

# normalize tag columns to be between 0 and 1 by dividing tag counts by total tag counts
for col in df_tags_column_names:
    df_filtered[col] = df_filtered[col] / total_tag_counts

# fill na values with 0 that resulted from divide by zero issues
df_filtered[df_tags_column_names] = df_filtered[df_tags_column_names].fillna(0)

# Write to csv
df_filtered.to_csv("../data/processed/filtered_records.csv", index=False)

################ Questions 7 and 8 #####################
df = pd.read_csv("../data/processed/filtered_records.csv")
seed = 12250697


# Question 7
# Define training and testing sets
y = df["average_rating"]
X = df[[
    "average_difficulty",
    "number_of_ratings",
    "received_a_pepper",
    "would_take_again",
    "number_of_ratings_online",
    "male_gender",
    "female_gender",
]]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

# Scale data - fit on training data, transform both training and testing data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Impute data using k nearest neighbor imputation
imputer = KNNImputer(n_neighbors=10)
X_train_scaled = imputer.fit_transform(X_train_scaled)
X_test_scaled = imputer.transform(X_test_scaled)

# Correlation matrix
corr = X_train.corr()

# Plot heatmap
plt.figure(figsize=(10, 5)) # Adjust the figure size as needed
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Numerical Features')
plt.show()

# simple linear regression
model = LinearRegression().fit(X_train_scaled, y_train)
simple_linear_model_scores = score_model(model, X_train_scaled, X_test_scaled, y_train, y_test)
print(simple_linear_model_scores)

simple_linear_model_coefficients = model.coef_
print(simple_linear_model_coefficients)
plt.bar(X.columns, simple_linear_model_coefficients)
plt.xlabel("Predictor")
plt.ylabel("Coefficient value")
plt.xticks(rotation=45)
plt.title("Standard Linear Regression Model Weights")
plt.show()

# Ridge regression
alphas = np.arange(0, 1000, 1)
out = alpha_search(Ridge, alphas)

plt.plot(out["alphas"], out["training_rmses"])
plt.xlabel("Regularization Parameter")
plt.ylabel("Training Root Mean Squared Error")
plt.title("Ridge Regression Varying Regularization Parameter")
plt.show()

# Lasso regression
alphas = np.arange(0.00, 1, 0.01)
out = alpha_search(Lasso, alphas)

plt.plot(alphas, out["training_rmses"])
plt.xlabel("Regularization Parameter")
plt.ylabel("Training Root Mean Squared Error")
plt.title("Lasso Regression Varying Regularization Parameter")
plt.show()


# Question 8

df_tags_column_names = [
    "tough_grader",
    "good_feedback",
    "respected",
    "lots_to_read",
    "participation_matters",
    "dont_skip_class_or_you_will_not_pass",
    "lots_of_homework",
    "inspirational",
    "pop_quizzes",
    "accessible",
    "so_many_papers",
    "clear_grading",
    "hilarious",
    "test_heavy",
    "graded_by_few_things",
    "amazing_lectures",
    "caring",
    "extra_credit",
    "group_projects",
    "lecture_heavy"
]

# Define training and testing sets
y = df["average_rating"]
X = df[df_tags_column_names]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

# Scale data - fit on training data, transform both training and testing data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Correlation matrix
corr = X_train.corr()

# 3. Plot the heatmap using Seaborn
plt.figure(figsize=(20, 5)) # Adjust the figure size as needed
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Tag Features')
plt.show()

# simple linear regression
model = LinearRegression().fit(X_train_scaled, y_train)
simple_linear_model_scores = score_model(model, X_train_scaled, X_test_scaled, y_train, y_test)
print(simple_linear_model_scores)

simple_linear_model_coefficients = model.coef_
print(simple_linear_model_coefficients)
plt.figure(figsize=(15, 6))
plt.bar(X.columns, simple_linear_model_coefficients)
plt.xlabel("Predictors")
plt.ylabel("Coefficient value")
plt.xticks(rotation=90)
plt.title("Standard Linear Regression Model Weights")
plt.show()

# Ridge regression
alphas = np.arange(0, 1000, 1)
out = alpha_search(Ridge, alphas)

plt.plot(out["alphas"], out["training_rmses"])
plt.xlabel("Regularization Parameter")
plt.ylabel("Training Root Mean Squared Error")
plt.title("Ridge Regression Varying Regularization Parameter")
plt.show()

# Lasso regression
alphas = np.arange(0.00, 1, 0.01)
out = alpha_search(Lasso, alphas)

plt.plot(alphas, out["training_rmses"])
plt.xlabel("Regularization Parameter")
plt.ylabel("Training Root Mean Squared Error")
plt.title("Lasso Regression Varying Regularization Parameter")
plt.show()






########### Questions 1-5, 9, and 10 ##################
# libraries 
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# statistical tests
from scipy import stats
from cliffs_delta import cliffs_delta as cd

# machine learning 
# train test split
from sklearn.model_selection import KFold, train_test_split, cross_val_score, cross_validate, GridSearchCV
# preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer

# evaluation
from sklearn.metrics import r2_score, root_mean_squared_error, mean_squared_error
# models 
from sklearn.linear_model import LinearRegression, LogisticRegression, LogisticRegressionCV, RidgeCV, LassoCV, ElasticNetCV



# reproducibility 
import random
# specify seed 
n_number = 12250697
random.seed(n_number)
np.random.seed(n_number)

# useful function
# simple significance test interpretation
def significance(alpha, p_value):
    """
    Decide based on alpha. 
    Note: p is the probability, under H0, of observing a statistic at least this extreme.

    """
    print(f"p-value: {p_value:.6g}")
    if p_value < alpha:
        print(f"p = {p_value:.6g} < α = {alpha} = Reject H0 (statistically significant).")
    else:
        print(f"p = {p_value:.6g} ≥ α = {alpha} = Fail to reject H0 (not statistically significant).")


# read and write locations
data_folder = os.path.join("..", "data")
raw_folder = os.path.join(data_folder, "raw")
processed_folder = os.path.join(data_folder, "processed")

# file names
rmp_num_filename = "rmpCapstoneNum.csv"
rmp_qual_filename = "rmpCapstoneQual.csv"
rmp_tags_filename = "rmpCapstoneTags.csv"


# import the data 
df_num = pd.read_csv(os.path.join(raw_folder, rmp_num_filename), header=None)
df_qual = pd.read_csv(os.path.join(raw_folder, rmp_qual_filename), header=None)
df_tags = pd.read_csv(os.path.join(raw_folder, rmp_tags_filename), header=None)

# headers for first dataset 
df_num_column_names = [
     "average_rating",
     "average_difficulty",
     "number_of_ratings",
     "received_a_pepper",
     "would_take_again",
     "number_of_ratings_online",
     "male_gender",
     "female_gender",
]

# headers for second dataset
df_qual_column_names = [
     "major",
     "university",
     "state",
]

# headers for third dataset 
df_tags_column_names = [
    "tough_grader",
    "good_feedback",
    "respected",
    "lots_to_read",
    "participation_matters",
    "dont_skip_class_or_you_will_not_pass",
    "lots_of_homework",
    "inspirational",
    "pop_quizzes",
    "accessible",
    "so_many_papers",
    "clear_grading",
    "hilarious",
    "test_heavy",
    "graded_by_few_things",
    "amazing_lectures",
    "caring",
    "extra_credit",
    "group_projects",
    "lecture_heavy"
]

# assign headers to dataframes using .columns attribute 
df_num.columns = df_num_column_names
df_qual.columns = df_qual_column_names
df_tags.columns = df_tags_column_names

# join the data for the easier manipulation 
df = df_num.join(df_qual).join(df_tags)

# drop null values from the dataset 
df = df.dropna(subset="average_rating")
# check the results
df.isna().sum()

# distribution of number of ratings 
df["number_of_ratings"].describe()

# plot for distribution of number of ratings 
plt.figure(figsize = (12, 10))

# histogram 
plt.hist(df["number_of_ratings"], 
         bins = 100, 
         alpha = 0.7)

# show median and mean value on the plot 
# median
plt.axvline(df["number_of_ratings"].median(), 
           color = "green", 
           linestyle = "dashed", 
           linewidth = 2, 
           label = f"median: {df['number_of_ratings'].median():.1f}")

# mean
plt.axvline(df["number_of_ratings"].mean(), 
           color = "red", 
           linestyle = "dashed", 
           linewidth = 2, 
           label= f"mean: {df['number_of_ratings'].mean():.1f}")

# aesthetics 
# title
plt.title("Distribution of number of ratings", 
          fontweight = "bold", 
          fontsize = 18)

# x axis 
plt.xlabel("Number of ratings", 
           fontsize = 16)
# limit
plt.xlim(0, 50)

# y axis
plt.ylabel("Frequency", 
           fontsize = 16)
# size for both x and y ticks 
plt.tick_params(axis = "both", 
                labelsize = 14)

# legend 
plt.legend(fontsize = 16)

# show the plot
plt.show()

# filter out the professors with less than 5 ratings
df_filtered = df[df["number_of_ratings"] >= 5].copy()

# calculate total # of tags for each professor
total_tag_counts = df_filtered[df_tags_column_names].sum(axis = 1)

# normalize tag columns to be between 0 and 1 by dividing tag counts by total tag counts
for col in df_tags_column_names:
    df_filtered[col] = df_filtered[col] / total_tag_counts

# fill na values with 0 that resulted from divide by zero issues
df_filtered[df_tags_column_names] = df_filtered[df_tags_column_names].fillna(0)

# check male and female gender error, where both are 1, or both 0
both_11 = df_filtered[(df_filtered["male_gender"] == 1) & (df_filtered["female_gender"] == 1)].shape[0]
both_00 = df_filtered[(df_filtered["male_gender"] == 0) & (df_filtered["female_gender"] == 0)].shape[0]

# print the results 
print(f"Number of professors with both male and female gender as 1: {both_11}")
print(f"Number of professors with both male and female gender as 0: {both_00}")

# drop rows where both male and female gender are 1 or 0
df_filtered_final = df_filtered[~((df_filtered["male_gender"] == 1) & (df_filtered["female_gender"] == 1)) & 
                          (~((df_filtered["male_gender"] == 0) & (df_filtered["female_gender"] == 0)))].copy()

# check the results
df_filtered_final.info()

# Question 1
# separate the average ratings into two groups based on gender
# Male = 1, Male = 0
df1_male = df_filtered_final[df_filtered_final["male_gender"] == 1]
df1_female = df_filtered_final[df_filtered_final["male_gender"] == 0]

# count number of professors in each group
print(f"Number of male professors: {df1_male.shape[0]}")
print(f"Number of female professors: {df1_female.shape[0]}")

# compare distributions of average rating between male and female professors
plt.figure(figsize=(12, 10))

# box plot for male and female professors 
sns.boxplot(data=df_filtered_final, 
            x='average_rating', 
            y='male_gender', 
            palette = "Set2", 
            showfliers = False, 
            medianprops = {"color": "red", "linewidth": 2}, 
            orient= "horizontal")

# aesthetics 
plt.title('Average Rating Distribution by Gender', 
          fontweight = "bold",
          fontsize = 18)

# xlabel
plt.xlabel('Average Rating', 
           fontsize = 16)
# ylabel
plt.ylabel('Gender (1 = Male, 0 = Female)', 
           fontsize = 16)

# legend
plt.legend(['0 - Female', 
            '1 - Male'], 
            fontsize = 16,
           loc = "upper left")
# ticks 
plt.tick_params(axis = "both", 
                labelsize = 14)

plt.show()

# conduct mann-whitney u test for average rating using our function
u_stat, p_value = stats.mannwhitneyu(df1_male['average_rating'], 
                                     df1_female['average_rating'], 
                                     alternative='greater')
# interpretation 
significance(0.005, p_value)

# Question 2
# again_separate the average ratings into two groups based on gender
# Male = 1, Male = 0
df2_male = df_filtered_final[df_filtered_final["male_gender"] == 1]
df2_female = df_filtered_final[df_filtered_final["male_gender"] == 0]


# count number of professors in each group
print(f"Number of male professors: {df2_male.shape[0]}")
print(f"Number of female professors: {df2_female.shape[0]}")

# plot the variance of average rating dipersion between male and female professors
# kernel density plot 
plt.figure(figsize=(12, 10))

# plot
sns.kdeplot(df2_male['average_rating'], 
           label='Male', 
           color='blue', 
           fill=True, 
           legend= True, 
           alpha=0.15
           )
sns.kdeplot(df2_female['average_rating'],
              label='Female', 
              color='green', 
              fill=False, 
              legend= True)

# aesthetics
plt.title('Average Rating Dispersion by Gender', 
          fontweight = "bold",
          fontsize = 18)
# xlabel
plt.xlabel('Average Rating',
              fontsize = 16)
# ylabel
plt.ylabel('Density',
                fontsize = 16)

# axis 
plt.tick_params(axis = "both", 
                labelsize = 14)
# legend 
plt.legend()
# ticks
plt.show()

# dispersion of average ratings 
# levene's test for equal variances
levene_stat, levene_p_value = stats.levene(df2_male['average_rating'], 
                                      df2_female['average_rating'], 
                                      center='median')

# check the result 
significance(0.005, levene_p_value)

# Question 3
 # divide average ratings by groups of gender and convert to arrays (bootstrap expects that)
df3_male = df_filtered_final[df_filtered_final["male_gender"] == 1]
df3_female = df_filtered_final[df_filtered_final["male_gender"] == 0]

# define function for effect size 
def cliffs_delta_trap(sample1, sample2, axis = 0):
    return cd(sample1, sample2)[0]

# Perform the bootstrap
result = stats.bootstrap(
    (df3_male['average_rating'].to_numpy(), 
    df3_female['average_rating'].to_numpy()), 
    statistic=cliffs_delta_trap,
    confidence_level=0.95, 
    random_state= n_number, 
    vectorized= False,
    method='BCa', 
    n_resamples=10000)

# compute the confidence interval
ci_lower, ci_upper = result.confidence_interval.low, result.confidence_interval.high

# print the results 
# point estimate
print(f"Effect size (Cliffs delta): {cd(df3_male['average_rating'], 
                                        df3_female['average_rating'])[0]:.4f}")

# confidence interval
print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

# plot boostrap distribution of cliffs delta
# figure size 
plt.figure(figsize=(12, 10))

# plot
plt.hist(result.bootstrap_distribution, 
         bins=40, 
         density=True, 
         color = "skyblue",
         edgecolor = "black",
         fill = True)

# point estimate line
plt.axvline(cd(df3_male['average_rating'].to_numpy(), 
                df3_female['average_rating'].to_numpy())[0], 
            color = "orange", 
            linestyle = "dashed", 
            label = f"Effect size: {cd(df3_male['average_rating'].to_numpy(), 
                                            df3_female['average_rating'].to_numpy())[0]:.4f}"
            )

# confidence interval lines
# lower
plt.axvline(ci_lower, 
            color = "red", 
            linestyle = "dashed", 
            label = f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]"
            )
# upper
plt.axvline(ci_upper, 
            color = "red", 
            linestyle = "dashed")

# aesthetics
# title 
plt.title("Bootstrap Distribution of Cliff's Delta for Average Rating by Gender", 
          fontweight = "bold",
          fontsize = 18)

# x and y axis labels
plt.xlabel("Cliff's Delta", 
           fontsize = 16)
plt.ylabel("Density", 
           fontsize = 16)

# ticks 
plt.tick_params(axis = "both", 
                labelsize = 14)

# legend 
plt.legend(fontsize = 16)

# show the plot 
plt.show()

# variance difference in average rating for male and female professors
def variance_diff(sample1, sample2, axis = 0):
    # compute variance for each sample
    variance1 = np.var(sample1, ddof = 1)
    variance2 = np.var(sample2, ddof = 1)
    return variance1 - variance2   # male minus female

# compute the bootstrap
result_variance = stats.bootstrap(
    data=(df3_male['average_rating'].to_numpy(), 
    df3_female['average_rating'].to_numpy()),
    statistic=variance_diff,
    confidence_level=0.95,
    n_resamples=10000,
    method='BCa',
    random_state=n_number,
    vectorized=False
)

# compute the confidence interval
ci_lower_variance, ci_upper_variance = result_variance.confidence_interval.low, result_variance.confidence_interval.high

# print the results 
# point estimate
print(f"Effect size (variance difference): {variance_diff(df3_male['average_rating'], df3_female['average_rating'])}")
# confidence interval
print(f"95% CI: [{ci_lower_variance:.4f}, {ci_upper_variance:.4f}]")

# plot boostrap distribution of variance difference
# figure size 
plt.figure(figsize=(12, 10))

# plot
plt.hist(result_variance.bootstrap_distribution, 
         bins=40, 
         density=True, 
         color = "orange",
         edgecolor = "black",
         fill = True)

# point estimate line
plt.axvline(variance_diff(df3_male['average_rating'], 
                df3_female['average_rating']), 
            color = "blue", 
            linestyle = "dashed", 
            label = f"Effect size: {variance_diff(df3_male['average_rating'], df3_female['average_rating']):.4f}"
            )

# confidence interval lines
# lower
plt.axvline(ci_lower_variance, 
            color = "red", 
            linestyle = "dashed", 
            label = f"95% CI: [{ci_lower_variance:.4f}, {ci_upper_variance:.4f}]"
            )
# upper
plt.axvline(ci_upper_variance, 
            color = "red", 
            linestyle = "dashed")

# aesthetics
# title 
plt.title("Bootstrap Distribution of Variance Difference for Average Rating by Gender", 
          fontweight = "bold",
          fontsize = 18)

# x and y axis labels
plt.xlabel("Variance Difference", 
           fontsize = 16)
plt.ylabel("Density", 
           fontsize = 16)

# ticks 
plt.tick_params(axis = "both", 
                labelsize = 14)

# legend 
plt.legend(fontsize = 16)

# show the plot 
plt.show()

# show the plot 
plt.show()

# Question 4
# filter our the data for this question 
df4_male = df_filtered_final[df_filtered_final["male_gender"] == 1]
df4_female = df_filtered_final[df_filtered_final["male_gender"] == 0]

# filter mask for all tags columns
tag_columns = df_tags_column_names
alpha = 0.005

# list for all tags 
results_q4 = []

# for loop for separating each movie subset into two groups and conducting the non-parametric test
for tag in tag_columns: 
    # separating into two groups
    males = df4_male[tag]
    females = df4_female[tag]

    # conducting the test 
    u_statistic, p_value = stats.mannwhitneyu(males, females, alternative= 'two-sided')
    
    # save data as well
    results_q4.append({
    "tag": tag,
    "n_female": len(females),
    "n_male": len(males),
    "U": u_statistic,
    "p_value": p_value,
    "significant": p_value < alpha
    })
        
    # # use the significance function created 
    # print(f"{tag} Result:")
    # print(f"{significance(0.005, p_value)}\n")

# check the results
results_q4_df = pd.DataFrame(results_q4).sort_values(by = 'p_value', ascending= False)
styled = results_q4_df.style.set_properties(**{
    "background-color": "white",
    "color": "black", 
     "border": "1px solid black"
}).format({"p_value": "{:.6e}"})

# sort by p-value
styled

# Question 4
# separate into two groups based on gender
# Male = 1, Male = 0
df5_male = df_filtered_final[df_filtered_final["male_gender"] == 1]
df5_female = df_filtered_final[df_filtered_final["male_gender"] == 0]


# check quantity 
# it is the same number of rows, so this code is just in case
print(f"Number of male professors: {df5_male.shape[0]}")
print(f"Number of female professors: {df5_female.shape[0]}")

# compare distributions of average rating between male and female professors
plt.figure(figsize=(12, 10))

# box plot for male and female professors 
sns.boxplot(data=df_filtered_final, 
            x='average_difficulty', 
            y='male_gender', 
            palette = "Set3", 
            showfliers = False, 
            medianprops = {"color": "red", "linewidth": 2}, 
            orient= "horizontal")

# aesthetics 
plt.title('Average Difficulty Rating Distribution by Gender', 
          fontweight = "bold",
          fontsize = 18)

# xlabel
plt.xlabel('Average Difficulty Rating', 
           fontsize = 16)
# ylabel
plt.ylabel('Gender (1 = Male, 0 = Female)', 
           fontsize = 16)

# legend
plt.legend(['0 - Female', 
            '1 - Male'], 
            fontsize = 16,
           loc = "upper left")
# ticks 
plt.tick_params(axis = "both", 
                labelsize = 14)

plt.show()

# conduct mann-whitney u test for average rating using our function
u_stat, p_value = stats.mannwhitneyu(df1_male['average_difficulty'], 
                                     df1_female['average_difficulty'], 
                                     alternative='two-sided')
# interpretation 
significance(0.005, p_value)

# Question 6
 # divide average difficulty ratings by groups of gender and convert to arrays (bootstrap expects that)
df6_male = df_filtered_final[df_filtered_final["male_gender"] == 1]
df6_female = df_filtered_final[df_filtered_final["male_gender"] == 0]

# define function for effect size 
def cliffs_delta_trap(sample1, sample2, axis = 0):
    return cd(sample1, sample2)[0]

# Perform the bootstrap
result_q6 = stats.bootstrap(
    (df6_male['average_difficulty'].to_numpy(), 
    df6_female['average_difficulty'].to_numpy()), 
    statistic=cliffs_delta_trap,
    confidence_level=0.95, 
    random_state= n_number, 
    vectorized= False,
    method='BCa', 
    n_resamples=10000)

# compute the confidence interval
ci_lower_q6, ci_upper_q6 = result_q6.confidence_interval.low, result_q6.confidence_interval.high

# print the results 
# point estimate
print(f"Effect size (Cliffs delta): {cd(df6_male['average_difficulty'], 
                                        df6_female['average_difficulty'])[0]:.4f}")

# confidence interval
print(f"95% CI: [{ci_lower_q6:.4f}, {ci_upper_q6:.4f}]")

# plot boostrap distribution of cliffs delta
# figure size 
plt.figure(figsize=(12, 10))

# plot
plt.hist(result_q6.bootstrap_distribution, 
         bins=40, 
         density=True, 
         color = "darkgreen",
         edgecolor = "black",
         fill = True)

# point estimate line
plt.axvline(cd(df6_male['average_difficulty'].to_numpy(), 
                df6_female['average_difficulty'].to_numpy())[0], 
            color = "orange", 
            linestyle = "dashed", 
            label = f"Effect size: {cd(df6_male['average_difficulty'].to_numpy(), 
                                            df6_female['average_difficulty'].to_numpy())[0]:.4f}"
            )

# confidence interval lines
# lower
plt.axvline(ci_lower_q6, 
            color = "red", 
            linestyle = "dashed", 
            label = f"95% CI: [{ci_lower_q6:.4f}, {ci_upper_q6:.4f}]"
            )
# upper
plt.axvline(ci_upper_q6, 
            color = "red", 
            linestyle = "dashed")

# aesthetics
# title 
plt.title("Bootstrap Distribution of Cliff's Delta for Average Difficulty Rating by Gender", 
          fontweight = "bold",
          fontsize = 18)

# x and y axis labels
plt.xlabel("Cliff's Delta", 
           fontsize = 16)
plt.ylabel("Density", 
           fontsize = 16)

# ticks 
plt.tick_params(axis = "both", 
                labelsize = 14)

# legend 
plt.legend(fontsize = 16)

# show the plot 
plt.show()

# Question 9
# first, make a subset of all variables that are needed. 
df_9 = df_filtered_final[["average_difficulty"] + df_tags_column_names].copy()

# check 
df_9.info()

# separate y (dependent variable) and x(independent variables)
y = df_9["average_difficulty"]
x = df_9.drop(columns = "average_difficulty")

# check 
x.info()

# split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=n_number)

# check distributions of the data 
print(f'Number of rows for train data:{y_train.shape[0]}\n')
print(f'Number of rows for test data:{y_test.shape[0]}\n')

# correlations between y and x predictors 
corr_yx = df_9.corr()["average_difficulty"].drop("average_difficulty")

corr_yx.sort_values(ascending=False)

# plot correlation of y and x 
plt.figure(figsize= (12, 10))

# plot itself 
sns.barplot(corr_yx, 
            orient= 'h', 
            color = 'skyblue', 
            edgecolor = 'black')

# title 
plt.title("Correlation of Predictors with Average Difficulty", 
          fontsize = 18, 
          fontweight = "bold")

# xlabel
plt.xlabel("Correlation coefficient", 
           fontsize = 16)
# ylabel 
plt.ylabel("Predictor", 
           fontsize = 16)
# ticks 
plt.tick_params(size = 14)

# predictors only
x = df_9.drop(columns="average_difficulty")

# correlation matrix among tags
corr_xx = x.corr()

# plot correlation between predictors 
plt.figure(figsize= (12, 10))

# plot itself
sns.heatmap(
    corr_xx,
    cmap="coolwarm",
    center=0,
    square=True,
    cbar=True,
    annot=True,
    fmt=".1f"
)

# title 
plt.title("Correlation Matrix of Tag Proportions", 
          fontweight = 'bold', 
          fontsize = 18)

# tickts 
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tick_params(size = 16)

plt.show()

# model with all variables 
# pipeline for model 
pipeline_q9 = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])

# cross-validation 
cv = KFold(n_splits=5, 
           shuffle=True, 
           random_state=n_number)

# run cross-validation
cv_results_raw = cross_validate(
    pipeline_q9,
    x_train,
    y_train,
    cv=cv,
    scoring=("r2", "neg_root_mean_squared_error"),
    return_train_score=False
)

cv_results = {
    "R2_mean": cv_results_raw["test_r2"].mean(),
    "RMSE_mean": (-cv_results_raw["test_neg_root_mean_squared_error"]).mean()
}

cv_results

# fit the data 
pipeline_q9.fit(x_train, y_train)

# predict 
y_test_pred = pipeline_q9.predict(x_test)

# evaluate
r2_test = r2_score(y_test, y_test_pred)
rmse_test = root_mean_squared_error(y_test, 
                                    y_test_pred)

# test results 
test_results = {
    "R2": r2_test, 
    "RMSE": rmse_test
}

# check 
test_results


# check coefficients of the model
coef = pipeline_q9.named_steps["model"].coef_

# coefficients in a dataframe
weights_df = pd.DataFrame({
    "tag": x_train.columns,
    "weight": coef
}).sort_values("weight", ascending=False)

# sort by magnitude 
weights_df = weights_df.reindex(weights_df["weight"].abs().sort_values(ascending=False).index)

# Plot weights (coefficients)
# figure size 
plt.figure(figsize=(14, 6))

# plot itself 
sns.barplot(
    data=weights_df,
    x="tag",
    y="weight",
    color="yellow",
    edgecolor="black"
)

# horizontal line
plt.axhline(0, color="black", linewidth=1)

# title 
plt.title("Linear Regression Weights Sorted by Magnitude for Predicting Average Difficulty", 
          fontsize = 18, 
          fontweight = "bold")

# xlabel 
plt.xlabel("Tag", fontsize = 16)
# ylabel
plt.ylabel("Standardized regression coefficient", 
           fontsize = 16)

# ticks
plt.xticks(rotation=45, ha="right")
plt.tick_params(size = 14)

# show the plot 
plt.show()

# first, make a subset of all variables that are needed. 
df_10 = df_filtered_final.copy()

# check 
df_10.info()

# drop female_gender as it is redundant 
df_10 = df_10.drop(columns = ['female_gender', 
                            'major', 
                            'university', 
                            'state'])
# check 
df_10.info()

# correlations between y and x predictors 
corr_yx_10 = df_10.corr()["received_a_pepper"].drop("received_a_pepper")

# check 
corr_yx_10.sort_values(ascending=False)

# plot correlation of y and x 
plt.figure(figsize= (12, 10))

# plot itself 
sns.barplot(corr_yx_10, 
            orient= 'h', 
            color = 'purple', 
            edgecolor = 'black')

# title 
plt.title("Correlation of Predictors with Received a pepper", 
          fontsize = 18, 
          fontweight = "bold")

# xlabel
plt.xlabel("Correlation coefficient", 
           fontsize = 16)
# ylabel 
plt.ylabel("Predictor", 
           fontsize = 16)
# ticks 
plt.tick_params(size = 14)