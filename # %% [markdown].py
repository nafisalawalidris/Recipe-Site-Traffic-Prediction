# %% [markdown]
# # Recipe High Traffic Prediction
# 

# %% [markdown]
# # 1. Data Validation and Cleaning Report
# 
# ## Overview
# The dataset contains 947 rows and 8 columns. After data validation, some modifications were made to handle missing values and ensure consistency in the data. This report provides an overview of the data validation process and the changes made.
# 
# ### Data Validation
# - **Recipe:** There are 947 unique identifiers without any missing values. After dataset cleaning, 52 rows were removed due to missing values in other columns.
# 
# - **Calories:** There are 895 non-null values. 52 missing values were filled with the mean of the "calories" column grouped by "category" and "servings".
# 
# - **Carbohydrate:** There are 895 non-null values. 52 missing values were filled with the mean of the "carbohydrate" column grouped by "category" and "servings".
# 
# - **Sugar:** There are 895 non-null values. 52 missing values were filled with the mean of the "sugar" column grouped by "category" and "servings".
# 
# - **Protein:** There are 895 non-null values. 52 missing values were filled with the mean of the "protein" column grouped by "category" and "servings".
# 
# - **Category:** There are 11 unique values without any missing values. It was found that there was an additional value "Chicken Breast," which was united with the "Chicken" category.
# 
# - **Servings:** There are 6 unique values without any missing values. The column type was changed to integer, and two extra values "4 as a snack" and "6 as a snack" were united with "4" and "6," respectively.
# 
# - **High Traffic:** There is only 1 non-null value ("High"). The null values were replaced with "Low."
# 
# ### Data Cleaning
# - Rows with missing values in "calories," "carbohydrate," "sugar," and "protein" columns were removed to maintain data integrity.
# 
# - "Chicken Breast" category was united with the "Chicken" category to ensure consistency in the categories.
# 
# - Extra values "4 as a snack" and "6 as a snack" in the "servings" column were united with "4" and "6," respectively, and the column type was changed to integer.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set the style for the plots (optional, but it enhances the visual appearance)
sns.set(style="whitegrid")

# Now you can use the seaborn library for creating more visually appealing plots

# %%
import pandas as pd

# Load the dataset into a DataFrame
df = pd.read_csv('recipe_site_traffic_2212.csv')

# Display information about the DataFrame
df.info()


# %%
import pandas as pd

# Load the dataset into a DataFrame
df = pd.read_csv('recipe_site_traffic_2212.csv')

# Display the first few rows of the DataFrame
print(df.head())


# %%
# Validate the "recipe" column for uniqueness
unique_recipes = df['recipe'].nunique()

# Display the number of unique recipes
print("Number of unique recipes:", unique_recipes)


# %%
# Validate the "calories" column for uniqueness
unique_calories = df['calories'].nunique()

# Display the number of unique calorie values
print("Number of unique calories:", unique_calories)


# %%
# Validate the "carbohydrate" column for uniqueness
unique_carbohydrate = df['carbohydrate'].nunique()

# Display the number of unique carbohydrate values
print("Number of unique carbohydrate values:", unique_carbohydrate)


# %%
# Validate the "sugar" column for uniqueness
unique_sugar = df['sugar'].nunique()

# Display the number of unique sugar values
print("Number of unique sugar values:", unique_sugar)


# %%
# Validate the "protein" column for uniqueness
unique_protein = df['protein'].nunique()

# Display the number of unique protein values
print("Number of unique protein values:", unique_protein)


# %%
# Validate the "category" column for uniqueness
unique_categories = df['category'].nunique()

# Display the number of unique category values
print("Number of unique categories:", unique_categories)


# %%
# Validate the "servings" column for uniqueness
unique_servings = df['servings'].nunique()

# Display the number of unique serving values
print("Number of unique servings:", unique_servings)


# %%
# Validate the "high_traffic" column for uniqueness
unique_high_traffic = df['high_traffic'].nunique()

# Display the number of unique values in the "high_traffic" column
print("Number of unique values in high_traffic column:", unique_high_traffic)


# %%
# Get the unique values in the "category" column
unique_categories = df['category'].unique()

# Display the unique categories
print("Unique categories:", unique_categories)


# %%
# Get the unique values in the "servings" column
unique_servings = df['servings'].unique()

# Display the unique serving values
print("Unique servings:", unique_servings)


# %%
# Filter rows where 'calories' column has missing values (NaN) and group by 'category' and 'servings'
missing_calories_count = df[df['calories'].isna()].groupby(['category', 'servings'])['recipe'].count()

# Display the count of recipes with missing calories for each category and servings group
print(missing_calories_count)


# %%
# Create a new DataFrame df2 for changing values and columns
df2 = df[list(df)]

# Define the list of columns with nutritional information
nutritional = ['calories', 'carbohydrate', 'sugar', 'protein']

# Remove all recipes with null values in calories, carbohydrate, sugar, and protein columns
df2 = df2.dropna(subset=nutritional)

# Fill missing values in the nutritional columns with the mean of their respective category and servings groups
for name in nutritional: 
    df2[name] = df2[name].fillna(df2.groupby(["category", "servings"])[name].transform('mean'))

# Display information about the DataFrame
df2.info()


# %%
# Check for null values in the nutritional columns before removing rows
print("Null values before removal:")
print(df2[nutritional].isnull().sum())

# Remove all recipes with null values in calories, carbohydrate, sugar, and protein columns
df2 = df2.dropna(subset=nutritional)

# Fill missing values in the nutritional columns with the mean of their respective category and servings groups
for name in nutritional: 
    df2[name] = df2[name].fillna(df2.groupby(["category", "servings"])[name].transform('mean'))

# Check for null values in the nutritional columns after removing rows
print("\nNull values after removal:")
print(df2[nutritional].isnull().sum())

# Display the unique categories in df2
print("\nUnique categories in df2:")
print(df2['category'].unique())


# %%
# Remove all recipes with null values in calories, carbohydrate, sugar, and protein columns
df2 = df2.dropna(subset=nutritional)

# Fill missing values in the nutritional columns with the mean of their respective category and servings groups
for name in nutritional: 
    df2[name] = df2[name].fillna(df2.groupby(["category", "servings"])[name].transform('mean'))

# Check for null values in the nutritional columns after removing rows
print("\nNull values after removal:")
print(df2[nutritional].isnull().sum())

# Display the unique servings in df2
print("\nUnique servings in df2:")
print(df2['servings'].unique())


# %%
# Remove all recipes with null values in calories, carbohydrate, sugar, and protein columns
df2 = df2.dropna(subset=nutritional)

# Fill missing values in the nutritional columns with the mean of their respective category and servings groups
for name in nutritional: 
    df2[name] = df2[name].fillna(df2.groupby(["category", "servings"])[name].transform('mean'))

# Group by 'servings' and count the number of recipes for each serving value
count_per_serving = df2.groupby(['servings'])['category'].count()

# Display the count of recipes for each serving value
print(count_per_serving)


# %%
# Fill missing values in the nutritional columns with the mean of their respective category and servings groups
for name in nutritional: 
    df2[name] = df2[name].fillna(df2.groupby(["category", "servings"])[name].transform('mean'))

# Clean the "servings" column by replacing non-integer values
df2['servings'] = df2['servings'].replace({"4 as a snack": '4', "6 as a snack": '6'}).astype('int')

# Group by 'servings' and count the number of recipes for each serving value
count_per_serving = df2.groupby(['servings'])['recipe'].count()

# Display the count of recipes for each serving value
print(count_per_serving)


# %%
# Fill missing values in the nutritional columns with the mean of their respective category and servings groups
for name in nutritional: 
    df2[name] = df2[name].fillna(df2.groupby(["category", "servings"])[name].transform('mean'))

# Clean the "servings" column by replacing non-integer values
df2['servings'] = df2['servings'].replace({"4 as a snack": '4', "6 as a snack": '6'}).astype('int')

# Modify the "category" column by replacing "Chicken Breast" with "Chicken"
df2['category'] = df2['category'].replace({"Chicken Breast": 'Chicken'})

# Replace null values in the "high_traffic" column with "Low"
df2['high_traffic'].fillna("Low", inplace=True)

# Display the first few rows of the DataFrame df2
df2.head()

# %%
# Fill missing values in the nutritional columns with the mean of their respective category and servings groups
for name in nutritional: 
    df2[name] = df2[name].fillna(df2.groupby(["category", "servings"])[name].transform('mean'))

# Clean the "servings" column by replacing non-integer values
df2['servings'] = df2['servings'].replace({"4 as a snack": '4', "6 as a snack": '6'}).astype('int')

# Modify the "category" column by replacing "Chicken Breast" with "Chicken"
df2['category'] = df2['category'].replace({"Chicken Breast": 'Chicken'})

# Replace null values in the "high_traffic" column with "Low"
df2['high_traffic'].fillna("Low", inplace=True)

# Add new columns for nutritional components in all servings
for name in nutritional:
    df2[name + '_1'] = df2[name] * df2['servings']

# Display the first few rows of the DataFrame df2
df2.head()


# %%
# Fill missing values in the nutritional columns with the mean of their respective category and servings groups
for name in nutritional: 
    df2[name] = df2[name].fillna(df2.groupby(["category", "servings"])[name].transform('mean'))

# Clean the "servings" column by replacing non-integer values
df2['servings'] = df2['servings'].replace({"4 as a snack": '4', "6 as a snack": '6'}).astype('int')

# Modify the "category" column by replacing "Chicken Breast" with "Chicken"
df2['category'] = df2['category'].replace({"Chicken Breast": 'Chicken'})

# Replace null values in the "high_traffic" column with "Low"
df2['high_traffic'].fillna("Low", inplace=True)

# Add new columns for nutritional components in all servings
for name in nutritional:
    df2[name + '_1'] = df2[name] * df2['servings']

# Display information about the DataFrame df2
df2.info()


# %%
# Validate negative values in numeric variables
numeric_columns = df2.select_dtypes(include='number').columns
has_negative_values = df2[numeric_columns].lt(0).any().any()

# Display if any negative values are present
print("Are there any negative values in numeric variables?", has_negative_values)

# Validate any negative values in numeric variables using describe
df2.describe()

# %%
# Create a new DataFrame df2 for changing values and columns (include all the previous data cleaning steps here)

# Drop duplicates and get the shape
shape_after_drop_duplicates = df2.drop_duplicates().shape

# Display the shape (number of rows and columns) after removing duplicates
print("Shape after removing duplicates:", shape_after_drop_duplicates)

# %% [markdown]
# # 2. Data Visualisation
# 
# After investigating the target variable (high_traffic) and the features of the recipe dataset, as well as exploring the relationship between the target variable and features, I have decided not to make any changes to the variables. The dataset appears to be suitable for further analysis and modeling as it is.
# 
# During the data visualisation process, I used various plots and charts to gain insights into the data distribution and patterns. I examined the distribution of single variables using histograms, box plots, and other visualisations. Additionally, I explored the relationships between the target variable (high_traffic) and other features using scatter plots, bar charts and groupby analysis.
# 
# Based on the visualisations, I observed that the dataset is well-structured, and there are no major anomalies or outliers that require immediate data transformation. Therefore, I believe the current variables are appropriate for the analysis and modeling tasks.

# %% [markdown]
# ## 2. Data Visualisation
# 
# **Target Variable - high_traffic**
# 
# Since I need to predict the high_traffic, the `high_traffic` variable would be the target variable. The goal is to build a predictive model that can determine whether a recipe will result in high traffic to the website when featured on the homepage.
# 
# After investigating the target variable (`high_traffic`) and the features of the recipe dataset, as well as exploring the relationship between the target variable and features, I have decided not to make any changes to the variables. The dataset appears to be suitable for further analysis and modeling as it is.
# 
# During the data visualisation process, I used various plots and charts to gain insights into the data distribution and patterns. I examined the distribution of single variables using histograms, box plots and other visualisations. Additionally, I explored the relationships between the target variable and other features using scatter plots, bar charts and groupby analysis.
# 
# Based on the visualisations, I observed that the dataset is well-structured, and there are no major anomalies or outliers that require immediate data transformation. Therefore, I believe the current variables are appropriate for the analysis and modeling tasks.

# %%
plt.figure(figsize=(10, 6))
colors = ['#ffbf80', '#66c2a5']  # Custom color palette
sns.set_style("whitegrid")
sns.set_palette(colors)
sns.countplot(data=df2, x="servings", hue="high_traffic")

# Set title and labels
plt.title("Count of High Traffic Recipes by Servings", fontsize=16, fontweight='bold', color='#333333')
plt.xlabel("Servings", fontsize=12, color='#666666')
plt.ylabel("Count", fontsize=12, color='#666666')

# Customize legend
plt.legend(title="High Traffic", labels=["Low", "High"], title_fontsize=12)

# Remove spines
sns.despine()

# Show the plot
plt.show()


# %% [markdown]
# ## Conclusion
# 
# Based on the data visualiation of the relationship between the `high_traffic` variable and the `servings` feature, it can be observed that for each serving size, the number of recipes with high traffic is higher than the number of recipes with low traffic. This indicates that the `servings` feature may not have a significant influence on the target variable (`high_traffic`).
# 
# The count plot shows that the recipes with high traffic tend to be distributed across various serving sizes, ranging from small to large. Similarly, recipes with low traffic are also distributed across different serving sizes. Therefore, it seems that the number of servings alone may not be a decisive factor in predicting whether a recipe will result in high traffic on the website when featured on the homepage.

# %%
plt.figure(figsize=(12, 8))
colors = ['#66c2a5', '#fc8d62']  # Custom color palette
sns.set_style("whitegrid")
sns.set_palette(colors)
sns.countplot(data=df2, y="category", hue="high_traffic")

# Set title and labels
plt.title("Count of High Traffic Recipes by Category", fontsize=16, fontweight='bold', color='#333333')
plt.xlabel("Count", fontsize=12, color='#666666')
plt.ylabel("Category", fontsize=12, color='#666666')

# Customize legend
plt.legend(title="High Traffic", labels=["Low", "High"], title_fontsize=12)

# Remove spines
sns.despine()

# Show the plot
plt.show()

# %% [markdown]
# ## Conclusion
# 
# Based on the count plot of high traffic recipes by category, we can draw the following observations:
# 
# 1. Potato, Pork and Vegetable categories have a significantly higher number of recipes with high traffic compared to recipes with low traffic. These categories seem to attract more user interest and engagement, resulting in a higher likelihood of high traffic when recipes from these categories are featured on the homepage.
# 
# 2. One Dish Meal, Lunch/Snacks, Meat, and Dessert categories also have more recipes with high traffic than with low traffic. While the difference in counts is not as substantial as in the previously mentioned categories, it indicates that recipes from these categories still have a good chance of generating high traffic when featured on the website.
# 
# These findings provide valuable insights into the popularity of recipe categories and their potential impact on website traffic. To leverage these insights effectively, the Tasty Bytes team can consider promoting recipes from the Potato, Pork and Vegetable categories more prominently on the homepage, as they have a higher likelihood of attracting more visitors. Additionally, they can focus on optimising recipes from the One Dish Meal, Lunch/Snacks, Meat and Dessert categories to increase their appeal and engagement.

# %% [markdown]
# ## Numeric Variables - calories, carbohydrate, sugar, protein
# 
# From the heatmap below, we can observe the correlation between the numeric variables - calories, carbohydrate, sugar, protein, and servings.

# %%
import seaborn as sns
import matplotlib.pyplot as plt

# Create a heatmap to visualize the correlation between numeric variables
numeric = df2[['calories_1', 'carbohydrate_1', 'sugar_1', 'protein_1', 'servings']]
correlation_matrix = numeric.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap between Numeric Variables", fontsize=16, fontweight='bold', color='#333333')

plt.show()

# %% [markdown]
# ## Categorical Variables - Servings, Category, High Traffic
# 
# I have analyzed the distribution of recipes based on the categorical variables - `servings`, `category`, and `high_traffic`.
# 
# ### Servings:
# 
# The bar chart shows the distribution of recipes across different serving sizes. From the plot, we can observe that recipes with 4 servings are the most popular, followed by recipes with 6 servings. This suggests that recipes designed for 4 servings are preferred by a larger portion of users, which could potentially have a positive influence on high traffic.
# 
# ### Category:
# 
# The bar chart displays the distribution of recipes across various categories. The "Chicken" category stands out as the most popular, indicating that chicken-based recipes are favored by a significant number of users. Other popular categories include "Potato," "Vegetable," and "Pork." These categories could have a notable impact on high traffic, considering their popularity among users.
# 
# ### High Traffic:
# 
# The bar chart presents the distribution of recipes based on the high traffic variable. It shows the number of recipes with high and low traffic on the website. Although the dataset contains significantly more recipes with high traffic, it's important to note that this variable is imbalanced, with only one non-null value for "High." Therefore, further analysis and modeling will be required to build a robust prediction model for high traffic recipes.
# 
# The distribution of recipes by these categorical variables provides valuable insights into user preferences and potential influencers on high traffic. To leverage these insights effectively, the Tasty Bytes team can focus on promoting recipes with 4 servings and explore more recipe options in the popular categories like "Chicken," "Potato," "Vegetable," and "Pork" to attract more visitors and increase website traffic.

# %%
import seaborn as sns
import matplotlib.pyplot as plt

# Create a count plot for servings
plt.figure(figsize=(8, 6))
sns.countplot(data=df2, x="servings")

# Set title and labels
plt.title("Distribution of Recipes by Servings", fontsize=16, fontweight='bold', color='#333333')
plt.xlabel("Servings", fontsize=12, color='#666666')
plt.ylabel("Count", fontsize=12, color='#666666')

# Show the plot
plt.show()


# %%
import seaborn as sns
import matplotlib.pyplot as plt

# Create a count plot for categories
plt.figure(figsize=(12, 6))
sns.countplot(data=df2, x="category")

# Rotate x-axis labels for better readability
plt.xticks(rotation=90)

# Set title and labels
plt.title("Distribution of Recipes by Category", fontsize=16, fontweight='bold', color='#333333')
plt.xlabel("Category", fontsize=12, color='#666666')
plt.ylabel("Count", fontsize=12, color='#666666')

# Show the plot
plt.show()

# %% [markdown]
# The boxplots display the distribution and spread of each numerical feature. The central line within each box represents the median, and the box spans the interquartile range (IQR). Any data points outside the whiskers are potential outliers.
# 
# Based on the boxplots, I can confirm that there are no visible outliers in the original numerical features - calories, carbohydrate, sugar, protein, and servings. The absence of outliers suggests that the dataset is relatively well-behaved and does not contain extreme values that could significantly impact the analysis and modeling tasks.

# %%
import seaborn as sns
import matplotlib.pyplot as plt

# Create a boxplot for the 'calories' variable
plt.figure(figsize=(8, 6))
sns.boxplot(x='calories', data=df2)

# Set title and labels
plt.title("Boxplot of 'Calories'", fontsize=16, fontweight='bold', color='#333333')
plt.xlabel("Calories", fontsize=12, color='#666666')

# Show the plot
plt.show()

# %%
import seaborn as sns
import matplotlib.pyplot as plt

# Create a boxplot for the 'sugar' variable
plt.figure(figsize=(8, 6))
sns.boxplot(x='sugar', data=df2)

# Set title and labels
plt.title("Boxplot of 'Sugar'", fontsize=16, fontweight='bold', color='#333333')
plt.xlabel("Sugar", fontsize=12, color='#666666')

# Show the plot
plt.show()


# %%
import seaborn as sns
import matplotlib.pyplot as plt

# Create a boxplot for the 'carbohydrate' variable
plt.figure(figsize=(8, 6))
sns.boxplot(x='carbohydrate', data=df2)

# Set title and labels
plt.title("Boxplot of 'Carbohydrate'", fontsize=16, fontweight='bold', color='#333333')
plt.xlabel("Carbohydrate", fontsize=12, color='#666666')

# Show the plot
plt.show()

# %%
import seaborn as sns
import matplotlib.pyplot as plt

# Create a boxplot for the 'protein' variable
plt.figure(figsize=(8, 6))
sns.boxplot(x='protein', data=df2)

# Set title and labels
plt.title("Boxplot of 'Protein'", fontsize=16, fontweight='bold', color='#333333')
plt.xlabel("Protein", fontsize=12, color='#666666')

# Show the plot
plt.show()


# %% [markdown]
# The bar charts display the count of the most frequent categories in the 'servings', 'category', and 'high_traffic' variables. The color palettes are chosen to provide better visibility and visual appeal.
# 
# From the bar charts, we can observe the following:
# 
# 1. Most Frequent Servings: Recipes with 4 servings are the most frequent, followed by recipes with 6 servings.
# 
# 2. Most Frequent Categories: The "Chicken" category appears to be the most frequent, indicating that chicken-based recipes are highly represented in the dataset. Other popular categories include "Potato," "Vegetable," and "Pork."
# 
# 3. Most Frequent High Traffic: The dataset contains significantly more recipes with high traffic compared to low traffic. However, it's important to note that this variable is imbalanced, with only one non-null value for "High."
# 
# These bar charts provide valuable insights into the most frequent categories, servings, and high traffic in the dataset, which can guide further analysis and decision-making for the Tasty Bytes team.

# %%
import seaborn as sns
import matplotlib.pyplot as plt

# Create subplots for the bar charts
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Define custom colors for the bars in each subplot
colors_servings = ['cadetblue', 'lightseagreen', 'mediumturquoise', 'deepskyblue', 'cornflowerblue', 'steelblue']
colors_category = ['lightgreen', 'limegreen', 'forestgreen', 'olivedrab', 'yellowgreen', 'darkgreen', 'lime', 'green', 'springgreen', 'palegreen', 'seagreen']
colors_high_traffic = ['lightblue', 'deepskyblue']

# Plot the count of servings
sns.countplot(x=df2['servings'], palette=colors_servings, ax=axes[0]).set(title='Count of Servings')
axes[0].set_xlabel('Servings')
axes[0].set_ylabel('Count')

# Plot the count of categories
sns.countplot(x=df2['category'], palette=colors_category, ax=axes[1]).set(title='Count of Category')
axes[1].set_xlabel('Category')
axes[1].set_ylabel('Count')
axes[1].tick_params(axis='x', rotation=90)

# Plot the count of high_traffic
sns.countplot(x=df2['high_traffic'], palette=colors_high_traffic, ax=axes[2]).set(title='Count of High Traffic')
axes[2].set_xlabel('High Traffic')
axes[2].set_ylabel('Count')

plt.tight_layout()

# Show the plots
plt.show()

# %% [markdown]
# In this heatmap, each cell represents the difference between the mean of high-traffic and low-traffic recipes for a specific combination of servings, categories, and numerical features. Positive values indicate that the mean of high-traffic recipes is higher, while negative values indicate that the mean of low-traffic recipes is higher. Values close to 0 suggest that there is little difference between the means of high-traffic and low-traffic recipes for that specific combination.
# 
# By visualising this difference, we can gain insights into how certain combinations of servings, categories, and numerical features influence the popularity of recipes and their potential impact on high traffic.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define the names of numerical features to analyze
nutritional_1 = ['calories_1', 'carbohydrate_1', 'sugar_1', 'protein_1']

def diff_for_numerical(df2, name):
    df2_high_agg = pd.pivot_table(df2[df2['high_traffic'] == 'High'], index=["category"], columns=["servings"], values=name, aggfunc=np.mean)
    df2_low_agg = pd.pivot_table(df2[df2['high_traffic'] == 'Low'], index=["category"], columns=["servings"], values=name, aggfunc=np.mean)
    df2_diff = df2_high_agg.subtract(df2_low_agg)

    f, ax = plt.subplots(figsize=(5, 6))
    sns.heatmap(df2_diff, annot=True, fmt=".1f", linewidths=.5, ax=ax, cmap="viridis", center=0)
    plt.title('Difference between means for {0}'.format(name))
    plt.show()

    return df2_diff

for name in nutritional_1:
    diff_for_numerical(df2, name)

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define the names of numerical features to analyze
nutritional_1 = ['calories_1', 'carbohydrate_1', 'sugar_1', 'protein_1']

# Create a pivot table to calculate the mean values of numerical features based on 'high_traffic'
num_pivot_df = pd.pivot_table(df2, index=["high_traffic"], values=nutritional_1, aggfunc=np.mean)

# Define custom colors for the bars
colors = ['royalblue', 'lightcoral', 'mediumseagreen', 'gold']

# Create a bar plot to visualize the mean values
num_pivot_df.plot(kind='bar', figsize=(10, 6), color=colors)

# Set title and labels
plt.title('Mean Values of Nutritional Components by High Traffic', fontsize=16, fontweight='bold')
plt.xlabel('High Traffic', fontsize=12)
plt.ylabel('Mean Value', fontsize=12)

# Show the plot
plt.show()


# %%
fig, ax = plt.subplots(2, 2, figsize=(12, 10))

# Define custom color palettes for the violin plots
palette_high = ['lightcoral', 'lightblue']
palette_low = ['salmon', 'skyblue']

# Create violin plots for each numerical feature
for i, name in enumerate(nutritional_1):
    sns.violinplot(data=df2, x='high_traffic', y=name, split=True, ax=ax[i // 2, i % 2], palette={'High': palette_high[i % 2], 'Low': palette_low[i % 2]})
    ax[i // 2, i % 2].set_xlabel('High Traffic')
    ax[i // 2, i % 2].set_ylabel(name)
    ax[i // 2, i % 2].set_title('Distribution of {0} by High Traffic'.format(name))

plt.tight_layout()
plt.show()


# %%
fig, ax = plt.subplots(2, 2, figsize=(12, 10))

# Define custom color palettes for the KDE plots
palette_high = ['tab:red', 'tab:blue']
palette_low = ['lightcoral', 'lightblue']

# Create KDE plots for each numerical feature
for i, name in enumerate(nutritional_1):
    sns.kdeplot(data=df2, hue='high_traffic', x=name, shade=True, ax=ax[i // 2, i % 2], palette={'High': palette_high[i % 2], 'Low': palette_low[i % 2]})
    ax[i // 2, i % 2].set_xlabel(name)
    ax[i // 2, i % 2].set_ylabel('Density')
    ax[i // 2, i % 2].set_title('KDE Plot of {0} by High Traffic'.format(name))

plt.tight_layout()
plt.show()


# %% [markdown]
# Conclusion:
# 
# After conducting exploratory data analysis using violin plots and kernel density estimation (KDE) plots, we observed that there are no significant dependencies between the traffic ('high_traffic' variable) and the following numerical features: calories, carbohydrate, protein, sugar, and servings. The distributions of these numerical features do not show clear patterns of separation between high-traffic and low-traffic recipes.
# 
# However, the categorical feature 'category' has a significant effect on the target variable ('high_traffic'). From the count plots and violin plots, we can see that certain categories, such as Potato, Pork, and Vegetable, have a much higher number of recipes with high traffic compared to low traffic. On the other hand, categories like One Dish Meal, Lunch/Snacks, Meat, and Dessert show a slightly higher count of recipes with high traffic compared to low traffic.
# 
# Overall, the analysis suggests that the 'category' feature plays a crucial role in determining whether a recipe will attract high traffic or not, while the numerical features do not exhibit strong relationships with the target variable.

# %% [markdown]
# # 3. Model Fitting #
# 
# For the task of predicting 'high_traffic', a binary classification problem, I have selected two models: Logistic Regression and Linear Support Vector Classification (Linear SVC).
# 
# - Logistic Regression: A widely-used algorithm for binary classification. It models the probability of belonging to a class using the logistic function.
# 
# - Linear SVC: A variant of Support Vector Machine (SVM) for binary classification. It aims to find an optimal hyperplane to separate the classes.
# 
# Both models will be trained on the labeled data, and their performance will be evaluated using metrics like accuracy, precision, recall, and F1-score.
# 
# The chosen model will be used to predict high-traffic recipes on the website's homepage, potentially increasing website traffic and subscriptions.

# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load the data and split into features and target variable
X = df2.drop('high_traffic', axis=1)
y = df2['high_traffic']

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the numerical features using StandardScaler or MinMaxScaler
scaler = StandardScaler()  # You can also use MinMaxScaler for scaling between 0 and 1
numerical_features = ['calories_1', 'carbohydrate_1', 'sugar_1', 'protein_1', 'servings']
X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_test[numerical_features] = scaler.transform(X_test[numerical_features])

# Encode the categorical feature 'category'
label_encoder = LabelEncoder()
X_train['category'] = label_encoder.fit_transform(X_train['category'])
X_test['category'] = label_encoder.transform(X_test['category'])

# Initialize and train the Logistic Regression model
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# Make predictions using Logistic Regression
y_pred_logreg = logreg.predict(X_test)

# Evaluate the Logistic Regression model
print("Logistic Regression Results:")
print(classification_report(y_test, y_pred_logreg))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_logreg))
print("Accuracy Score:", accuracy_score(y_test, y_pred_logreg))

# Initialize and train the Linear SVC model
svc = LinearSVC()
svc.fit(X_train, y_train)

# Make predictions using Linear SVC
y_pred_svc = svc.predict(X_test)

# Evaluate the Linear SVC model
print("Linear SVC Results:")
print(classification_report(y_test, y_pred_svc))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_svc))
print("Accuracy Score:", accuracy_score(y_test, y_pred_svc))


# %% [markdown]
# ## Prepare Data for Modelling ##
# 
# Step 1: Choose the features and target variable for modeling. In this case, the selected features are 'calories', 'carbohydrate', 'sugar', 'protein', 'servings', and 'category'. The target variable is 'high_traffic'.
# 
# Step 2: Convert the categorical variable 'category' into a numeric feature. This is done using the LabelEncoder, which assigns a unique numeric value to each category in the 'category' feature.
# 
# Step 3: Normalize the numeric features to bring them to a common scale. The StandardScaler is used to transform the numeric features ('calories', 'carbohydrate', 'sugar', 'protein', and 'servings') so that they have a mean of 0 and a standard deviation of 1.
# 
# Step 4: Split the data into a training set and a test set. The dataset is divided into two parts: 80% for training the model and 20% for testing the model's performance.
# 
# With these steps, the data is now ready for modeling with the selected features and the target variable suitably prepared and split into training and testing sets.

# %%
# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Create a copy of the dataframe
df3 = df2.copy()

# Convert the categorical variables into numeric features
labelencoder = LabelEncoder()
df3['category'] = labelencoder.fit_transform(df3['category'])
df3['high_traffic'] = df3['high_traffic'].replace({"High": 1, "Low": 0})

# Select features and target variable
num_features = ['calories', 'carbohydrate', 'sugar', 'protein', 'servings']
cat_feature = ['category']
target_variable = 'high_traffic'
X = df3[num_features + cat_feature]  # Features
y = df3[target_variable]  # Target variable

# Normalize the numeric features
scaler = StandardScaler()
X[num_features] = scaler.fit_transform(X[num_features])

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# %%
# Create a dictionary to map the encoded categories to their original labels
labels_dict = dict(zip(labelencoder.classes_, range(len(labelencoder.classes_))))

# %% [markdown]
# ## Model 1. Logistic Regression ##

# %%
# Import necessary libraries
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

# Define the hyperparameter grid
grid = {
    "C": np.logspace(-3, 3, 7),  # C: Inverse of regularization strength, it's a set of values from 0.001 to 1000 (7 steps in logarithmic scale)
    "penalty": ["l1", "l2", "elasticnet", None],  # Regularization penalty to be used (L1, L2, ElasticNet, or None)
    "multi_class": ["auto", "ovr", "multinomial"]  # Strategy for handling multiple classes (Auto, One-vs-Rest, or Multinomial)
}

# Create a Logistic Regression model
logreg = LogisticRegression()

# Perform Grid Search Cross Validation with 10 folds
logreg_cv = GridSearchCV(logreg, grid, cv=10)

# Fit the model with training data
logreg_cv.fit(X_train, y_train)

# Print the best hyperparameters found
print("Tuned hyperparameters:", logreg_cv.best_params_)

# %% [markdown]
# ## Model 2: Linear Support Vector Classification ##

# %%
# Define the hyperparameter grid for LinearSVC
grid = {
    "C": np.logspace(-3, 3, 7),  # C: Inverse of regularization strength, it's a set of values from 0.001 to 1000 (7 steps in logarithmic scale)
    "penalty": ["l1", "l2"],     # Regularization penalty to be used (L1 or L2)
    "loss": ["hinge", "squared_hinge"]  # Loss function to be used (Hinge or Squared Hinge)
}

# Create a Linear Support Vector Classification (LinearSVC) model
svm = LinearSVC()

# Perform Grid Search Cross Validation with 10 folds
svm_cv = GridSearchCV(svm, grid, cv=10)

# Fit the model with training data
svm_cv.fit(X_train, y_train)

# Print the best hyperparameters found
print("Tuned hyperparameters:", svm_cv.best_params_)

# %% [markdown]
# # 4. Model Evalution #
# 
# For the model evaluation, I will focus on three key metrics: Precision, Recall, and F1 Score.
# 
# - Precision: Precision measures the proportion of true positive predictions among all the positive predictions made by the model. In other words, it calculates the accuracy of the positive predictions.
# 
# - Recall: Recall measures the proportion of true positive predictions among all the actual positive instances in the dataset. It indicates the model's ability to correctly identify positive instances.
# 
# - F1 Score: The F1 Score is the harmonic mean of precision and recall. It provides a balance between precision and recall, and it is useful when the class distribution is imbalanced.
# 
# By considering these metrics, we can better assess the performance of the model in predicting high traffic recipes accurately.

# %%
logreg2 = LogisticRegression(C=0.001, multi_class='multinomial', penalty="l2")  # best parameters
logreg2.fit(X_train, y_train)
y_pred_logreg = logreg2.predict(X_test)

print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_logreg))
print("Classification report:\n", classification_report(y_test, y_pred_logreg))


# %%
resultdict = {}
for i, feature in enumerate(list(X)):
    resultdict[feature] = logreg2.coef_[0][i]

plt.bar(resultdict.keys(), resultdict.values())
plt.xticks(rotation='vertical')
plt.title('Feature Importance for Logistic Regression model')
plt.show()

# %% [markdown]
# As observed earlier, the "category" feature has the most significant effect on the prediction of high traffic. The Logistic Regression model's feature importance analysis further confirms this finding. The bar chart above shows that the "category" feature has the highest coefficient value, indicating its strong influence on predicting high traffic. Other features such as "calories_1", "carbohydrate_1", "sugar_1", "protein_1", and "servings" also contribute to the prediction, but their coefficients are comparatively smaller. This highlights the importance of considering the recipe category when trying to predict high traffic on the website.

# %%
svm2 = LinearSVC(C=0.01, loss='squared_hinge', penalty='l2') # best parameters
svm2.fit(X_train, y_train)
y_pred_svm = svm2.predict(X_test)

print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_svm))
print("Classification report:\n", classification_report(y_test, y_pred_svm))

# %%
# Create a dictionary to store feature importance
resultdict = {}
for i in range(len(list(X))):
    resultdict[list(X)[i]] = svm2.coef_[0][i]

# Create a colorful bar chart for feature importance
plt.figure(figsize=(10, 6))
bars = plt.bar(resultdict.keys(), resultdict.values(), color='lightblue')

# Customize the plot
plt.xticks(rotation='vertical')
plt.title('Feature Importance for Linear SVC Model')
plt.xlabel('Features')
plt.ylabel('Importance')

# Add data labels to the bars
for bar in bars:
    height = bar.get_height()
    plt.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 3),  # 3 points vertical offset
                 textcoords="offset points",
                 ha='center', va='bottom')

# Show the plot
plt.show()

# %% [markdown]
# Conclusion:
# 
# The Logistic Regression model outperforms the Linear Support Vector Classification model in predicting High traffic based on Precision, Recall, and F1 Score. The Logistic Regression model achieves a Precision of 0.82, Recall of 0.80, and F1 Score of 0.81, while the Linear Support Vector Classification model achieves a Precision of 0.80, Recall of 0.77, and F1 Score of 0.79. Therefore, the Logistic Regression model provides a better fit to the features and exhibits lower prediction errors compared to the Linear Support Vector Classification model.

# %% [markdown]
# # 5. Business Focus & Metrics #
# The primary business goals for this project are twofold:
# 
# - To predict which recipes will have high traffic.
# - To predict the "High" value of traffic for recipes with at least 80% probability.
# 
# The Logistic Regression model successfully accomplishes both of these goals, as it achieves high Precision, Recall, and F1 Score, all of which are equal to or greater than 80%. This indicates that the model is effective in identifying recipes with high traffic and making accurate predictions with a high level of confidence. As a result, the Logistic Regression model is well-suited for the business objectives and can provide valuable insights into recipe performance and traffic on the website.

# %% [markdown]
# # Recommendations for Future Actions #
# 
# 1. Deployment of Logistic Regression Model: Deploy the trained Logistic Regression model into production to help the Product Manager predict recipes with high traffic. With approximately 81% accuracy in predicting high traffic, this model can assist in generating more traffic to the website and boost overall performance.
# 
# 2. Optimise Model Deployment: Explore efficient ways to deploy the model in terms of performance and cost. Consider deploying the machine learning model on edge devices for improved convenience and security. Additionally, test the model with newly hired product analysts to gather real-world feedback and enhance its performance.
# 
# 3. Data Collection: Collect additional data to improve the model's predictive capabilities. Gather data on factors such as time to make the recipe, cost per serving, ingredient details, site duration time (how long users spend on the recipe page), incoming links (source of user traffic to the recipe page), and combinations of recipes visited during the same session.
# 
# 4. Feature Engineering: Enhance the model's features by increasing the number of categories and creating more meaningful features from existing variables. This could involve grouping similar categories, extracting relevant information from the recipe description, or incorporating user interaction data to capture user preferences better.
# 
# By implementing these recommendations, the Product Manager can benefit from a more accurate and effective predictive model, leading to better insights into recipe performance and website traffic. The iterative improvement of the model and continuous data collection will enable the Product Manager to make data-driven decisions and optimize recipe content to attract more users and drive higher traffic to the website.

# %% [markdown]
# ### Predictive System ###
# 
# let's check the trained Logistic Regression model to predict the traffic category (high or low) for any recipe from the test data. I randomly select a recipe from the test set and use its features as input to the model.

# %%
# Randomly select a recipe from the test set
import random
random.seed(42)
random_recipe_index = random.randint(0, len(X_test) - 1)
random_recipe_features = X_test.iloc[random_recipe_index]

# Reshape the features to match the model's input shape
random_recipe_features = random_recipe_features.values.reshape(1, -1)

# Make the prediction using the Logistic Regression model
predicted_traffic = logreg2.predict(random_recipe_features)[0]
if predicted_traffic == 1:
    predicted_traffic_category = "High"
else:
    predicted_traffic_category = "Low"

# Get the actual traffic category from the test set
actual_traffic = y_test.iloc[random_recipe_index]
if actual_traffic == 1:
    actual_traffic_category = "High"
else:
    actual_traffic_category = "Low"

# Print the results
print("Randomly Selected Recipe Features:")
print(random_recipe_features)
print("\nPredicted Traffic Category: ", predicted_traffic_category)
print("Actual Traffic Category: ", actual_traffic_category)


# %% [markdown]
# ## KPI and the performance of 2 models using KPI ##
# 
# The company's key performance indicator (KPI) for evaluating the models' performance is the accuracy of predictions for high traffic. A higher percentage indicates a better-performing model in terms of correctly predicting high traffic recipes. Upon evaluation, the Logistic Regression model achieved an accuracy of 76%, while the Linear SVC model had a slightly lower accuracy of 74%. Therefore, the Logistic Regression model outperforms the Linear SVC model in terms of accurately predicting high traffic recipes.

# %%
# Accuracy score on the train data
X_train_pred = logreg2.predict(X_train)
train_accuracy = accuracy_score(y_train, X_train_pred)
print('Accuracy score of the train data:', train_accuracy)

# Accuracy score on the test data
X_test_pred = logreg2.predict(X_test)
test_accuracy = accuracy_score(y_test, X_test_pred)
print('Accuracy score of the test data:', test_accuracy)

# %%
# Accuracy score on the train data
y_train_pred = svm2.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
print('Accuracy score of the train data:', train_accuracy)

# Accuracy score on the test data
y_test_pred = svm2.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print('Accuracy score of the test data:', test_accuracy)


