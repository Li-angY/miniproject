import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import f_oneway, ttest_ind, chi2_contingency, kurtosis, skew, kruskal, shapiro
from wordcloud import WordCloud
import statsmodels.api as sm

def load_dataset(file_path):
    """Load CSV dataset and return DataFrame"""
    try:
        df = pd.read_csv(file_path)
        print("Dataset loaded successfully!")
        return df
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return None

def inspect_data(df):
    """Inspect the dataset and output basic statistical information"""
    print("\nHere is the basic information of the dataset:")
    print(df.info())

    # Statistical variables
    stats = {}
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            stats[column] = {
                'Mean': np.mean(df[column]),
                'Median': np.median(df[column]),
                'Mode': df[column].mode()[0],
                'Kurtosis': kurtosis(df[column]),
                'Skewness': skew(df[column])
            }

    # Convert to DataFrame for easier display
    stats_df = pd.DataFrame(stats).T
    print("\nStatistical Information:")
    print(stats_df)

def select_variable(df, prompt):
    """Select a variable from the DataFrame"""
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    print(prompt)
    all_vars = numeric_cols + categorical_cols
    for i, col in enumerate(all_vars, start=1):
        print(f"{i}. {col}")

    # Add back and exit options
    print(f"{len(all_vars) + 1}. back")
    print(f"{len(all_vars) + 2}. exit")
    
    try:
        choice = int(input("Please enter the number of the variable: "))
    except ValueError:
        print("Invalid input. Please enter a number.")
        return None

    if 1 <= choice <= len(all_vars):
        return all_vars[choice - 1]
    elif choice == len(all_vars) + 1:
        return 'back'  # Back option
    elif choice == len(all_vars) + 2:
        return 'exit'  # Exit option
    else:
        print("Invalid choice. Please check your input.")
        return None

def plot_variable_distribution(df):
    """Plot the distribution of a variable"""
    while True:
        var_name = select_variable(df, "\nThe following variables can be used to plot the distribution:")

        if var_name == 'back':
            print("Returning to the previous step.")
            return
        elif var_name == 'exit':
            print("Exiting the program.")
            exit()

        if var_name is not None and var_name in df.columns:
            plt.figure(figsize=(10, 6))  # Set figure size
            plt.hist(df[var_name].dropna(), bins=30, alpha=0.7, color='green')
            plt.title(f'Distribution plot for \'{var_name}\'', fontsize=16)  # Set title
            plt.xlabel(var_name, fontsize=14)  # Set X-axis label
            plt.ylabel('Frequency', fontsize=14)  # Set Y-axis label
            plt.grid()
            plt.show()
            plt.close()
            # Prompt user to continue
            input("The plot has been closed. Press Enter to return to the menu...")
            continue  # Return to variable selection dialog

def perform_anova(df):
    """Perform ANOVA or Kruskal-Wallis test and provide result interpretation"""

    # Extract numerical and categorical variables
    numeric_vars = df.select_dtypes(include=np.number).columns.tolist()
    categorical_vars = df.select_dtypes(include=['object', 'category']).columns.tolist()

    print("\nThe following variables are available for analysis:\n")
    print(f"{'Variable':<20} {'Type'}")
    for var in numeric_vars:
        print(f"{var:<20} Numerical (Interval/Ratio)")
    for var in categorical_vars:
        print(f"{var:<20} Categorical (Ordinal/Nominal)")

    # Input continuous and categorical variables
    continuous_var = input("\nPlease enter a continuous variable (numerical): ").strip()
    categorical_var = input("Please enter a categorical variable: ").strip()

    # Check if the input variables exist in the dataset
    if continuous_var in df.columns and categorical_var in df.columns:
        # Extract valid data
        valid_data = df[continuous_var].dropna()
        if valid_data.empty:
            print(f"No valid data for {continuous_var}.")
            return

        # Generate Q-Q plot
        print("\nGenerating Q-Q plot...")
        sm.qqplot(valid_data, line='s')
        plt.title(f'{continuous_var} Q-Q Scatter Plot')
        plt.show()

        # Use Shapiro-Wilk test for normality
        stat, p = shapiro(valid_data)
        print(f"Shapiro-Wilk test result: Statistic = {stat}, p-value = {p}")

        if p < 0.05:
            # Not normally distributed, perform Kruskal-Wallis test
            print(f"\n‘{continuous_var}’ does not follow a normal distribution... performing Kruskal-Wallis test...")
            groups = [df[continuous_var][df[categorical_var] == cat].dropna() for cat in df[categorical_var].unique()]
            h_stat, p_value = kruskal(*groups)

            print(f"\nKruskal-Wallis test result:\nStatistic = {h_stat}\np-value = {p_value}")
            if p_value < 0.05:
                print("The result is statistically significant.\nTherefore, the null hypothesis is rejected.")
                print(f"The mean of '{continuous_var}' differs significantly across categories of '{categorical_var}'.")
            else:
                print("The result is not statistically significant.\nCannot reject the null hypothesis.")
        else:
            # Normally distributed, perform ANOVA
            print(f"\n‘{continuous_var}’ follows a normal distribution... performing ANOVA...")
            groups = [df[continuous_var][df[categorical_var] == cat].dropna() for cat in df[categorical_var].unique()]
            f_stat, p_value = f_oneway(*groups)

            print(f"\nANOVA test result:\nF-statistic = {f_stat}\np-value = {p_value}")
            if p_value < 0.05:
                print("The result is statistically significant.\nTherefore, the null hypothesis is rejected.")
                print(f"The mean of '{continuous_var}' differs significantly across categories of '{categorical_var}'.")
            else:
                print("The result is not statistically significant.\nCannot reject the null hypothesis.")
    else:
        print("Selected variables are not in the dataset. Please check your input.")

def conduct_t_test(df):
    """Perform t-test and determine whether to reject the null hypothesis based on normality"""
    print("\nPerforming t-test")

    # Extract numerical variables
    numeric_vars = df.select_dtypes(include=np.number).columns.tolist()

    print("\nThe following numerical variables are available for T-test:\n")
    print(f"{'Variable':<20}")
    for var in numeric_vars:
        print(f"{var:<20}")

    # Select the first numerical variable
    var1 = input("\nPlease select the first numerical variable: ").strip()
    if var1 not in df.columns:
        print(f"{var1} is not in the dataset. Please check your input.")
        return

    # Select the second numerical variable
    var2 = input("Please select the second numerical variable: ").strip()
    if var2 not in df.columns:
        print(f"{var2} is not in the dataset. Please check your input.")
        return

    # Extract valid data
    data1 = df[var1].dropna()
    data2 = df[var2].dropna()

    if data1.empty or data2.empty:
        print(f"No valid data available for T-test.")
        return

    # Generate Q-Q plot for the first variable
    print(f"\nGenerating Q-Q plot for {var1}...")
    sm.qqplot(data1, line='s')
    plt.title(f'{var1} Q-Q Scatter Plot')
    plt.show()

    # Generate Q-Q plot for the second variable
    print(f"\nGenerating Q-Q plot for {var2}...")
    sm.qqplot(data2, line='s')
    plt.title(f'{var2} Q-Q Scatter Plot')
    plt.show()

    # Perform Shapiro-Wilk test for the first variable
    stat1, p1 = shapiro(data1)
    print(f"\nShapiro-Wilk test result for {var1}: Statistic = {stat1}, p-value = {p1}")

    # Perform Shapiro-Wilk test for the second variable
    stat2, p2 = shapiro(data2)
    print(f"\nShapiro-Wilk test result for {var2}: Statistic = {stat2}, p-value = {p2}")

    # If both variables are normally distributed, perform T-test
    if p1 >= 0.05 and p2 >= 0.05:
        print(f"\nBoth variables follow a normal distribution. Performing independent samples T-test...")
        t_stat, p_value = ttest_ind(data1, data2)
        print(f"\nT-test result: t-statistic = {t_stat}, p-value = {p_value}")
        if p_value < 0.05:
            print("The result is significant. Reject the null hypothesis.")
            print(f"There is a statistically significant difference between '{var1}' and '{var2}'.")
        else:
            print("The result is not significant. Cannot reject the null hypothesis.")
    else:
        print(f"\nAt least one variable does not follow a normal distribution. Cannot perform T-test. Consider using non-parametric test methods.")

def conduct_chi_square(df):
    """Perform Chi-square test"""
    print("\nPerforming Chi-square test")
    
    var1 = select_variable(df, "Please select the first categorical variable:")
    if var1 == 'back':
        return
    elif var1 == 'exit':
        exit()

    var2 = select_variable(df, "Please select the second categorical variable:")
    if var2 == 'back':
        return
    elif var2 == 'exit':
        exit()
    
    if var1 in df.columns and var2 in df.columns:
        contingency_table = pd.crosstab(df[var1], df[var2])
        chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
        print(f"Chi-square Statistic = {chi2_stat}, p-value = {p_value}")
        if p_value < 0.05:
            print("The result is significant. Reject the null hypothesis.")
        else:
            print("The result is not significant. Cannot reject the null hypothesis.")
    else:
        print("Selected variables do not exist in the dataset. Please check your input.")

def conduct_sentiment_analysis(df):
    """Perform simple sentiment analysis (example)"""
    print("\nPerforming sentiment analysis")
    if 'Sentiment' in df.columns:
        text = ' '.join(df['Sentiment'].dropna())
        wordcloud = WordCloud(width=800, height=400).generate(text)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.show()
    else:
        print("Looking for text data in your dataset... Sorry, your dataset does not have suitable length text data. Therefore, sentiment analysis is not possible. Returning to the previous menu.")

def conduct_regression(df):
    """Perform linear regression analysis"""
    print("\nPerforming linear regression analysis")
    
    dependent_var = select_variable(df, "Please select the dependent variable (target variable):")
    if dependent_var == 'back':
        return
    elif dependent_var == 'exit':
        exit()

    print("Please select independent variables (feature variables), separated by commas:")
    independent_vars = []
    while True:
        var = select_variable(df, "Please select an independent variable:")
        if var == 'back':
            return
        elif var == 'exit':
            exit()
        
        if var:
            independent_vars.append(var)
            more = input("Do you want to add more independent variables? (y/n): ").lower()
            if more != 'y':
                break

    if dependent_var in df.columns and all(var in df.columns for var in independent_vars):
        X = df[independent_vars]
        y = df[dependent_var]

        # Add constant term
        X = sm.add_constant(X)

        # Normality test
        stat, p_value = shapiro(y.dropna())
        print(f"\nShapiro-Wilk test for the dependent variable '{dependent_var}': Statistic = {stat}, p-value = {p_value}")
        if p_value < 0.05:
            print("Null hypothesis (dependent variable follows a normal distribution) is rejected. The dependent variable does not follow a normal distribution.")
        else:
            print("Null hypothesis (dependent variable follows a normal distribution) is not rejected. The dependent variable follows a normal distribution.")

        # Create and fit the regression model
        model = sm.OLS(y, X).fit()
        
        # Output results
        print(model.summary())
    else:
        print("Selected variables do not exist in the dataset. Please check your input.")

def main():
    # User inputs dataset path
    file_path = input("Please enter the path to the dataset: ")
    
    # Load dataset
    df = load_dataset(file_path)
    
    if df is not None:
        inspect_data(df)

        while True:
            print("\nHow would you like to analyze your data?")
            print("1. Plot variable distribution")
            print("2. Perform ANOVA")
            print("3. Perform t-test")
            print("4. Perform Chi-square test")
            print("5. Perform regression analysis")
            print("6. Perform sentiment analysis")
            print("7. Exit")
            choice = input("Please enter your choice (1-7): ")

            if choice == '1':
                plot_variable_distribution(df)
            elif choice == '2':
                perform_anova(df)
            elif choice == '3':
                conduct_t_test(df)
            elif choice == '4':
                conduct_chi_square(df)
            elif choice == '5':
                conduct_regression(df)
            elif choice == '6':
                conduct_sentiment_analysis(df)
            elif choice == '7':
                print("Exiting the program.")
                break
            else:
                print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
