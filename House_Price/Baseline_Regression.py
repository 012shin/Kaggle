
import pandas as pd
import numpy as np
from scipy.stats import skew
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import seaborn as sns
def Overview_Missing_Values(df):
    """
    
    Input: DataFrame
    Output:
    -DataFrame with the number of missing values and the percentage of miss
    """
    
    # Get the null values
    null_values = df.isnull().sum()
    
    # Get the percentage of null values
    percentage_null_values = (null_values/df.shape[0]) * 100
    
    # Create the DataFrame with nulll values and the percentage of null_values
    null_df = pd.DataFrame({'Null Values':null_values, 'Percentage of Null': percentage_null_values})
    null_df = null_df.sort_values(by = 'Null Values', ascending = False)
    
    # # Create a heatmap of missing values
    # fig = px.imshow(df.isnull(), x = df.columns, y=df.index, color_continuous_scale= px.colors.sequential)
    # fig.show()
    
    # Only return the rows with null values > 0
    head = null_df['Null Values'][null_df['Null Values']>0].count()
    return null_df.head(head)


def Overview_Correlation(df,target):
    """
    Input:
        df: DataFrame
        target: target column
    output:DataFrame with the correlation of the target with the other features
    """    

    # Get the correlation of the target with the other features
    df = df.select_dtypes(exclude = ['object']) # Object data 제거
    corr = df.corr()
    target_corr = corr[target]
    
    # Create a DataFrame with the correlation of the target with the other features
    corr_df = pd.DataFrame(target_corr.sort_values(ascending = False))
    
    return corr_df.head(len(target_corr))



def Overview_Unique_Values(df,percentage = 0.85):
    """
    Input: DataFrame
    Output: print the unique values and the percentage of unique values
    """
    for i in df.columns:
        # We get the number of null values also, becaues it is not counted as a unique value
        sum_null = df[i].isnull().sum()
        max_value = df[i].value_counts().max()
        if max_value > sum_null:
            # Check if the unique value is more than 85% of the total values
            if max_value > df.shape[0]*percentage:
                print(f'{i} consist {(max_value/df.shape[0])*100}% out of unique values:{df[i].value_counts().index.max()}')
                print(df[i].value_counts())
                print('/////////////////////////////////////////////////////')
        else:
             # Check if the null values is more than 85% of the total values
                if sum_null > df.shape[0]*percentage:
                    print(f'{i} consist {(sum_null/df.shape[0])*100}% out of null values')
                    print(df[i].value_counts())
                    print('/////////////////////////////////////////////////////')
                    


def Overview_Inconsistencies(train,test,target):
    """
    Input: train: train dataset
            test: test dataset
            target: target column
    output:
        - print the inconsitencies between the train and test dataFrame
        - print the value counts in train
        - print the value counts in test
        - print the average sale price of the inconsistencies
    """
    for i in train.columns:
        if i == target:
            continue
            
        #Get the Unique values
        train_1 = train[i].value_counts().index.unique()
        test_1 = test[i].value_counts().index.unique()
            
        # Check if there are inconsistencies
        inconst_train = list(set(train_1)-set(test_1))
        inconst_test = list(set(test_1)- set(train_1))
        if len(inconst_train) > 0 or len(inconst_test) > 0:
            print(f'{i}')
            print(f'Only in train[{i}]: + {inconst_train[:5]}')
            print(f'Only in inconst_test[{i}]: + {inconst_test[:5]}')
            print("--------------------------------------------------")
            print("Train - Value Counts")
            print(train[i].value_counts())
            print("--------------------------------------------------")
            print("Test - Value Counts")
            print(test[i].value_counts())
            print("--------------------------------------------------")
            print('Average Sale Price')
            print(train.groupby(i)[target].mean())
            print("//////////////////////////////////////////////////")


def Overview_Categories(df):
    """
    Input: DataFrame
    Output: DataFrame with only categorical features, the unique values and  the number of unique value
    """
    # Categorical features
    df_categories = df.select_dtypes(include = ['object'])
    
    # Create DataFrame
    overview_categories = pd.DataFrame(df_categories.columns,columns = ['Feature'])
    
    # Add unique values and counts
    unique_values = []
    unique_counts = []
    for col in df_categories.columns:
        unique_values.append(df_categories[col].unique())
        unique_counts.append(len(df_categories[col].unique()))
    overview_categories['Categories'] = unique_values
    overview_categories['Number'] = unique_counts
        
    return overview_categories.head(len(overview_categories))




def Check_Correlation_ScatterPlot(df,target):
    """
    Input:
        df: DataFrame
        target: target column
    output:
            -Print the scatterplot between all the features and target
    """
    for i in range(0,len(df.columns),5):
        sns.pairplot(data = df,
                x_vars = df.columns[i:i+5],
                y_vars = [target])



def Check_Outliers(df,target,feature,num,operator):
    """
    Input:
        -df: dataFrame
        -target: target column 
        -feature: feature column
        -num: Number
        -operator: Operator
            
    output:
        -Scatterplot before removing outliers
        -Scatterplot after removing outliers
        -Correlation before removing outliers
        -Correlation after removing outliers
        -Difference between the correlation before and after removing outliers
    """

    #Scatterplot before removing outliers
    sns.relplot(kind = 'scatter', data = df,x =feature , y=target)
    before_corr = df[feature].corr(df[target])
    
    #Scatterplot after removing outliers
    a_df = df.copy()
    if operator == 'greater':
        a_df.loc[a_df[feature] > num,feature] = num
    elif operator == 'smaller':
        a_df.loc[a_df[feature] < num,feature] = num
    sns.relplot(kind = 'scatter',data = a_df, x= feature, y=target)    
    after_corr = a_df[feature].corr(a_df[target])
    #Difference between the correlation before and after removing outliers
    print(f'Before: {before_corr}')
    print(f'After: {after_corr}')
    print(f'Difference: {(after_corr-before_corr)/before_corr*100}%')



def Check_Skewness(df):
    """
    Input: dataframe
    output: Dataframe skewness of the features
    """
    # Get the skewness of the features
    skewed_features = df.apply(lambda x: x.skew()).sort_values(ascending = False)
    # Create the DataFrame
    skewness_table = pd.DataFrame({'Skew':skewed_features})
    return skewness_table.head(30)




def Log_Transform(train,test,target,threshold):
    """
    Input: 
        - train: training set
        - test: test set
        - target: target column
        - threshold: Skewness
    Output:
        - train: Training set with log transfromation
        - test: Test set with log transformation
    """
    # Get the skewness of the features
    skew_features = train.apply(lambda x: x.skew()).sort_values(ascending = False)
    skew_table = pd.DataFrame({'Skew':skew_features})    
    # Get the features with skewness greater than skewness
    skewness = skew_table[abs(skew_table)> threshold]
    # Create new DataFrames
    train_skew = train.copy()
    test_skew = test.copy()
    # Log Transfromation
    for i in skewness.index:
        if i == target:
            train_skew[i] = np.log1p(train_skew[i])
            continue
        train_skew[i] = np.log1p(train_skew[i])
        test_skew[i] = np.log1p(test_skew[i])
    return train_skew,test_skew



def Check_INF_values(df):
    """
    Input: Dataframe
    Output: Print the features with infinite values
    """
    # Check for infinite values
    for i in df.columns:
        if ((df[i] == np.inf) | (df[i] == -np.inf)).sum() > 0:
            print(i)
    print('Done')   
