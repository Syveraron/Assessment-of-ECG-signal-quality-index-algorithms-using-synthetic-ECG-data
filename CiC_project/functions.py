import matplotlib.pyplot as plt

def plotting_df(df, variable):
    plt.figure(figsize=(10, 6))
    plt.plot(df['Noise Type'], df['SQA1'], color='blue', label='SQA1')
    plt.plot(df['Noise Type'], df['SQA2'], color='orange', label='SQA2')
    plt.plot(df['Noise Type'], df['SQA3'], color='green', label='SQA3')
    plt.plot(df['Noise Type'], df['SQA4'], color='purple', label='SQA4')

    # Dotted line at 0.5 for probability
    plt.axhline(y=0.5, color='red', linestyle='-.')

    # Add labels and title
    plt.xlabel('variable')
    plt.ylabel('Probability of returning acceptable result')
    plt.title(variable)
    plt.legend()

    # Display the plot
    plt.show()

import pandas as pd


def first_below_05(df, col_name):
    """
    Find the first instance of a value below 0.5 in the specified column
    and return the adjacent value from the 'Noise Type' column.
    
    """
    # Check if the 'Noise Type' column exists
    if 'Noise Type' not in df.columns:
        return None
    
    # Find the first index where the value in the specified column is below 0.5
    below_05_idx = next((idx for idx, val in df[col_name].items() if val < 0.5), None)
    
    # If no value below 0.5 is found, return None
    if below_05_idx is None:
        return None
    
    # Return the adjacent value from the 'Noise Type' column
    return df.loc[below_05_idx, 'Noise Type']