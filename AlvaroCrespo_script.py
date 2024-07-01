import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import seaborn as sns
from IPython.display import display

import datetime

# General Government Net debt ( https://macrosynergy.com/academy/notebooks/general-government-finance-ratios/ )
# General Government Overall Balance ( https://macrosynergy.com/academy/notebooks/general-government-finance-ratios/ )
# PPI ( https://macrosynergy.com/academy/notebooks/producer-price-inflation/ )
# Merchandise Trade Balance: ( https://macrosynergy.com/academy/notebooks/external-balance-ratios/ )
# Plus returns for generic 5-year government bonds: ( https://macrosynergy.com/academy/notebooks/government-bond-returns/ )

#For Japan and the UK.

#The task is to ingest the data, understand it, and show us with some pandas and python code what you would do if you were to study the relationships 
# between the above data and the government bond returns. 

#Could you please also share with us a small paragraph that explains:

# -The value you find in such data
# -Your thinking around the studying of the relationships.

# Just one note: no need for anything overly complicated, we just want to assess how you interact with the data from a coding perspective. 
# It would be great if you could send it to us by next Wednesday and you can take some inspiration from here: https://macrosynergy.com/research/ . 
# Don’t use the Macrosynergy package.


## PART 1 - INITIAL DATA

# Read CSV and load data to a dataframe
file_path = 'data/Data.csv'
data = pd.read_csv(file_path)

# Display the cleaned data and columns to check dataframe form
data.head()
data.info()

print(data.columns)

# Convert the Date column to datetime format
data['Date'] = pd.to_datetime(data['Date'], format='%d-%b-%y')

# Filter data for Japan and the UK

japan_data = data[['Date', ' General government debt, % of GDP: net JPY',
                   'General government balance, % of GDP: overall JPY', 
                   'Main producer price index: %oya, 3mma Japan', 
                   ' Merchandise trade balance ratio change (sa): 3M/3M Japan',
                   ' Generic government bond returns: 5-year maturity Japan']]

uk_data = data[['Date', 'General government debt, % of GDP: net GBP', 
                ' General government balance, % of GDP: overall GBP', 
                'Main producer price index: %oya, 3mma GBP',
                ' Merchandise trade balance ratio change (sa): 3M/3M GBP',
                'Generic government bond returns: 5-year maturity GBP']]

# Rename columns for simplification
japan_data.columns = ['Date', 'Net Debt', 'Overall Balance', 'PPI', 'Trade Balance', 'Bond Returns']
uk_data.columns = ['Date', 'Net Debt', 'Overall Balance', 'PPI', 'Trade Balance', 'Bond Returns']


# FIGURE 1) Plotting initial historical trends JAPAN & UK without correcting data
plt.figure(figsize=(14, 10))

# Plot for UK data
plt.subplot(2, 1, 1)
plt.plot(uk_data['Date'], uk_data['Net Debt'], label='Net Debt')
plt.plot(uk_data['Date'], uk_data['Overall Balance'], label='Overall Balance')
plt.plot(uk_data['Date'], uk_data['PPI'], label='PPI')
plt.plot(uk_data['Date'], uk_data['Trade Balance'], label='Trade Balance')
plt.ylabel('Indicators [%GDP]', fontsize=14)
plt.xlabel('Date', fontsize=14)
plt.legend(loc='upper left', fontsize = 14)

# Secondary y-axis for Bond Returns
ax1 = plt.gca()
ax2 = ax1.twinx()
ax2.plot(uk_data['Date'], uk_data['Bond Returns'], color='purple', label='Bond Returns')
ax2.set_ylabel('Bond Returns [%]', fontsize=14)
ax2.legend(loc='upper right')
ax1.tick_params(axis='both', which='major', labelsize=14)
ax2.tick_params(axis='both', which='major', labelsize=14)
plt.title('UK - Economic Indicators and Bond Returns')

# Plot for Japan data
plt.subplot(2, 1, 2)
plt.plot(japan_data['Date'], japan_data['Net Debt'], label='Net Debt')
plt.plot(japan_data['Date'], japan_data['Overall Balance'], label='Overall Balance')
plt.plot(japan_data['Date'], japan_data['PPI'], label='PPI')
plt.plot(japan_data['Date'], japan_data['Trade Balance'], label='Trade Balance')
plt.ylabel('Indicators [%GDP]', fontsize=14)
plt.xlabel('Date', fontsize=14)
plt.legend(loc='upper left', fontsize = 14)

# Secondary y-axis for Bond Returns
ax3 = plt.gca()
ax4 = ax3.twinx()
ax4.plot(japan_data['Date'], japan_data['Bond Returns'], color='purple', label='Bond Returns')
ax4.set_ylabel('Bond Returns [%]', fontsize=14)
ax4.legend(loc='upper right')
ax3.tick_params(axis='both', which='major', labelsize=14)
ax4.tick_params(axis='both', which='major', labelsize=14)
plt.title('Japan - Economic Indicators and Bond Returns')

plt.tight_layout()
plt.savefig('historical_trends.png')



## PART 2 - DATA ANALYSIS

# Dctionary to store the first non-null dates
first_non_null_dates = {}
# Iterate over columns and find first non-null value
for column in data.columns[1:]:
    first_valid_index = data[column].first_valid_index()
    first_non_null_dates[column] = data.loc[first_valid_index, 'Date']

# Convert dict to a DataFrame for better visualization
first_non_null_dates_df = pd.DataFrame(list(first_non_null_dates.items()), columns=['Column', 'First Non-Null Date'])
display(first_non_null_dates_df)

# Retrieve the start times for the UK and Japan bond returns
start_time_UK_bonds = first_non_null_dates['Generic government bond returns: 5-year maturity GBP']
start_time_japan_bonds = first_non_null_dates[' Generic government bond returns: 5-year maturity Japan']

# Retrieve the end times for the UK and Japan bond returns
last_valid_UK_index = uk_data['Bond Returns'].last_valid_index()
end_time_UK_bonds = uk_data.loc[last_valid_UK_index, 'Date'] if last_valid_UK_index is not None else None
last_valid_Japan_index = uk_data['Bond Returns'].last_valid_index()
end_time_japan_bonds = uk_data.loc[last_valid_Japan_index, 'Date'] if last_valid_Japan_index is not None else None

# Resize the UK and Japan dataframes
uk_data_filtered = uk_data[(uk_data['Date'] >= start_time_UK_bonds) & (uk_data['Date'] <= end_time_UK_bonds)]
japan_data_filtered = japan_data[(japan_data['Date'] >= start_time_japan_bonds) & (japan_data['Date'] <= end_time_japan_bonds)]


# Count the number of NaN values per column in each DataFrame
uk_nan_counts = uk_data_filtered.drop(columns=['Date']).isna().sum()
japan_nan_counts = japan_data_filtered.drop(columns=['Date']).isna().sum()

# Combine the counts into a single DataFrame for visualization
nan_counts_df = pd.DataFrame({
    'UK': uk_nan_counts,
    'Japan': japan_nan_counts
}).T

# FIGURE 2)  Plot the missing values 
plt.figure(figsize=(10, 4))
sns.heatmap(nan_counts_df, annot=True, cmap='YlOrRd', cbar=False, linewidths=.2)
plt.title('Missing days of data from: UK 2007-04-02 and Japan 2012-03-15')
plt.xlabel('Economic Indicators')
plt.ylabel('Country')
plt.xticks(rotation=0)

plt.tight_layout()
plt.savefig('missing_values_matrix.png')


# FIGURE 3) Plot time evolution of all filtered data during analysis time
plt.figure(figsize=(14, 12))

# Plot for UK data
plt.subplot(2, 1, 1)
line1, = plt.plot(uk_data_filtered['Date'], uk_data_filtered['Net Debt'], label='Net Debt')
line2, = plt.plot(uk_data_filtered['Date'], uk_data_filtered['Overall Balance'], label='Overall Balance')
line3, = plt.plot(uk_data_filtered['Date'], uk_data_filtered['PPI'], label='PPI')
line4, = plt.plot(uk_data_filtered['Date'], uk_data_filtered['Trade Balance'], label='Trade Balance')
plt.ylabel('Indicators [%GDP]', fontsize=14)
plt.xlim(start_time_UK_bonds, end_time_UK_bonds)

# Secondary y-axis for Bond Returns
ax1 = plt.gca()
ax2 = ax1.twinx()
line5, = ax2.plot(uk_data_filtered['Date'], uk_data_filtered['Bond Returns'], color='purple', label='Bond Returns')
ax2.set_ylabel('Bond Returns [%]', fontsize=14)
ax1.tick_params(axis='both', which='major', labelsize=14)
ax2.tick_params(axis='both', which='major', labelsize=14)
plt.title('UK - Economic Indicators and Bond Returns')

# Plot for Japan data
plt.subplot(2, 1, 2)
plt.ylabel('Indicators [%GDP]', fontsize=14)
plt.xlabel('Date', fontsize=14)
plt.xlim(start_time_japan_bonds, end_time_japan_bonds)

# Secondary y-axis for Bond Returns
ax3 = plt.gca()
ax4 = ax3.twinx()
line10, = ax4.plot(japan_data_filtered['Date'], japan_data_filtered['Bond Returns'], color='purple', label='Bond Returns')
ax4.set_ylabel('Bond Returns [%]', fontsize=14)
ax3.tick_params(axis='both', which='major', labelsize=14)
ax4.tick_params(axis='both', which='major', labelsize=14)
plt.title('Japan - Economic Indicators and Bond Returns')

# Legend
fig = plt.gcf()
lines = [line1, line2, line3, line4, line5]
labels = [line1.get_label(), line2.get_label(), line3.get_label(), line4.get_label(), line5.get_label()]
fig.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=5, fontsize=12)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('historical_trends_v2.png')


## PART 3 - ANALYSIS OF ECONOMIC DATA

# Function to calculate cumulative returns for Bond returns
def calculate_cumulative_returns(df, column):
    daily_returns = df[column] / 100
    cumulative_returns = (1 + daily_returns).cumprod() - 1  # apply cumulative product
    return cumulative_returns * 100 

# Calculate cumulative returns for UK and Japan bond returns
uk_data_filtered['Cumulative Bond Returns'] = calculate_cumulative_returns(uk_data_filtered, 'Bond Returns')
japan_data_filtered['Cumulative Bond Returns'] = calculate_cumulative_returns(japan_data_filtered, 'Bond Returns')


# FIGURE 4) Cumulative Bond returns over time
plt.figure(figsize=(14, 8))

plt.subplot(2, 1, 1)
plt.plot(uk_data_filtered['Date'], uk_data_filtered['Cumulative Bond Returns'], label='UK Cumulative Bond Returns')
plt.title('UK - Cumulative Bond Returns 5-year maturity')
plt.ylabel('Cumulative Returns [%]', fontsize =14)

plt.subplot(2, 1, 2)
plt.plot(japan_data_filtered['Date'], japan_data_filtered['Cumulative Bond Returns'], label='Japan Cumulative Bond Returns')
plt.title('Japan - Cumulative Bond Returns 5-year maturity')
plt.xlabel('Date', fontsize =14)
plt.ylabel('Cumulative Returns [%]', fontsize =14)

plt.tight_layout()
plt.savefig('cumulative_bond_returns.png')


## Calculate volatility

# Function to calculate volatility
def calculate_volatility(df, column):
    df[f'{column} Decimal'] = df[column] / 100  # Convert to decimal
    daily_volatility = df[f'{column} Decimal'].std()  # Daily standard deviation
    annual_volatility = daily_volatility * np.sqrt(252)  # Annualized volatility considering 252 trading days
    return annual_volatility

# Calculate Volatility
volatility_uk = calculate_volatility(uk_data_filtered, 'Bond Returns')
volatility_japan = calculate_volatility(japan_data_filtered, 'Bond Returns')

print(f"Annualized Volatility for UK Bond Returns: {volatility_uk:.2%}")
print(f"Annualized Volatility for Japan Bond Returns: {volatility_japan:.2%}")

# Check the range and basic statistics of bond returns
print(uk_data_filtered['Bond Returns'].describe())
print(japan_data_filtered['Bond Returns'].describe())



## Figure 5) Calculate and Visualize correlations

# Calculate correlation matrix for Japan & UK
japan_corr = japan_data_filtered[['Net Debt', 'Overall Balance', 'PPI', 'Trade Balance', 'Bond Returns', 'Cumulative Bond Returns']].corr()
uk_corr = uk_data_filtered[['Net Debt', 'Overall Balance', 'PPI', 'Trade Balance', 'Bond Returns', 'Cumulative Bond Returns']].corr()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

# Generate a mask for the upper triangles
mask_japan = np.triu(np.ones_like(japan_corr, dtype=bool))
mask_uk = np.triu(np.ones_like(uk_corr, dtype=bool))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap for Japan & UK with the mask and correct aspect ratio
sns.heatmap(japan_corr, mask=mask_japan, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True, fmt=".2f", ax=ax1)
ax1.set_title('Japan - Correlation Matrix')

sns.heatmap(uk_corr, mask=mask_uk, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True, fmt=".2f", ax=ax2)
ax2.set_title('UK - Correlation Matrix')

plt.tight_layout()
plt.savefig('correlation_matrix_v2.png')


plt.show()


