
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
import scipy.stats as sps

def get_customer_info(customers,transactions,districts):

	#set date of transactions to datetime type using month and year

	transactions['DATE'] = pd.to_datetime(transactions['DATE'],format='%Y%m%d')
	transactions['YEARMONTH'] = transactions['DATE'].map(lambda x: 100*x.year + x.month)
	transactions['YEARMONTH'] = pd.to_datetime(transactions['YEARMONTH'],format='%Y%m')

	#count transactions for each customer

	index_count = pd.Index(transactions['ACCOUNT_ID']).value_counts()
	print('Average number of transactions per customer = ', np.mean(index_count.values))
	index_count = index_count.sort_index()
	customers.index = customers['ACCOUNT_ID']
	customers['TRANSACTION_NUM'] = index_count

	#calculate active time for each customer

	startdate = transactions.groupby('ACCOUNT_ID')['DATE'].min()
	time_active = transactions.groupby('ACCOUNT_ID')['DATE'].max() - startdate
	customers['TIME_ACTIVE'] = time_active.dt.days

	#calculate the monthly average balance/withdrawal/credit for each customer and add to the customer dataframe

	mean_balance = transactions.groupby(['ACCOUNT_ID','YEARMONTH'], as_index=False)['BALANCE'].mean()
	monthly_balance = mean_balance.groupby('ACCOUNT_ID')['BALANCE'].mean()
	
	mean_transaction = transactions.groupby(['ACCOUNT_ID','YEARMONTH','TYPE'], as_index=False)['AMOUNT'].mean()

	mean_withdrawal = mean_transaction[mean_transaction['TYPE'] == 'WITHDRAWAL']
	monthly_withdrawal = mean_withdrawal.groupby('ACCOUNT_ID')['AMOUNT'].mean()

	mean_credit = mean_transaction[mean_transaction['TYPE'] == 'CREDIT']
	monthly_credit = mean_credit.groupby('ACCOUNT_ID')['AMOUNT'].mean()

	customers['AVG_MONTHLY_BAL'] = monthly_balance
	customers['AVG_MONTHLY_WITH'] = monthly_withdrawal
	customers['AVG_MONTHLY_CRED'] = monthly_credit
	
	#calculate average monthly number of transactions
	
	trans_means = transactions.groupby(['ACCOUNT_ID','YEARMONTH'], as_index=False)['TYPE'].count()
	monthly_trans = trans_means.groupby('ACCOUNT_ID')['TYPE'].mean()
	customers['AVG_MONTHLY_TRANS'] = monthly_trans
	
	#calculate average monthly change in balance

	transactions_sorted = transactions.sort_values(by=['ACCOUNT_ID', 'YEARMONTH'])
	transactions['BALANCE_CHANGE'] = transactions_sorted.groupby(['ACCOUNT_ID','YEARMONTH'])['BALANCE'].diff().fillna(0)
	balance_change = transactions.groupby(['ACCOUNT_ID','YEARMONTH'],as_index=False)['BALANCE_CHANGE'].mean()

	mean_change = balance_change.groupby('ACCOUNT_ID')['BALANCE_CHANGE'].mean()

	customers['AVG_BALANCE_CHANGE'] = mean_change
	
	#add district's average salaries to the customers df
	
	districts.index = districts['DISTRICT_ID']
	customers['AVG_SALARY'] = customers['DISTRICT_ID'].map(districts['AVG_SALARY'])
	
	#calculate customers' average monthly excess (credits - withdrawals)
	
	customers['AVG_EXCESS'] = customers['AVG_MONTHLY_CRED'] - customers['AVG_MONTHLY_WITH']
	
	#calculate customers' relative monthly withdrawals (relative to monthly balance)
	
	customers['REL_WITH'] = customers['AVG_MONTHLY_WITH']/customers['AVG_MONTHLY_BAL']
	
	#calculate customers' average end of month balance

	end_balance = transactions.groupby(['ACCOUNT_ID','YEARMONTH'],as_index=False)['BALANCE'].last()
	monthly_end = end_balance.groupby('ACCOUNT_ID')['BALANCE'].mean()
	
	customers['MONTHLY_END_BALANCE'] = monthly_end
	
	#calculate customers' average yearly income
	
	transactions['YEAR'] = transactions['YEARMONTH'].dt.year
	yearly_total = transactions.groupby(['ACCOUNT_ID','YEAR','TYPE'], as_index=False)['AMOUNT'].sum()
	yearly_credit = yearly_total[yearly_total['TYPE'] == 'CREDIT']
	yearly_mean = yearly_credit.groupby('ACCOUNT_ID')['AMOUNT'].mean()

	customers['YEARLY_CREDIT'] = yearly_mean
	
	now = pd.Timestamp(datetime.now())
	customers['BIRTH_DT'] = pd.to_datetime(customers['BIRTH_DT'], format='%Y%m%d') 
	customers['AGE'] = (now - customers['BIRTH_DT']).astype('<m8[Y]')
	
	
	return customers,transactions,districts









