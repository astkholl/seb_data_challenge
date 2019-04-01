import numpy as np
import pandas as pd



def get_check_data(files,names,check_missing=True):

	if len(files) != len(names):
		raise Exception('The number of files should equal the number of desired dataframes!')

	for i in len(files):
	
		names[i] = pd.read_csv(files[i])
		
		if check_missing == True:
		
			#check for missing values and print column heads for reference

			missing_val_count_by_column = (names[i].isnull().sum())
			print("Missing data from ",names[i],": ",missing_val_count_by_column[missing_val_count_by_column > 0])

	return names
