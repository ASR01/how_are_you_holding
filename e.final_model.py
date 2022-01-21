import pandas as pd
import pickle as pk
import numpy as np
import time



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer

from catboost              import CatBoostClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict, GroupKFold

from sklearn.metrics import accuracy_score, balanced_accuracy_score, plot_confusion_matrix, confusion_matrix

from sklearn.pipeline      import Pipeline, make_pipeline

# from sklearn.tree          import DecisionTreeClassifier
# from sklearn.ensemble      import RandomForestClassifier
from sklearn.ensemble      import ExtraTreesClassifier
# from sklearn.ensemble      import AdaBoostClassifier
# from sklearn.ensemble      import GradientBoostingClassifier
# from sklearn.experimental  import enable_hist_gradient_boosting # Necesary for HistGradientBoostingClassifier
# from sklearn.ensemble      import HistGradientBoostingClassifier
# from xgboost               import XGBClassifier
# from lightgbm              import LGBMClassifier


def run_model(file):

	df = pd.read_csv(file)

	y = df['42']
	g = df['0']
	x = df.drop(['0', '1', '42'], axis = 1)
    
	num_vars = x.select_dtypes(exclude=[object]).columns.values.tolist() 
	pn = Pipeline ( [ ('scaling', StandardScaler() ) ] )

	prep = ColumnTransformer(transformers=[('num', pn, num_vars)]) 

	gkf = GroupKFold(n_splits = 3)
	gkf = gkf.get_n_splits(x, y, g)

	# Select Model
  
	#model =  CatBoostClassifier(verbose = False)
	#model = RandomForestClassifier()
	model = ExtraTreesClassifier()
	#model = AdaBoostClassifier ()
	#model = GradientBoostingClassifier
	#model =  enable_hist_gradient_boosting # Necesary for HistGradientBoostingClassifier
	#model = HistGradientBoostingClassifier()
	#model = XGBClassifier()
	#model = LGBMClassifier()
    
	start_time = time.time()
    
	pipe = make_pipeline(prep, model)

	pred = cross_val_predict(pipe, x, y,cv=gkf) #gkf

	total_time = time.time() - start_time
	results = {
                              "Accuracy": accuracy_score(y, pred)*100,
                              "Bal Acc.": balanced_accuracy_score(y, pred)*100,
                              "Time":     total_time}

       
    
	print(results)
    
    #acc = accuracy_score(y_test, y_pred)
    #print(acc)
    #print('The confusion matrix is a follows:')
    #print(confusion_matrix(y_test, y_pred))
    #plot_confusion_matrix(model, X_test, y_test)
    
	save = input('Do you want to save the model? Y/N:  ')
	if save in ('Y', 'y'):
		model_name = './model/optimal_model.pkl'
		with open(model_name, 'wb') as file:
			pk.dump(model, file)
		print("The model has been saved in ./model/optimal_model")

      

        
## Test run
def main():

	file = './data/grouped_data.csv'
	run_model(file)

if __name__ == '__main__':
    main()

