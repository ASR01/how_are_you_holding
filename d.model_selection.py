import numpy    as np
import pandas   as pd
import matplotlib.pyplot as plt
import time


from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict, GroupKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, plot_confusion_matrix, confusion_matrix


##Import all Classifiers

from sklearn.tree          import DecisionTreeClassifier
from sklearn.ensemble      import RandomForestClassifier
from sklearn.ensemble      import ExtraTreesClassifier
from sklearn.ensemble      import AdaBoostClassifier
from sklearn.ensemble      import GradientBoostingClassifier
from sklearn.ensemble      import HistGradientBoostingClassifier
from xgboost               import XGBClassifier
from lightgbm              import LGBMClassifier
from catboost              import CatBoostClassifier


##Import Data


## Column enhancement


def model_analisys(df):

	df = pd.read_csv('./data/grouped_data.csv')

	y = df['42']
	g = df['0']
	x = df.drop(['0', '1', '42'], axis = 1)

#     #Remove the last identifier
#     y =np.ravel(y)
#     g = np.ravel(g)
    
	num_vars = x.select_dtypes(exclude=[object]).columns.values.tolist() 

	pn = Pipeline ( [ ('scaling', StandardScaler() ) ] )

	prep = ColumnTransformer(transformers=[('num', pn, num_vars)]) 

     #print('check point 0')

	# Models definition 

	tree_classifiers = {
        "Decision Tree": DecisionTreeClassifier(),
        "Extra Trees": ExtraTreesClassifier(),
        "Random Forest":RandomForestClassifier(),
        "AdaBoost": AdaBoostClassifier(),
        #"Skl GBM": GradientBoostingClassifier(),
        "Skl HistGBM": HistGradientBoostingClassifier() ,
        #"XGBoost": XGBClassifier() ,
        #"LightGBM":LGBMClassifier() ,
        #"CatBoost": CatBoostClassifier(verbose = False),
        }


	d1 = {}

	for n,v in tree_classifiers.items():
		p = Pipeline([('prep', prep),(n,v) ]) 
		d1.update({n:p})
        #tree_classifiers.update({n:p})
        #print(k)

	tree_classifiers = {name: make_pipeline(prep, model) for name, model in tree_classifiers.items()}

#     #print('check point 1')

  ## Models beauty contest

#    skf = StratifiedKFold(n_splits = 3, shuffle = True, random_state = 0)
	gkf = GroupKFold(n_splits = 3)
	gkf = gkf.get_n_splits(x, y, g)

	results = pd.DataFrame({'Model': [], 'Accuracy': [], 'Bal Acc.': [], 'Time': []})

	for model_name, model in tree_classifiers.items():
        
        #print('check point 2')
		start_time = time.time()
		print(model_name)
		pred = cross_val_predict(model, x, y,cv=gkf) #gkf

		total_time = time.time() - start_time
		results = results.append({
      							"Model":    model_name,
								"Accuracy": accuracy_score(y, pred)*100,
								"Bal Acc.": balanced_accuracy_score(y, pred)*100,
								"Time":     total_time},
								ignore_index=True)
                          

    #print('check point 3')
       
	results_ord = results.sort_values(by=['Accuracy'], ascending=False, ignore_index=True)
	results_ord.index += 1 
	results_ord.style.bar(subset=['Accuracy', 'Bal Acc.'], vmin=0, vmax=100, color='#5fba7d')

	results_ord.to_csv('results.csv')
    
	return(results_ord) 


def main():
	file = './data/grouped_data_timed.csv'
	res = model_analisys(file)
	print(res)

if __name__ == '__main__':
	main()
 



