from typing import no_type_check
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def corr_analysis(df):

	# Analysis of the correlation of the features

    x = df.drop('target', axis = 1)
    #print(x)
    plt.figure(figsize=(15, 10))
    matrix = np.triu(x.corr())
    sns.heatmap(x.corr(), annot=True, linewidth=.8, mask=matrix, cmap="PuBuGn")
    plt.title("Correlation Matrix without target field", fontsize=20)
    plt.savefig('./figures/correlation.png')
    plt.show()
    
    
    return


def draw_analysis(df):
    
    x = df.drop('target', axis = 1)
    
    x = x[x['Timestamp'] == 0]
    

    #print(x)
    plt.figure(figsize=(15, 10))
    sns.scatterplot(x['X'],x['Y'] )
    
    plt.title("DrawPlot", fontsize=20)
    #plt.savefig('./figures/correlation.png')
    plt.show()
    
    
    return

def draw_analysis_2(df):
    
    x = df.drop('target', axis = 1)
    
    
    #print(x)
    plt.figure(figsize=(15, 10))
    sns.scatterplot(x['X'],x['Y'] )
    
    plt.title("DrawPlot", fontsize=20)
    #plt.savefig('./figures/correlation.png')
    plt.show()
    
    
    return

def main():
	df = pd.read_csv('./data/consolidated_data.csv')
	corr_analysis(df)
	#draw_analysis(df)
	draw_analysis_2(df)
 
if __name__ == '__main__':
     main()