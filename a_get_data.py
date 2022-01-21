import os
from numpy import float64
import pandas as pd
import math

path = './data/hw_dataset/'


def data_to_csv(path):
    
	c_data = pd.DataFrame() 
    
	l =os.listdir(path)
	cols = ['X', 'Y', 'Z', 'Pressure', 'GripAngle', 'Timestamp', 'Test_ID']

	for folder in os.listdir(path):

		#print(folder)
		counter = 0
		if folder in ("control","parkinson"):
			for f in os.listdir(path+"/"+folder):
				filepath = path + folder + '/' + f
				#print(filepath)
				# Get the .txt out
				f = f.split('.', 1)[0]
				data= pd.read_csv(filepath, sep = ';', names = cols, dtype = float)
				data.insert(0,'ID', f) # to identify Pxxxx and Cxxxx : Parkinson and Control
				data.insert(1,'Group', counter) # to numerate the patient
				if folder == 'control':
					target_class = 0
				elif folder == 'parkinson':
					target_class = 1
				data.insert(len(data.columns),'target', target_class)
				counter +=1
                #Timestamp correction. 
                               
				data = data.sort_values('Timestamp')
				m = (data['Timestamp'].min())      
				data.loc[:,'Timestamp'] = data['Timestamp'] - m          
				# Creating polar coordinates
				# We aproximate the center of the spiral as the mean of the values.
				x0 = data['X'].mean()
				y0 = data['Y'].mean()
				vX = data['X'] - x0
				vY = data['Y'] - y0
				data.insert(len(data.columns)-1,'Radius', vX**2 + vY**2)
				#data['Angle'] = vY/vX
				data['Radius'].apply(lambda x: math.sqrt(x))
				#data['Angle'].apply(lambda x: math.atan(x))
                
                
                                         
                
                
                # Write the data to a single dataframe
				c_data =pd.concat([c_data, data])
				


	return c_data

def main():
	s = data_to_csv(path)
	s.to_csv('./data/consolidated_data.csv', index = False)

#print(s.groupby(['ID']).count())
#print(s)

if __name__ == '__main__':
	main()
