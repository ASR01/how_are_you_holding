import pandas as pd
import numpy as np



def prepare_timely_data(data, window):
    
	#group the data in bathce fo the windows length 
    
#	print('Original Data', data.shape)
	data.insert(2, 'S_ID', 0)
#	data = data.head(50)
#	print('Used data ', data.shape)
	#print(data)
	x = []
	data_arr = data.to_numpy()
	id = data_arr[0,0]
	counter = -1
	
	#print(id,data_arr.shape, data_arr[:5])
	#First loop 
	for i in range(0, data_arr.shape[0]):
		#if the dataset from the same patient and still in range	
		if id ==  data_arr[i,0] and i != data_arr.shape[0] - 1 :
			counter += 1
		else:
            
			if counter != window:
				rest = int(round((counter/window - counter//window)*window,0))
				#print(rest)
			data_arr[(i-rest)-1:i,1] = -1
			counter = 0
        
		ISG = counter//window
		data_arr[i,1] = ISG       
		id = data_arr[i,0]
		#print(data_arr[i,0:2])
        
	data_arr = np.delete(data_arr,np.where(data_arr[:,1] == -1), 0 ) 
    #data_arr = np.delete(data_arr, 287246, 0 ) 
       
	x = []

	# Loop to group the items in lines. 
	for i in range(0, data_arr.shape[0]-1, window):
			seq = np.zeros((window, data.shape[1]-4),dtype = 'object')
			for j in range(window):
				seq[j] = data_arr[i+j,3:-1]
				if j ==0:
					trail = data_arr[i,0:2]
					tail = data_arr[i,-1]
			#print('seq', seq, 'trail',  trail, 'tail', tail)		
			seq = seq.flatten()
			seq = np.append(trail, seq)
			seq = np.append(seq, tail)
			x.append(seq)

	x = np.array(x)
	#print('x', x.shape)


	return x

def main():

	df = pd.read_csv('./data/consolidated_data.csv')
	window = 10
	x = prepare_timely_data(df, window)   

	x = pd.DataFrame(x)
	x.to_csv('./data/grouped_data.csv', index = False)
	
	
if __name__ == '__main__':
    main()




