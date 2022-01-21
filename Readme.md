# Introduction 

Usin the data provided by the University of California Irvine under their paper *Improved spiral test using digitized graphics tablet for monitoring parkinsons disease* the data related to this are available under http://archive.ics.uci.edu/ml/datasets/ under the
name of *Parkinson+Disease+Spiral+Drawings+Using+Digitized+Graphics+Tablet*

## Analysis

The data is grouped in two folwo der with the Parkison patients and with the control ones.

In the data we can see the following data:
- x
- y
- z
- pressure
- grip angle
- timestamp
- test_id
It can be useful to be able to check the radius and the angle ot every point in the case of drawing spirals. Because the complexity of the angle due to the 2PI border, we opt to not calculate it.

So if we run the file **a_get_data** we get the consolidated data file where we have the original data consolidated in a file with new columns:
- ID -> File name without extension
- Coder -> Batch
- Radius -> The radius according to a center determined by the mean of the coordinates x and y.
 
 If we run the file **b_data_analysis.py** we can check the data correlation and see the data file represented.
## Preparation for Time Analysis

All the data is compacted in lines. So many original rows in a new row as the window factor. 

This is made with the **c_time_prep.py**


##  Selection of models

The following module **d_model_selection.py** allow us to check different models in order to select the best performing one and use it in a different module and try to optimise it. Thsi analysis is made with no grid.

##  Model Data
With the final module **e_final_model.py** it is possible to train a model trying to parametrize the data and saving it for future use in a file.

