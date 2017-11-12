import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation

from keras.layers.advanced_activations import LeakyReLU

#1. Create some data
#See example at https://medium.com/technology-invention-and-more/how-to-build-a-simple-neural-network-in-9-lines-of-python-code-cc8f23647ca1
X = np.array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
Y = np.array([[0, 1, 1, 0]]).T

#2. Create a model object
model = Sequential()
model.add(Dense(24,input_dim=3))#,activation='relu'))
model.add(LeakyReLU(alpha=.01))
model.add(Dense(24))
model.add(LeakyReLU(alpha=.01))
model.add(Dense(1,activation='sigmoid'))
#Note, models will often have 2-3 layers, not just one. 
#The standard example had model.add(Dense(12,input_dim=3,activation='relu'))
#as a layer, with relu. Using more neurons (24 each layer), and a Leakyrelu, the discrimination
#improved from the 0=.44, 1 = .55, to 0=.06, 1 = .95 range.

#3. Compile the model. Note the output here is assumed to be binary, indicating
#the type of 
#from. https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#4. Fit the model
model.fit(X, Y, epochs=350, 
                batch_size=4, 
                verbose=0) #suppresses output.
        #batch size is how many of the whole set is fed into the model at an iteration. 
        #More epochs = improved accuracy too -- 150 -> 350 by order 10.
#5. Evaluate the model
scores = model.evaluate(X, Y)
#print('\n{0:s}: {1:.2f}'.format(model.metrics_names[1],scores[1]))

#6. Make some predictions on the original data
predictions = model.predict(X)
# round predictions
preds = [x[0] for x in predictions]
print('Predictions :',preds)

#Test on new data...
new_x = np.array([[1,0,0]])
print('Predict a new x (should be = 1):',model.predict(new_x))