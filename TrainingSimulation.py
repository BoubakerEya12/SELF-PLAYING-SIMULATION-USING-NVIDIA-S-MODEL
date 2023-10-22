import matplotlib.pyplot as plt

print('Setting UP')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from utlis import *
from sklearn.model_slection import train_test_split


####Step1
path = 'myData'
data = importDataInfo(path)

####Step2
balanceData(data,display=False)

####Step3
imagesPath, steerings = loadData(path,data)

####Step4
xTrain, xVal, yTrain, yVal = train_test_split(imagesPath, steerings, test_size=0.2,random_state=10)
print('Total Training Images: ',len(xTrain))
print('Total Validation Images: ',len(xVal))

####Step5


####Step6

####Step7

####Step8
model = createModel()
model.summary()

####Step9
history = model.fit(batchGen(xTrain,yTrain,100,1),steps_per_epoch=300,epochs=10,
          validation_data=batchGen(xVal,yVal,100,0),validation_steps=200)

####Step10
model.save('model.h5')
print('Model Saved')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training','Validation'])
plt.ylim([0,1])
plt.title('Loss')
plt.xlabel('Epoch')
plt.show()