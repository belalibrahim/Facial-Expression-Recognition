from core import load_testing_dataframe, view_image
from model import *

imgs, actual = load_testing_dataframe()

# model = get_model('CNN')
model = get_model('SVM')

model.load_weights(saved_weights_name)

predictions = model.predict_classes(imgs)

score = model.evaluate(imgs, actual)
print('Test Loss: ', score[0])
print('Test Accuracy: ', score[1]*100)

for i in range(len(imgs)):
    view_image(imgs[i], actual[i], predictions[i])
