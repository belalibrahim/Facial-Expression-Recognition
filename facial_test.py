from core import load_testing_dataframe, view_image
from model import get_model, saved_weights_name

imgs = load_testing_dataframe()

model = get_model()
model.load_weights(saved_weights_name)

testing_imgs = imgs[50:70]

predictions = model.predict(testing_imgs)
print(predictions.shape)
for img, prediction in zip(testing_imgs, predictions):
    view_image(img, prediction)
