from core import load_testing_dataframe, view_image
from model import get_model, saved_weights_name

imgs, expected = load_testing_dataframe()

model = get_model()
model.load_weights(saved_weights_name)

predictions = model.predict_classes(imgs)
print(predictions.shape)
print(predictions)
for img, prediction in zip(imgs, predictions):
    view_image(img, prediction)
