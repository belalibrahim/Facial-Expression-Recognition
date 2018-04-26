from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
from model import *
from core import *


epochs = 10

images, predictions = load_training_dataframe()

checkpoint = ModelCheckpoint(saved_weights_name,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             )
csv_logger = CSVLogger('ModelLogger.csv')
early_stop = EarlyStopping(patience=3, monitor='val_loss')

model = get_model('CNN')
# model = get_model('SVM')

history = model.fit(images, predictions,
                    epochs=epochs,
                    shuffle=True,
                    validation_split=0.2,
                    callbacks=[checkpoint,
                               csv_logger,
                               early_stop])

score = model.evaluate(images, predictions)
print('Train Loss: ', score[0])
print('Train Accuracy: ', score[1]*100)
