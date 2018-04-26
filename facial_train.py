from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping

from core import *
from model import *

epochs = 1
batch_size = 20

images, predictions = load_training_dataframe()

checkpoint = ModelCheckpoint(saved_weights_name,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             )
csv_logger = CSVLogger('v.csv')
model = get_model()
early_stop = EarlyStopping(patience=3, monitor='val_loss')
history = model.fit(images, predictions,
                    batch_size=batch_size,
                    epochs=epochs,
                    shuffle=True,
                    validation_split=0.4,
                    callbacks=[checkpoint,
                               csv_logger,
                               early_stop])
