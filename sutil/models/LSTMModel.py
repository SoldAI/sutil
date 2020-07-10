from sutil.models.Model import Model
from tensorflow.keras.layers import Dense, Input, LSTM, Embedding, Dropout
from tensorflow.keras.layers import GlobalMaxPool1D
from tensorflow.keras.models import Model as KerasModel
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np

class LSTMModel(Model):

    @classmethod
    def fromDataset(cls, data, input_size, output_size, embedding_size):
        lb = preprocessing.LabelBinarizer()
        lb.fit(data.y)
        return cls(input_size, 'LSTM Model', 20000, embedding_size, output_size, lb)

    def __init__(self, maxlen, name = 'LSTM Model', max_features = 20000, embed_size = 128, number_of_classes = 1, binarizer = None):
        self.name = name
        self.binarizer = binarizer
        self.model = self.defineModel(maxlen, max_features, embed_size, number_of_classes)
        self.model.summary()

    def defineModel(self, maxlen, max_features = 20000, embed_size = 128, number_of_classes = 1):
        #Input layer
        inp = Input(shape=(maxlen, )) 
        x = Embedding(max_features, embed_size)(inp)
        #LSTM layer
        x = LSTM(60, return_sequences=True,name='lstm_layer')(x)
        x = GlobalMaxPool1D()(x)
        x = Dropout(0.1)(x)
        x = Dense(50, activation="relu")(x)
        x = Dropout(0.1)(x)
        x = Dense(number_of_classes, activation="softmax")(x)
        model = KerasModel(inputs=inp, outputs=x)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def trainModel(self, data, batch_size = 32, epochs = 2):
        binary_y = self.binarizer.transform(data.y)
        self.model.fit(data.getBiasedX(),binary_y, batch_size=batch_size, epochs=epochs, validation_split=0.1)
        
    def predict(self, X):
        return self.model.predict(X)
    
    def score(self, X, y, bias = True):
        if bias:
            X = np.hstack((np.ones((len(X), 1)), X))
        predictions = self.predict(X)
        self.evaluation = self.model.evaluate(X, y, batch_size=128)
        try:
            self.classification_report = classification_report(y, predictions)
            print(self.classification_report)
            self.accuracy = accuracy_score(y, predictions, normalize=False)
            self.recall = recall_score(y, predictions)
            self.precission = precision_score(y, predictions)
            self.f1 = f1_score(y, predictions)
            #self.roc = ModelROC(self, X, y, self.name)
        except:
            print("An exception occurred")

    def plotHistory(self):
        history = self.model.history
        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        # Get number of epochs\n",
        epochs = range(len(acc))

        # Plot training and validation accuracy per epoch\n",
        plt.plot(epochs, acc)
        plt.plot(epochs, val_acc)
        plt.title('Training and validation accuracy')
        plt.figure()

        # Plot training and validation loss per epoch\n",
        plt.plot(epochs, loss)
        plt.plot(epochs, val_loss)
        plt.title('Training and validation loss')
        plt.show()