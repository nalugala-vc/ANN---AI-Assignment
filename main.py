from keras import regularizers
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense, Dropout
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

dataframe = read_csv("Nairobi_Office_Price_Ex.csv")
# print(dataframe.describe())

X = dataframe["SIZE"]
y = dataframe["PRICE"]

# print(X)
# print(y)

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2,random_state=20)
# print(X_train)
# print(X_test)

#scalling the data
scaler = StandardScaler()
scaler.fit(X_train.values.reshape(-1, 1))

X_train_scaled = scaler.transform(X_train.values.reshape(-1, 1))
X_test_scaled = scaler.transform(X_test.values.reshape(-1, 1))


model = Sequential()
model.add(Dense(64, input_dim=1, activation='relu'))
model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(optimizer='adam', loss='mean_squared_error',metrics=['mae'])
model.summary()

history = model.fit(X_train_scaled,y_train,validation_split=0.2,epochs=1000)
print(history)

#plot the training and validation accuracy at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1,len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel("Epochs")
plt.ylabel("loss")
plt.legend()
# plt.show()

acc = history.history['mae']
val_acc = history.history['val_mae']
plt.plot(epochs, acc, 'y', label='Training MAE')
plt.plot(epochs, val_acc, 'r', label='Validation MAE')
plt.title('Training and validation MAE')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
# plt.show()

# Create a scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(X, y, color='blue', alpha=0.5)  # Alpha controls the transparency of the points
plt.title('House Size vs. Price')
plt.xlabel('Size')
plt.ylabel('Price')
plt.show()

#predictions on test data
predictions = model.predict(X_test_scaled[:5])
print("Predicted Values : ",predictions)
print("Real values are: ", y_test[:5])

#predict size 100
size_to_predict = 100
scaled_size = scaler.transform([[size_to_predict]])
predicted_price = model.predict(scaled_size)

print("Predicted price for a house with size 100:", predicted_price)

#evaluating my model
mse_neural,mae_neural = model.evaluate(X_test_scaled, y_test)
print("Mean squared error from neural net:", mse_neural)
print("Mean absolute error from neural net: ", mae_neural)
