# -*- coding: utf-8 -*-
"""
Created on Sat May  1 09:57:55 2021

@author: harro
"""

def build_model(X, learning_rate):
    from tensorflow.keras import models
    from tensorflow.keras import layers 
    from tensorflow.keras import optimizers
    model = models.Sequential()
    model.add(layers.Dense(16, input_shape=(X.shape[1],), activation='relu'))
    model.add(layers.Dense(8, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(4, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(1, activation='sigmoid'))
    
    lr = learning_rate
    adam = optimizers.Adam(learning_rate=lr)
    
    model.compile(optimizer=adam,
                  loss='binary_crossentropy',
                  metrics=['AUC'])
    return model

def sklearn_compatible_model():
    from tensorflow.keras import models
    from tensorflow.keras import layers 
    from tensorflow.keras import optimizers
    model = models.Sequential()
    model.add(layers.Dense(16, input_dim=634, activation='relu'))
    model.add(layers.Dense(8, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(4, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(1, activation='sigmoid'))
    
    model.compile(optimizer=optimizers.Adam(),
                  loss='binary_crossentropy',
                  metrics=['AUC'])
    return model

def build_report(X_train, y_train, X_val, y_val, epochs, batch_size):
    learning_curves = {}
    history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                      verbose=1, 
                      validation_data=(X_val, y_val))
    learning_curves = history.history
    
    # Create report
    lc = pd.DataFrame(learning_curves)
    print("Max val score: {:.2f}%".format(lc.iloc[:,3].max()*100))
    lc.plot(lw=2,style=['b:','r:','b-','r-']);
    plt.xlabel('Epochs');
    plt.show()
    
    print(model.summary)
    pass

def evaluate_model(X_test, y_test):
    score = model.evaluate(X_test, y_test, verbose = 0)
    return score[1] # index 1 has the accuracy, index 0 the loss
    
if __name__ == "__main__":
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split, StratifiedKFold
    from roc_auc_curve import plot_roc_nn
    
    # Import function to preprocess data
    import preprocessing_data
    X, y = preprocessing_data.preprocessing()
    
    # We do an 80-20 split for the training and test set, and then again a 80-20 split into training and validation data
    X_train_all, X_test, y_train_all, y_test = train_test_split(X,y, stratify=y, train_size=0.8, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_all,y_train_all, stratify=y_train_all, train_size=0.8, random_state=42)
    
    # Build the model
    model = build_model(X_train, learning_rate=0.01)
    
    EPOCHS = 5
    BATCH_SIZE = 32
    
    build_report(X_train, y_train, X_val, y_val, EPOCHS, BATCH_SIZE)

    test_auc = evaluate_model(X_test, y_test)
    print(test_auc)
    
    plot_roc_nn(model, X, y)