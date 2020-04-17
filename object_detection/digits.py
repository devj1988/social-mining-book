from sklearn import datasets, metrics
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split


digits = datasets.load_digits()

X, y = digits.data / 255. , digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=100, alpha=1e-4, solver='adam', verbose=10, tol=1e-4,
                    random_state=1, learning_rate_init=.1)

mlp.fit(X_train, y_train)

print()
print("Training set score:  {0}".format(mlp.score(X_train, y_train)))
print("Test set score:  {0}".format(mlp.score(X_test, y_test)))

import matplotlib.pyplot as plt
#
# fig, axes = plt.subplots(10, 10)
# fig.set_figwidth(20)
# fig.set_figheight(20)
#
# for coef, ax in zip(mlp.coefs_[0].T, axes.ravel()):
#     ax.matshow(coef.reshape(8, 8), cmap=plt.cm.gray, interpolation='bicubic')
#     ax.set_xticks(())
#     ax.set_yticks(())

#plt.show()

predicted = mlp.predict(X_test)

import numpy as np

for i in range(5):
    image = np.reshape(X_test[i], (8,8))

    print('Ground truth {0}'.format(y_test[i]))
    print('Predicted {0}'.format(predicted[i]))

    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.axis('off')
    plt.show()


