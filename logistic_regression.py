import numpy as np
import matplotlib.pyplot as plt
X = np.random.randn(100, 3)
y = np.random.randint(0, 2, (100, 1))
X_b = np.c_[np.ones((100, 1)), X]

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

n_samples, n_features = X_b.shape
theta_best = np.random.randn(n_features, 1)
n_epochs = 50
t0, t1 = 5, 50

def learning_schedule(t):
    return t0 / (t + t1)

for epoch in range(n_epochs):
    for i in range(n_samples):
        # Стохастический выбор одного примера
        idx = np.random.randint(n_samples)
        xi = X_b[idx:idx+1]
        yi = y[idx:idx+1]
        
        # Вычисление градиента для логистической регрессии
        prediction = sigmoid(xi.dot(theta_best))
        gradient = xi.T.dot(prediction - yi)
        
        # Обновление весов
        eta = learning_schedule(epoch * n_samples + i)
        theta_best -= eta * gradient






targets = [1,1,0,0,0,1,0,1,0,0]
predicts = [0.8,0.8,0.3,0.2,0.2,0.8,0.2,0.6,0.2,0.2]
thresholds = [i for i in np.arange(1, -0.1, -0.1)]

roc_dots = set()
for threshold in thresholds:
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(len(predicts)):
        if targets[i] == int(predicts[i] > threshold):
            if int(predicts[i] > threshold) == 1:
                tp += 1
            else:
                tn += 1
        else:
            if int(predicts[i] > threshold) == 1:
                fp += 1
            else:
                fn += 1
    
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)

    roc_dots.add((fpr, tpr))
    print(tp, tn, fp, fn)

roc_dots = sorted(roc_dots)

fpr, tpr = zip(*roc_dots)
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, 'b-o', linewidth=2, markersize=8)
plt.plot([0, 1], [0, 1], 'r--', alpha=0.5)  # диагональ (случайный классификатор)
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve')
plt.grid(True, alpha=0.3)
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.show()
    





