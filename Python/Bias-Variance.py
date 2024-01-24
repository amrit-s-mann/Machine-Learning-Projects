'''Setting up the Regression'''

# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Loading training data (x = ROE ; y = P/B ratio)
x=np.array([0.01, 0.019, 0.04, 0.07, 0.14,0.27, 0.52,0.53,0.55, 0.6,0.65, 0.66, 0.7, 0.715,0.8,0.85])
y=np.array([1.79, 1.82, 3.20, 5.31, 6.75, 6.80, 8.86, 5, 5.5, 8.8, 8.9, 6.6, 9.02, 9.03,9.03,9.03])

# Loading validation data
x_test = np.linspace(-0.2, 0.85, 1000)

# Scatter plot of training data
plt.scatter(x, y, marker='o', color='red')
plt.xlabel('ROE %')
plt.ylabel('P/B ratio')
plt.show()

'''Using PolynomialFeatures & LinearRegression'''

# Creating inputs for plot and using polynomial degrees - 1, 2 and 4
plot_titles = ['d = 1(under-fit; high bias)', 'd = 2', 'd = 4(over-fit ; high variance)']
degrees = [1, 2, 4]

# Creating a new figure & adjusts the layout parameters
fig = plt.figure(figsize=(9, 3.5))
fig.subplots_adjust(left = 0.06, right = 0.98, bottom = 0.15, top = 0.85, wspace = 0.05)

# Creating multiple plots using iterative method and adding subplots
for i, d in enumerate(degrees):
    ax = fig.add_subplot(131 + i, xticks = [], yticks = [])
    ax.scatter(x, y, marker = 'x', c = 'k', s = 50)

    # Creating pipeline of polynomial features
    model = make_pipeline(PolynomialFeatures(d), LinearRegression())
    # Fitting the best plot of validation data to the existing plot
    model.fit(x[:, np.newaxis], y)
    # Adding best fit of validation data to the existing training data plot
    ax.plot(x_test, model.predict(x_test[:, np.newaxis]), '-b')

    # Setting the limits & labels of x & y axis as well as titles of plot
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(0, 10)
    ax.set_xlabel('ROE %')
    if i == 0:
        ax.set_ylabel('P/B ratio')
    ax.set_title(plot_titles[i])

# Output of validation data plot
plt.show()

    