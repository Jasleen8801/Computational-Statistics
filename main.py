# Modules imported
from turtle import pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statistics
from sklearn.linear_model import LinearRegression
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg


# Choose the function to be called
def choice(i):
    switcher = {
        1: SimpleLinearRegression,
        2: VarianceInflationFactor,
        3: MultipleLinearRegression,
        4: ConfusionMatrix,
        5: NoiseMatrix,
        6: exit
    }
    switcher[i]()


# Simple Linear Regression
def SimpleLinearRegression():
    print("*******SIMPLE LINEAR REGRESSION*******")
    print("Press 1 to perform SLR using Manual Calculation of a random dataset")
    print("Press 2 to perform SLR using Manual Calculation of a Kaggle dataset")
    print("Press 3 to perform SLR using Inbuilt Python Function of a random dataset")
    print("Press 4 to perform SLR using Inbuilt Python Function of a Kaggle dataset")
    val = int(input("Enter the value to perform corresponding function: "))
    switch1 = {
        1: SLR_1,
        2: SLR_2,
        3: SLR_3,
        4: SLR_4
    }
    switch1[val]()


def SLR_1():
    X = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    y = np.array([1, 3, 2, 5, 7, 8, 8, 9, 10, 12])
    b = regrr_coeff(X, y)
    print("Estimated coefficients; \n b_0 = {} \n b_1 = {}".format(b[0], b[1]))
    print(f"Coefficient of Determination: {coeff_of_determination(X, y)}")
    plot_regr_line(X, y, b)


def SLR_2():
    data = pd.read_csv('Salary_Data.csv')
    X = data['YearsExperience']
    y = data['Salary']
    X = ((X - X.min())/(X.max() - X.min()))
    y = ((y - y.min())/(y.max() - y.min()))
    b = regrr_coeff(X,y)
    print("Estimated coefficients:\n b_0 = {}  \n b_1 = {}".format(b[0], b[1]))
    print(f"Coefficient of Determination: {coeff_of_determination(X,y)}")
    plot_regr_line(X, y, b)


def SLR_3():
    X = np.array([0,1,2,3,4,5,6,7,8,9]).reshape((-1, 1))
    y = np.array([1,3,2,5,7,8,8,9,10,12])
    model = LinearRegression().fit(X, y)
    print(f"Coefficient of Determination: {model.score(X, y)}")
    print(f"Intercept: {model.intercept_}")
    print(f"Slope: {model.coef_}")


def SLR_4():
    data = pd.read_csv('Salary_Data.csv')
    X = np.array(data['YearsExperience']).reshape((-1, 1))
    y = np.array(data['Salary'])
    X = ((X - X.min())/(X.max() - X.min()))
    y = ((y - y.min())/(y.max() - y.min()))
    model = LinearRegression().fit(X, y)
    print(f"Coefficient of Determination: {model.score(X, y)}")
    print(f"Intercept: {model.intercept_}")
    print(f"Slope: {model.coef_}")


# Functions used in SLR
def regrr_coeff(X, y):
    n = len(X)
    mean_x = statistics.mean(X)
    mean_y = statistics.mean(y)
    SS_xy = sum(y*X) - n*mean_y*mean_x
    SS_xx = sum(X*X) - n*mean_x*mean_y
    b_1 = SS_xy / SS_xx
    b_0 = mean_y - b_1 * mean_x
    return (b_0, b_1)


def plot_regr_line(X, y, b):
    plt.scatter(X, y, color="b")
    y_pred = b[0] + b[1] * X
    plt.plot(X, y_pred, color="r")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


def coeff_of_determination(X, y):
    n = len(X)
    r2_num = (n*sum(X*y) - sum(X)*sum(y))**2
    r2_den = (n*sum(X*X) - (sum(X))**2) * (n*sum(y*y) - (sum(y))**2)
    r_2 = r2_num / r2_den
    return r_2


# Variance Inflation Factor and Multicollinearity
def VarianceInflationFactor():
    print("*******VARIANCE INFLATION FACTOR*******")
    print("Press 1 to perform VIF using Manual Calculation")
    print("Press 2 to perform VIF using Inbuilt Python Function")
    val = int(input("Enter the value to perform corresponding function: "))
    switch2 = {
        1: VIF_1,
        2: VIF_2,
    }
    switch2[val]()


def VIF_1():
    df = pd.read_csv("Car_sales.csv")
    considered_features = ['Fuel_efficiency', 'Power_perf_factor', 'Engine_size', 'Horsepower', 'Fuel_capacity', 'Curb_weight']
    for (i, j) in (considered_features, considered_features):
        if(i != j):
            print(f"{i}{j}")
            compute_vif1(df[i], df[j])


def VIF_2():
    df = pd.read_csv("Car_sales.csv")
    plt.figure(figsize=(10, 7))
    # Generate a mask to only show the botton triangle
    mask = np.triu(np.ones_like(df.corr(), dtype=bool)) 
    # Generating a Heatmap
    sns.heatmap(df.corr(), annot=True, mask=mask, vmin=-1, vmax=1)
    plt.title('Correlation Coefficient of Predictors')
    plt.show()
    # Computing VIF
    considered_features = ['Fuel_efficiency', 'Power_perf_factor', 'Engine_size', 'Horsepower', 'Fuel_capacity', 'Curb_weight']
    compute_vif2(considered_features).sort_values('VIF', ascending=False)


def compute_vif1(x1, x2):
    df = pd.read_csv("Car_sales.csv")
    r_sq = coeff_of_determination(x1, x2)
    vif = 1 / (1 - r_sq)
    print(f"VIF-> {vif}")


def compute_vif2(considered_features):
    df = pd.read_csv("Car_sales.csv")
    X = df[considered_features]
    # the calculation of variance inflation requires a constant
    X['intercept'] = 1
    vif = pd.DataFrame()
    vif["Variable"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif = vif[vif['Variable'] != 'intercept']
    return vif


# Multiple Linear Regression
def MultipleLinearRegression():
    print("*******MULTIPLE LINEAR REGRESSION*******")
    print("Press 1 to perform MLR using Manual Calculation")
    print("Press 2 to perform MLR using Inbuilt Python Function")
    val = int(input("Enter the value to perform corresponding function: "))
    switch3 = {
        1: MLR_1,
        2: MLR_2,
    }
    switch3[val]()


def MLR_1():
    boston = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv')
    class MultipleRegression(object):
        def __init__(self) -> None:
            self.coefficients = []

        def fit(self, X, y):
            if len(X.shape) == 1: 
                X = self._reshape_x(X)
            X = self._concatenate_ones(X)
            self.coefficients = np.linalg.inv(X.transpose().dot(X)).dot(X.transpose()).dot(y)

        def _reshape_x(self, X):
            return X.reshape(-1, 1)

        def _concatenate_ones(self, X):
            ones = np.ones(shape=X.shape[0]).reshape(-1, 1)
            return np.concatenate((ones, X), 1)

    X = boston.drop('medv', axis=1).values
    y = boston['medv'].values
    model = MultipleRegression()
    model.fit(X, y)
    print(f"Coefficients: {model.coefficients}")


def MLR_2():
    dataset = pd.read_csv("Advertising.csv")
    X = dataset[['TV', 'Radio', 'Newspaper']]
    y = dataset['Sales']
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
    mlr = LinearRegression()
    mlr.fit(x_train, y_train)
    print(f"Intercept: {mlr.intercept_}")
    print("Coefficients: ")
    print(list(zip(X, mlr.coef_)))


# Confusion Matrix - Accuracy, Precision, Recall, F1-score
def ConfusionMatrix():
    class ConfusionMat(object):
        def __init__(self) -> None:
            self.params = []
            self.actual = [1, 3, 3, 2, 5, 5, 3, 2, 1, 4, 3, 2, 1, 1, 2]
            self.predicted = [1, 2, 3, 4, 2, 3, 3, 2, 1, 2, 3, 1, 5, 1, 1]
            self.confusion_matrix = self.comp_confmat()

        def totalSum(self):
            '''to calculate total sum of elements in a confusion matrix'''
            tot_sum = 0
            for i in range(0, len(self.confusion_matrix)):
                for j in range(0, len(self.confusion_matrix)): 
                    tot_sum += self.confusion_matrix[i][j]
            return tot_sum

        def param(self, mat, n):
            '''to calculate the values of parameters in each class'''
            col_sum = [sum([row[i] for row in mat]) for i in range(0,len(mat[0]))]
            row_sum = [sum(mat[i]) for i in range(len(mat))]
            tp = mat[n][n]
            fp = row_sum[n] - tp
            fn = col_sum[n] - tp
            tn = self.totalSum() - fp - fn - tp
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            self.params.append(accuracy)
            precision = tp / (tp + fp)
            self.params.append(precision)
            recall = tp / (tp + fn)
            self.params.append(recall)
            f1_score = (2 * precision * recall) / (precision + recall)
            self.params.append(f1_score)
            return self.params

        def display_val(self):
            '''to display the required values'''
            for i in range(len(self.confusion_matrix)):
                print(f"For the class {i+1}, the values are as follows: ")
                print(f"Accuracy: {self.param(self.confusion_matrix,i)[0]}")
                print(f"Precision: {self.param(self.confusion_matrix,i)[1]}")
                print(f"Recall: {self.param(self.confusion_matrix,i)[2]}")
                print(f"F1-Score: {self.param(self.confusion_matrix,i)[3]}")
                print("*********************")

        def comp_confmat(self):
            '''to find the confusion matrix using actual and predicted values'''
            # extract the different classes
            classes = np.unique(self.actual)
            # initialize the confusion matrix
            confmat = np.zeros((len(classes), len(classes)))
            # loop across the different combinations of actual / predicted classes
            for i in range(len(classes)):
                for j in range(len(classes)):
                    # count the number of instances in each combination of actual / predicted classes
                    confmat[i, j] = np.sum((self.actual == classes[i]) & (self.predicted == classes[j]))
            return confmat

    conMat = ConfusionMat()
    conMat.display_val()


def NoiseMatrix():
    img = mpimg.imread('gray_img.jpg')
    R, G, B = img[:,:,0], img[:,:,1], img[:,:,2]
    imgGray = 0.2989*R + 0.5870*G + 0.1140*B
    plt.imshow(imgGray, cmap='gray')
    plt.show()
    # imgGray.shape

    # plt.hist(imgGray.ravel(),256,[0,256])
    # plt.show()

    var = int(input("Enter the variance you want to add: "))
    noise = np.random.normal(0, var, imgGray.shape)
    # print(noise)

    new_data = noise + imgGray
    plt.imshow(new_data, cmap='gray')

    # In histogram format
    # plt.hist(new_data.ravel(),256,[0,256])
    # plt.show()

    # Format - 2
    ax = plt.hist(new_data.ravel(), bins=256)
    plt.show()
    
    approx_median = np.percentile(new_data, 50)
    
    ax = plt.hist(new_data.ravel(), bins=256)
    plt.axvline(approx_median, color='orange')
    plt.show()
    
    quartiles = [25, 50, 75]
    ax = plt.hist(new_data.ravel(), bins=256)
    for q in np.percentile(new_data, quartiles):
        plt.axvline(q, color='orange')
    plt.show()


# Main Program
print("*******MAIN PROGRAM*******")
print("Press 1 to perform Simple Linear Regression.")
print("Press 2 to find the Variance Inflation Factor.")
print("Press 3 to perform Multiple Linear Regression.")
print("Press 4 to perform Confusion Matrix Operation.")
print("Press 5 to perform Noise Matrix operation.")
print("Press 6 to exit the program.")
i = int(input("Enter the value to perform corresponding functions: "))
choice(i)
