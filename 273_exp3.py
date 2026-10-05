import seaborn as sns                                   
from sklearn.model_selection import train_test_split     
from sklearn.linear_model import LinearRegression        

data = sns.load_dataset("mpg").dropna()                  
X = data[["cylinders","displacement","horsepower","weight","acceleration","model_year"]]  s
y = data["mpg"]                          
X_train, X_test, y_train, y_test = train_test_split(     
    X, y, test_size=0.2, random_state=1)
model = LinearRegression()                               
model.fit(X_train, y_train)                              
print("R2 score:", round(model.score(X_test, y_test), 3)) 
