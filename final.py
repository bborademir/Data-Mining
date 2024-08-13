import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import warnings

# Uyarıları filtreleme
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Veri okuma
data1 = pd.read_csv("audi.csv")
audi = data1.copy()

# Veriye genel bakış
print(audi.head())

# Kullanılmayacak sütunlar atılıyor
audi = audi.drop(columns=["model", "year"], axis=1)

# Kukla (dummy) değişkenler oluşturuluyor
audi["transmission"] = [1 if x == "Manual" else 0 for x in audi["transmission"]]
audi["fuelType"] = [1 if x == "Petrol" else 0 for x in audi["fuelType"]]

sns.heatmap(audi.corr(), annot=True)

# Veri bilgisi
print(audi.info())

# Null değer kontrolü
print(audi.isnull().sum())

# Veri türleri ve örnek gözlemler
print(audi.columns)
print(audi.dtypes)
print(audi.head())

### Basit Doğrusal Regresyon

# Basit Doğrusal Regresyon için veri ön işleme
focusedData = audi.loc[:, ["mileage", "price"]]
focusedData.head()

# Mileage ve price arasındaki ilişki görselleştirme
sns.regplot(x="mileage", y="price", data=focusedData, fit_reg=False)
sns.regplot(x="mileage", y="price", data=focusedData, fit_reg=True)
sns.jointplot(x="mileage", y="price", data=focusedData, kind="reg")

# Train ve test setleri oluşturma
focusedTrain, focusedTest = train_test_split(focusedData, test_size=0.3, random_state=0)
XTrain, yTrain, XTest, yTest = focusedTrain["mileage"], focusedTrain["price"], focusedTest["mileage"], focusedTest["price"]

print(XTrain.shape, XTest.shape, yTrain.shape, yTest.shape)

# Array'e dönüştürme işlemi
XTrain = XTrain.values.reshape(-1, 1)
XTest = XTest.values.reshape(-1, 1)

# Lineer Regresyon modeli oluşturma
lR = LinearRegression()
lRModel = lR.fit(XTrain, yTrain)

# Model parametreleri
print(lRModel.intercept_)  # B0
print(lRModel.coef_)  # Katsayı (B1)

# Modelin başarı değeri
print(lRModel.score(XTrain, yTrain))

# Tahminler
print(lRModel.predict(XTrain)[0:10])  # Train seti için ilk 10 tahmin
print(lRModel.predict(XTest)[0:10])  # Test seti için ilk 10 tahmin

# Train seti için hatalar ve metrikler
yHatTrain = pd.DataFrame(lRModel.predict(XTrain), index=yTrain.index)
residualsTrain = pd.concat([yTrain, yHatTrain], axis=1)
residualsTrain.columns = ["Real Price", "Predicted Price"]
residualsTrain["Residuals"] = residualsTrain["Real Price"] - residualsTrain["Predicted Price"]
residualsTrain["SQR Residuals"] = residualsTrain["Residuals"]**2
residualsTrain["Pr Residuals"] = abs(residualsTrain["Residuals"] / residualsTrain["Real Price"])

# Hata metrikleri
print(np.mean(residualsTrain["SQR Residuals"]))  # MSE
print(np.mean(residualsTrain["SQR Residuals"])**0.5)  # RMSE
print(np.mean(residualsTrain["Pr Residuals"]))  # MAPE
print(np.median(residualsTrain["Pr Residuals"]))  # MdAPE

# Test seti için hatalar ve metrikler
yHatTest = pd.DataFrame(lRModel.predict(XTest), index=yTest.index)
residualsTest = pd.concat([yTest, yHatTest], axis=1)
residualsTest.columns = ["Real Price", "Predicted Price"]
residualsTest["Residuals"] = residualsTest["Real Price"] - residualsTest["Predicted Price"]
residualsTest["SQR Residuals"] = residualsTest["Residuals"]**2
residualsTest["Pr Residuals"] = abs(residualsTest["Residuals"] / residualsTest["Real Price"])

# Hata metrikleri
print(np.mean(residualsTest["SQR Residuals"]))  # MSE
print(np.mean(residualsTest["SQR Residuals"])**0.5)  # RMSE
print(np.mean(residualsTest["Pr Residuals"]))  # MAPE
print(np.median(residualsTest["Pr Residuals"]))  # MdAPE

### Çoklu Doğrusal Regresyon

audiTrain, audiTest = train_test_split(audi, test_size=0.3, random_state=3000)
scaler = MinMaxScaler()
numericVars = ["price", "mileage", "tax", "mpg", "engineSize"]

audiTrain[numericVars] = scaler.fit_transform(audiTrain[numericVars])
audiTest[numericVars] = scaler.fit_transform(audiTest[numericVars])

yTrain = audiTrain["price"]
XTrain = audiTrain.drop(columns="price", axis=1)
yTest = audiTest["price"]
XTest = audiTest.drop(columns="price", axis=1)

lmRFE = LinearRegression()
lmRFE.fit(XTrain, yTrain)

# En önemli 10 değişken
rfe = RFE(lmRFE, n_features_to_select=10)
rfe.fit(XTrain, yTrain)

list(zip(audiTrain.columns, rfe.support_, rfe.ranking_))

yPredTest = rfe.predict(XTest)
r2_score(y_true=yTest, y_pred=yPredTest)

## GridSearchCV

folds = KFold(n_splits = 5, shuffle = True, random_state = 0)

parameters1 = [{"n_features_to_select": list(range(1, 14))}]

lmCVGS = LinearRegression().fit(XTrain, yTrain)
rfeCVGS = RFE(lmCVGS)

model1CVGS = GridSearchCV(estimator=rfeCVGS, param_grid=parameters1, scoring="r2", cv=folds, verbose=1,
                          return_train_score=True).fit(XTrain, yTrain)

results1 = pd.DataFrame(model1CVGS.cv_results_)
results1

prediction1 = model1CVGS.predict(XTest)

plt.figure(figsize=(16, 6));
plt.plot(results1["param_n_features_to_select"],
results1["mean_test_score"]);
plt.plot(results1["param_n_features_to_select"], 
results1["mean_train_score"]);
plt.xlabel("number of features");
plt.ylabel("r-squared");
plt.title("Optimal Number of Features");
plt.legend(["test score", "train score"], loc="upper left")

model1R2 = r2_score(y_true=yTest, y_pred=prediction1)
model1MAE = mean_absolute_error(y_true=yTest, y_pred=prediction1)
model1MSE = mean_squared_error(y_true=yTest, y_pred=prediction1)
model1RMSE = mean_squared_error(y_true=yTest, y_pred=prediction1, squared=False)


#DECISION TREE #########################




y = audi["price"]
X = audi.drop(["price"],axis = 1)



#Veri seti bölümlemesi

audi_X_train, audi_X_test, audi_y_train, audi_y_test = train_test_split(X,y,test_size = 0.25,random_state=3000)

#model oluşturma

decisiontree = DecisionTreeRegressor(random_state=3000).fit(audi_X_train,audi_y_train)

dir(decisiontree)
#?decisiontree
#TAHMİN VE PERFORMANS

y_pred_decisiontree_train = decisiontree.predict(audi_X_train)

y_pred_decisiontree_test = decisiontree.predict(audi_X_test)


#test için

model5RMSE= np.sqrt(mean_squared_error(audi_y_test, y_pred_decisiontree_test))
model5R2 = r2_score(audi_y_test, y_pred_decisiontree_test)
model5MAPE = mean_absolute_percentage_error(audi_y_test, y_pred_decisiontree_test)
model5MAE = mean_absolute_error(audi_y_test, y_pred_decisiontree_test)


print(model5RMSE,model5R2,model5MAPE,model5MAE)

#train için

model5RMSEtra = np.sqrt(mean_squared_error(audi_y_train, y_pred_decisiontree_train))
model5R2tra = r2_score(audi_y_train, y_pred_decisiontree_train)
model5MAPEtra = mean_absolute_percentage_error(audi_y_train, y_pred_decisiontree_train)
model5MAEtra = mean_absolute_error(audi_y_train, y_pred_decisiontree_train)


print(model5RMSEtra,model5R2tra,model5MAPEtra,model5MAEtra)


#MODEL TUNING

folds = KFold(n_splits = 5, shuffle = True, random_state = 3000)

parameters5 = {"max_depth":[5,6,7,8,9,10],
               "min_samples_split":[2,3,5,10,30,50,100]}

decisiontreefCV = DecisionTreeRegressor(random_state=3000)

model5CVGS = GridSearchCV(decisiontreefCV, param_grid=parameters5, scoring="r2", cv=folds,
                          return_train_score=True).fit(audi_X_train, audi_y_train)

results5 = pd.DataFrame(model5CVGS.cv_results_)
results5

prediction5 = model5CVGS.predict(audi_X_test)

plt.figure(figsize=(10, 5));
plt.plot(results5["param_max_depth"],
results5["mean_test_score"]);
plt.plot(results5["param_min_samples_split"], 
results5["mean_train_score"]);
plt.xlabel("number of features");
plt.ylabel("r-squared");
plt.title("Optimal Number of Features");
plt.legend(["test score", "train score"], loc="upper left")



model5CVGS.best_params_

model5tuned = DecisionTreeRegressor(max_depth = 10, min_samples_split = 5).fit(audi_X_train,audi_y_train)


model5tuned_test = model5tuned.predict(audi_X_test)

model5tuned_train = model5tuned.predict(audi_X_train)

#performans
model5RMSEtuned= np.sqrt(mean_squared_error(audi_y_test, model5tuned_test))
model5R2tuned = r2_score(audi_y_test, model5tuned_test)
model5MAPEtuned = mean_absolute_percentage_error(audi_y_test, model5tuned_test)
model5MAEtuned = mean_absolute_error(audi_y_test, model5tuned_test)

print(model5RMSEtuned,model5R2tuned,model5MAPEtuned,model5MAEtuned)


#RANDOM FOREST ########################


y = audi["price"]
X = audi.drop(["price"],axis = 1)



#Veri seti bölümlemesi

audi_X_train, audi_X_test, audi_y_train, audi_y_test = train_test_split(X,y,test_size = 0.25,random_state=3000)

#model oluşturma

randomforest = RandomForestRegressor(n_estimators=100,random_state=3000).fit(audi_X_train,audi_y_train)

dir(randomforest)
#?randomforest
#TAHMİN VE PERFORMANS

y_pred_randomforest_train = randomforest.predict(audi_X_train)

y_pred_randomforest_test = randomforest.predict(audi_X_test)


#test için

model6RMSE= np.sqrt(mean_squared_error(audi_y_test, y_pred_randomforest_test))
model6R2 = r2_score(audi_y_test, y_pred_randomforest_test)
model6MAPE = mean_absolute_percentage_error(audi_y_test, y_pred_randomforest_test)
model6MAE = mean_absolute_error(audi_y_test, y_pred_randomforest_test)


print(model6RMSE,model6R2,model6MAPE,model6MAE)

#train için

model6RMSEtra = np.sqrt(mean_squared_error(audi_y_train, y_pred_randomforest_train))
model6R2tra = r2_score(audi_y_train, y_pred_randomforest_train)
model6MAPEtra = mean_absolute_percentage_error(audi_y_train, y_pred_randomforest_train)
model6MAEtra = mean_absolute_error(audi_y_train, y_pred_randomforest_train)


print(model6RMSEtra,model6R2tra,model6MAPEtra,model6MAEtra)


#MODEL TUNING

folds = KFold(n_splits = 5, shuffle = True, random_state = 3000)



parameters6 ={"max_depth":[5,6,7,8,9,10],
              "min_samples_split":[2,10,80,100]}


randomforestfCV = RandomForestRegressor(n_estimators=100,random_state=3000)

model6CVGS = GridSearchCV(randomforestfCV, param_grid=parameters6, scoring="r2", cv=folds,
                          return_train_score=True).fit(audi_X_train, audi_y_train)

results6 = pd.DataFrame(model6CVGS.cv_results_)
results6

prediction6 = model6CVGS.predict(audi_X_test)


plt.figure(figsize=(10, 5));
plt.plot(results6["param_max_depth"],
results6["mean_test_score"]);
plt.plot(results6["param_min_samples_split"], 
results6["mean_train_score"]);
plt.xlabel("number of features");
plt.ylabel("r-squared");
plt.title("Optimal Number of Features");
plt.legend(["test score", "train score"], loc="upper left")



model6CVGS.best_params_

model6tuned = RandomForestRegressor(max_depth = 10, min_samples_split = 2).fit(audi_X_train,audi_y_train)


model6tuned_test = model6tuned.predict(audi_X_test)

model6tuned_train = model6tuned.predict(audi_X_train)

#performans
model6RMSEtuned= np.sqrt(mean_squared_error(audi_y_test, model6tuned_test))
model6R2tuned = r2_score(audi_y_test, model6tuned_test)
model6MAPEtuned = mean_absolute_percentage_error(audi_y_test, model6tuned_test)
model6MAEtuned = mean_absolute_error(audi_y_test, model6tuned_test)

print(model6RMSEtuned,model6R2tuned,model6MAPEtuned,model6MAEtuned)

