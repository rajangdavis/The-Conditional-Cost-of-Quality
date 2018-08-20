from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def scaled_linear_regression(X, y):
	X_train, X_test, y_train, t_test = train_test_split(X,y)
	ss = StandardScaler()
	X_train_scaled = ss.fit_transform(X_train)
	X_test_scaled = ss.fit_transform(X_test)

	lr = LinearRegression()
	lr.fit(X_train_scaled, y_train)
	lr.score(X_train_scaled, y_train)
	lr.score(X_test_scaled, y_test)

