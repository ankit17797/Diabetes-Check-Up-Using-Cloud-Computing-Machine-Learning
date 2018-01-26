# Load libraries
	import pandas

	from sklearn import model_selection
	from  sklearn.linear_model import LogisticRegression
	from sklearn.metrics import classification_report
	from sklearn.metrics import confusion_matrix
	from sklearn.metrics import accuracy_score
	#imports the algorithms
	from sklearn.linear_model import LogisticRegression
	from sklearn.tree import DecisionTreeClassifier
	from sklearn.neighbors import KNeighborsClassifier
	from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
	from sklearn.naive_bayes import GaussianNB

	#Basic information about Data
	#Names of the attributes in the dataset
	names=['preg','plas','pres','skin','test','mass','pedi','age','class']
	#load the dataset
	dataset = pandas.read_csv('data1.csv', names=names)

	# Numbers of row and column in dataset
	print('number of rows and column' )
	print(dataset.shape)

	# print the data from tail
	print(dataset.tail(20))
	print(dataset.groupby('class').size())

	#print the data from head
	print(dataset.head(20))
	print(dataset.groupby('class').size())

	# Print the data by  class
	print('data group by   class');
	print(dataset.groupby('class').size())

	#operation performed on dataset
	# Split-out validation dataset
	array = dataset.values
	X = array[:,0:8]
	Y = array[:,8]
	validation_size = 0.33
	X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size)

	# Test options and evaluation metric
	#scoring = 'accuracy'
	# Spot Check Algorithms
	models = []
	models.append(('LR', LogisticRegression()))
	models.append(('LDA', LinearDiscriminantAnalysis()))
	models.append(('KNN', KNeighborsClassifier()))
	models.append(('CART', DecisionTreeClassifier()))
	models.append(('NB', GaussianNB()))

	# evaluate each model in turn
	results = []
	names = []
	for name, model in models:
		kfold = model_selection.KFold(n_splits=6)
		cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold)
		results.append(cv_results)
		names.append(name)
		msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
		print(msg)

	# Make predictions on validation dataset
	nb=GaussianNB()
	nb.fit(X_train,Y_train)
	predictions = nb.predict(X_validation)
	print(accuracy_score(Y_validation, predictions))
	print(confusion_matrix(Y_validation, predictions))

	#print the result that is recall, precision,f1-score
	print('The results after applying valid algorithmm'  )
	print(classification_report(Y_validation, predictions))



