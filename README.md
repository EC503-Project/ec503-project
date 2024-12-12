
# Readme file for: Comparing K-Means, Ridge Regression, Decision Trees and Random Forests for Datasets with Outliers and Missing Features

Quargs Greene, Daniel Gergeus, Shivam Goyal, Dhiraj Simhadri
{qgreene,  dgergeus, sgoyal15, dhirajs}@bu.edu

**Links to project code and original datasets:**
* Link to project code: https://github.com/EC503-Project/ec503-project
* Link to Titanic dataset: https://www.kaggle.com/competitions/titanic/overview
* Link to Book Recommendation dataset: https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset/discussion?sort=hotness
* Link to Fashion MNIST dataset: https://www.kaggle.com/datasets/zalando-research/fashionmnist

**Functions implemented:**
* entropy: calculate entropy of decision tree
* information_gain: calculate information gain of decision tree
* DecisionTree.train: train decision tree model
* DecisionTree.predict: make predictions based on decision tree model
* predict: make predictions based on random forest model
* distance: compute Euclidean distance between two feature vectors
* initialize_clusters: initialize centers in k-means algorithm
* run_kmeans: run k-means algorithm
* compute_wcss: compute Within Cluster Sum of Squares (WCSS)
* compute_slihouette_score: compute silhouette score for k-means
	
**Python library and package functions (see corresponding documentation for more information):**
* pandas.DataFrame.info: 
* numpy.unique
* numpy.log
* pandas.DataFrame.append
* pandas.DataFrame.loc
* warnings.filterwarnings
* print
* pandas.DataFrame.reset_index
* numpy.sum
* sklearn.tree.RandomForestClassifier
* sklearn.tree.score
* sklearn.tree.export_graphviz
* matplotlib.pyplot.plot
* matplotlib.pyplot.show
* numpy.argmax
* numpy.eye
* numpy.linalg.inv
* numpy.array.flatten
* matplotlib.xlabel
* matplotlib.ylabel
* matplotlib.grid
* numpy.mean
* numpy.std
* Numpy.figure
* numpy.arange
* numpy.exp
* matplotlib.legend
* matplotlib.title
* pandas.DataFrame.head
* pandas.DataFrame.describe

**Code for cleaning and preprocessing (see corresponding documentation for any Python library code):**
* preprocess.py: reads CSV files, shapes them, recreates train-test divisions, synthetically generates pseudorandom Gaussian data with induced outliers,0s, and NaN values, creates cleaned control datasets trimmed of outliers, saves removed portions and outliers from datasets, displays datasets in command line and first-order statistics
* divide_data: an alternative to sklearn.model_selection.train_test_split for splitting training and testing data
* pandas.DataFrame.fillna
* pandas.DataFrame.to_numpy
* pandas.DataFrame.iloc
* numpy.random.seed
* pandas.DataFrame.fit
* sklearn.model_selection.train_test_split
* sklearn.preprocessing.MinMaxScaler
* numpy.random.normal
* pandas.DataFrame.transform
* numpy.random.randint
* numpy.percentile
* pandas.DataFrame.mask
* pandas.read_csv
* pandas.DataFrame.drop
* np.random.choice
* sklearn.preprocessing.fit_transform

**For dependencies, see requirements.txt and import statements at the top of each file and their corresponding documentation.**
