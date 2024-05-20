# 5th-chapter-ML

## 1. Data Collection
- **Data Sources**
  - Online databases (e.g., UCI Machine Learning Repository)
  - APIs (e.g., Twitter API, OpenWeather API)
  - Web scraping
  - Sensor data
  - User-generated content
  - Public datasets (e.g., Kaggle)

- **Data Acquisition Techniques**
  - Manual data entry
  - Automated data collection scripts
  - Use of data aggregation tools

- **Data Storage**
  - Databases (SQL, NoSQL)
  - Data lakes
  - Cloud storage solutions (AWS S3, Google Cloud Storage)

- **Data Privacy and Ethics**
  - Data anonymization
  - Informed consent
  - Compliance with regulations (GDPR, CCPA)

## 2. Data Normalization and Regularization
Normalization and regularization are essential techniques in data preprocessing and machine learning model training, serving distinct but complementary purposes. Let's delve deeper into each aspect.

### Normalization Techniques

**1. Min-Max Scaling:**
   - **Purpose:** Rescales the data to a fixed range, usually [0, 1].
   - **Formula:** x' = (x - x_min) / (x_max - x_min)
   - **Use Case:** Useful when the data needs to be bounded within a specific range, especially for algorithms like neural networks that expect input values within a certain scale.

**2. Z-score Standardization:**
   - **Purpose:** Transforms the data to have a mean of 0 and a standard deviation of 1.
   - **Formula:** x' = (x - μ) / σ
   - **Use Case:** Effective for algorithms that assume data is normally distributed, such as linear regression, logistic regression, and k-means clustering.

**3. Robust Scaling:**
   - **Purpose:** Uses the median and interquartile range (IQR) to scale data, making it robust to outliers.
   - **Formula:** x' = (x - median) / IQR
   - **Use Case:** Ideal when the dataset contains many outliers or is not normally distributed.

### Regularization Techniques

**1. L1 Regularization (Lasso):**
   - **Purpose:** Adds the absolute value of the coefficients as a penalty term to the loss function.
   - **Formula:** Loss = Loss_original + λ ∑ |w_i|
   - **Use Case:** Performs feature selection by shrinking some coefficients to zero, effectively removing irrelevant features.

**2. L2 Regularization (Ridge):**
   - **Purpose:** Adds the squared value of the coefficients as a penalty term to the loss function.
   - **Formula:** Loss = Loss_original + λ ∑ w_i^2
   - **Use Case:** Prevents overfitting by constraining the size of the coefficients, but does not perform feature selection.

**3. Elastic Net:**
   - **Purpose:** Combines L1 and L2 regularization penalties.
   - **Formula:** Loss = Loss_original + λ₁ ∑ |w_i| + λ₂ ∑ w_i^2
   - **Use Case:** Provides a balance between feature selection (L1) and coefficient shrinking (L2), useful when dealing with highly correlated features.

### Purpose of Normalization and Regularization

**Normalization:**
- **Objective:** Ensures that features contribute equally to the model by standardizing their scales. This is particularly important for algorithms that compute distances between data points (e.g., k-nearest neighbors, SVMs) or perform gradient-based optimization (e.g., neural networks).
- **Benefit:** Improves model performance and convergence speed by making the training process more stable.

**Regularization:**
- **Objective:** Prevents overfitting by adding a penalty to the loss function for large coefficients. This encourages the model to maintain simpler, more generalizable representations.
- **Benefit:** Enhances the model's ability to generalize to new, unseen data by reducing variance and controlling complexity.

  

## 3. Dimensionality Reduction
Dimensionality reduction techniques are essential in data analysis and machine learning for simplifying datasets, reducing computation time, and improving model performance. Here's an overview of the techniques you mentioned:

### 1. Principal Component Analysis (PCA)
**Description**: PCA is a linear dimensionality reduction technique that transforms the data into a new coordinate system. It does this by projecting the data onto a set of orthogonal (perpendicular) axes, called principal components, which capture the maximum variance in the data.

**Key Points**:
- **Variance Maximization**: PCA seeks to maximize the variance along the new axes.
- **Orthogonality**: The principal components are orthogonal to each other.
- **Eigenvalues and Eigenvectors**: The directions of maximum variance (principal components) are given by the eigenvectors of the data covariance matrix, and their corresponding eigenvalues indicate the amount of variance captured by each component.
- **Applications**: PCA is widely used in exploratory data analysis, noise reduction, and as a pre-processing step for other algorithms.

### 2. Linear Discriminant Analysis (LDA)
**Description**: LDA is a technique used for both dimensionality reduction and classification. It aims to project the data onto a lower-dimensional space in such a way that the class separability is maximized.

**Key Points**:
- **Class Separability**: LDA focuses on maximizing the ratio of between-class variance to the within-class variance.
- **Discriminant Axes**: The new axes (linear discriminants) are chosen to maximize the separation between different classes.
- **Supervised Technique**: Unlike PCA, LDA is supervised and takes class labels into account.
- **Applications**: LDA is used in pattern recognition, face recognition, and bioinformatics.

### 3. t-Distributed Stochastic Neighbor Embedding (t-SNE)
**Description**: t-SNE is a non-linear dimensionality reduction technique primarily used for the visualization of high-dimensional data. It converts similarities between data points into joint probabilities and tries to minimize the Kullback-Leibler divergence between these joint probabilities in a lower-dimensional space.

**Key Points**:
- **Preserving Local Structure**: t-SNE is particularly good at preserving the local structure of the data.
- **Non-linear Mapping**: It captures complex, non-linear relationships in the data.
- **Visualizing High-Dimensional Data**: t-SNE is commonly used to create 2D or 3D visualizations of high-dimensional datasets.
- **Computationally Intensive**: t-SNE can be slow and computationally expensive for large datasets.
- **Applications**: Data exploration, visualizing clusters in high-dimensional data, and understanding the structure of complex datasets.

### 4. Autoencoders
**Description**: Autoencoders are neural network-based models designed to learn efficient representations of the data. They consist of an encoder that compresses the data into a lower-dimensional latent space and a decoder that reconstructs the data from this latent space.

**Key Points**:
- **Unsupervised Learning**: Autoencoders are trained in an unsupervised manner.
- **Reconstruction Loss**: The training objective is to minimize the difference between the input data and its reconstruction.
- **Non-linear Capability**: Autoencoders can capture complex, non-linear relationships in the data.
- **Variations**: Variational Autoencoders (VAEs) and Denoising Autoencoders are popular variants.
- **Applications**: Feature learning, anomaly detection, image compression, and data denoising.

### Summary Table

| Technique                      | Type      | Key Feature                            | Applications                                          |
|-------------------------------|-----------|----------------------------------------|-------------------------------------------------------|
| **PCA**                       | Linear    | Maximizes variance along orthogonal axes | Exploratory data analysis, noise reduction, preprocessing |
| **LDA**                       | Linear    | Maximizes class separability             | Pattern recognition, face recognition, bioinformatics   |
| **t-SNE**                     | Non-linear| Preserves local structure               | Data visualization, understanding data structure        |
| **Autoencoders**              | Non-linear| Learns compressed representations       | Feature learning, anomaly detection, image compression  |


## 4. Data Augmentation
Data augmentation is a crucial technique in machine learning and deep learning used to increase the diversity of the training set without collecting new data. This helps in building more robust models and mitigates overfitting by providing the model with more varied examples to learn from. Here's a detailed look at various data augmentation techniques for different types of data:

### Image Augmentation

1. **Rotation**:
   - Rotates the image by a random or fixed angle.
   - Helps the model to become invariant to the orientation of objects.

2. **Flipping**:
   - Horizontally or vertically flips the image.
   - Helps the model learn that the object remains the same despite the flip.

3. **Cropping**:
   - Randomly crops a portion of the image.
   - Forces the model to focus on different parts of the image and not just the central region.

4. **Color Jittering**:
   - Randomly changes the brightness, contrast, saturation, and hue of the image.
   - Encourages the model to be robust to different lighting conditions and color variations.

5. **Adding Noise**:
   - Adds random noise to the image.
   - Helps the model become robust to noisy inputs.

### Text Augmentation

1. **Synonym Replacement**:
   - Randomly replaces words in the text with their synonyms.
   - Helps the model understand the context and meaning rather than just the specific words used.

2. **Random Insertion**:
   - Inserts random words from the existing text at random positions.
   - Increases the variability of the text data.

3. **Random Deletion**:
   - Randomly deletes words from the text.
   - Forces the model to learn to handle missing information and still understand the context.

4. **Back Translation**:
   - Translates the text to another language and then back to the original language.
   - Provides a paraphrased version of the original text, increasing variability.

### Time-series Augmentation

1. **Window Slicing**:
   - Extracts smaller windows or segments from the original time series data.
   - Allows the model to learn from different segments of the time series, capturing various patterns.

2. **Resampling**:
   - Changes the frequency of the time series data (e.g., upsampling or downsampling).
   - Helps the model handle different time scales and resolutions.

3. **Adding Noise**:
   - Adds random noise to the time series data.
   - Helps the model become robust to variations and anomalies in the data.

### Purpose of Data Augmentation

1. **Increases the Diversity of the Training Set**:
   - By artificially expanding the dataset with augmented samples, the model is exposed to a wider range of scenarios and variations, which can improve generalization.

2. **Helps in Building More Robust Models**:
   - Augmented data helps the model learn to recognize patterns and features under different conditions, making it more resilient to variations in real-world data.

3. **Mitigates Overfitting**:
   - By providing the model with more varied examples, data augmentation reduces the risk of the model memorizing the training data, which helps in achieving better performance on unseen data.

- **Purpose**
  - Increases the diversity of the training set
  - Helps in building more robust models
  - Mitigates overfitting

## 5. Modeling/Grid Search/Cross-validation
To delve deeper into modeling, grid search, and cross-validation, let's explore each concept and their applications:

### Types of Models

1. **Linear Regression**
   - **Definition**: A statistical method for modeling the relationship between a dependent variable and one or more independent variables.
   - **Applications**: Predicting continuous outcomes, like house prices, stock prices.

2. **Decision Trees**
   - **Definition**: A tree-like model of decisions and their possible consequences, including chance event outcomes, resource costs, and utility.
   - **Applications**: Classification and regression tasks, interpretability of models.

3. **Random Forest**
   - **Definition**: An ensemble learning method using multiple decision trees to improve the model's accuracy and control overfitting.
   - **Applications**: Both classification and regression problems.

4. **Support Vector Machines (SVM)**
   - **Definition**: A supervised learning model that analyzes data for classification and regression analysis by finding the hyperplane that best divides a dataset into classes.
   - **Applications**: Image recognition, text categorization.

5. **Neural Networks**
   - **Definition**: A series of algorithms that attempt to recognize underlying relationships in a set of data through a process that mimics the way the human brain operates.
   - **Applications**: Complex pattern recognition tasks, like image and speech recognition.

### Ensemble Methods

1. **Bagging (Bootstrap Aggregating)**
   - **Definition**: An ensemble method that improves the stability and accuracy of machine learning algorithms by training multiple models on different subsets of the data and averaging their predictions.
   - **Applications**: Reducing variance and helping to avoid overfitting.

2. **Boosting**
   - **Definition**: An ensemble technique that builds models sequentially, each new model correcting errors made by the previous models.
   - **Applications**: Improving the predictive accuracy of models, particularly in cases of weak models.

3. **Stacking**
   - **Definition**: An ensemble learning technique that combines multiple classification or regression models via a meta-classifier or meta-regressor.
   - **Applications**: Leveraging the strengths of multiple models to improve overall performance.

### Grid Search

- **Definition**: An exhaustive search method used to find the optimal hyperparameters for a model by evaluating all possible combinations of parameters specified in a grid.
- **Process**:
  1. Define the model and the hyperparameters grid.
  2. Perform a grid search by training the model with each combination of hyperparameters.
  3. Evaluate the performance using a scoring metric (e.g., accuracy, F1-score).
  4. Select the best combination of hyperparameters.
- **Applications**: Optimizing model performance, tuning hyperparameters for various algorithms.

### Cross-Validation

1. **k-Fold Cross-Validation**
   - **Definition**: A technique that involves dividing the dataset into k subsets (folds). The model is trained on k-1 folds and validated on the remaining fold, and this process is repeated k times.
   - **Applications**: Ensuring model robustness, preventing overfitting.
   - **Advantages**: Provides a good estimate of model performance by averaging results from multiple folds.

2. **Stratified k-Fold Cross-Validation**
   - **Definition**: A variant of k-fold cross-validation that ensures each fold has a proportional representation of all classes in the target variable.
   - **Applications**: Used especially in classification tasks where class distribution is imbalanced.
   - **Advantages**: Maintains class distribution, leading to more reliable validation results.

3. **Leave-One-Out Cross-Validation (LOOCV)**
   - **Definition**: A special case of k-fold cross-validation where k equals the number of data points in the dataset. Each iteration uses a single data point for validation and the rest for training.
   - **Applications**: Small datasets where other cross-validation methods might not be feasible.
   - **Advantages**: Provides a nearly unbiased estimate of model performance but is computationally intensive.

### Example Workflow

1. **Model Selection**: Choose the type of model (e.g., Random Forest).
2. **Hyperparameter Tuning**: Use grid search to find the best hyperparameters.
3. **Cross-Validation**: Apply k-fold or stratified k-fold cross-validation to evaluate the model.
4. **Model Evaluation**: Assess the final model's performance using metrics like accuracy, precision, recall, F1-score.

## 6. Visualization
- **Techniques**
  - Scatter plots
  - Line graphs
  - Bar charts
  - Histograms
  - Box plots
  - Heatmaps

- **Libraries**
  - Matplotlib
  - Seaborn
  - Plotly
  - Bokeh

- **Model Visualization**
  - Feature importance
  - Partial dependence plots
  - Confusion matrix
  - ROC/AUC curves

## 7. GPU Support
- **Purpose**
  - Accelerates computation for large-scale machine learning tasks
  - Essential for training deep learning models

- **Libraries**
  - CUDA
  - cuDNN
  - TensorFlow with GPU support
  - PyTorch with GPU support

- **Hardware**
  - NVIDIA GPUs
  - Cloud-based GPU services (AWS, Google Cloud, Azure)

## 8. Introduction to Distributed Architectures
- **Concept**
  - Distributes computational tasks across multiple machines

- **Techniques**
  - Data Parallelism: Splits data across multiple nodes, each node trains on its subset of data.
  - Model Parallelism: Splits the model across multiple nodes.

- **Frameworks**
  - Apache Spark
  - Hadoop
  - Horovod (for TensorFlow and PyTorch)

- **Benefits**
  - Handles large datasets
  - Reduces training time
  - Enables scalability

### Concept
Distributed architectures refer to systems that distribute computational tasks across multiple machines, enabling the system to process large datasets and complex computations more efficiently. These architectures leverage the combined power of several machines to achieve performance that would be difficult or impossible to achieve on a single machine.

### Techniques
Distributed architectures employ various techniques to manage and optimize the distribution of tasks. Two primary techniques include:

1. **Data Parallelism**: 
   - **Definition**: This technique involves splitting data across multiple nodes. Each node processes its subset of the data independently.
   - **Application**: Commonly used in training machine learning models where large datasets are divided among nodes to parallelize the training process.
   
2. **Model Parallelism**: 
   - **Definition**: This technique involves splitting the model itself across multiple nodes. Each node is responsible for computing a part of the model.
   - **Application**: Often used when models are too large to fit into the memory of a single machine, distributing different layers or parts of the model across multiple machines.

### Frameworks
Several frameworks facilitate the implementation of distributed architectures. Notable ones include:

1. **Apache Spark**:
   - A unified analytics engine for big data processing, with built-in modules for streaming, SQL, machine learning, and graph processing.
   - Provides an interface for programming entire clusters with implicit data parallelism and fault tolerance.

2. **Hadoop**:
   - An open-source framework that allows for the distributed processing of large data sets across clusters of computers using simple programming models.
   - Consists of modules like HDFS (Hadoop Distributed File System) and YARN (Yet Another Resource Negotiator) for resource management and job scheduling.

3. **Horovod**:
   - A distributed training framework for TensorFlow and PyTorch.
   - Simplifies the process of scaling training across multiple GPUs and nodes by using a ring-allreduce algorithm for efficient communication.

### Benefits
Distributed architectures offer several significant benefits:

1. **Handles Large Datasets**:
   - Enables the processing and analysis of massive datasets that exceed the capacity of a single machine, making it possible to work with big data efficiently.

2. **Reduces Training Time**:
   - By parallelizing tasks across multiple machines, the overall computation time is significantly reduced, speeding up processes such as model training and data analysis.

3. **Enables Scalability**:
   - Provides the ability to scale resources up or down based on demand, allowing systems to handle varying workloads effectively and efficiently.

## 9. Scikit-learn Tools for ML Architectures
- **Data Preprocessing**
  - `StandardScaler`, `MinMaxScaler`, `Normalizer`
  - `LabelEncoder`, `OneHotEncoder`
  - `Imputer` for missing values

- **Model Selection**
  - `GridSearchCV`, `RandomizedSearchCV`
  - `train_test_split`
  - `cross_val_score`

- **Feature Selection**
  - `SelectKBest`, `RFE` (Recursive Feature Elimination)

- **Dimensionality Reduction**
  - `PCA`, `LDA`

- **Pipeline**
  - `Pipeline` class for chaining multiple processing steps

- **Model Evaluation**
  - `accuracy_score`, `precision_score`, `recall_score`, `f1_score`
  - `confusion_matrix`
  - `roc_curve`, `auc`
### Data Preprocessing

1. **StandardScaler**
   - **Purpose**: Standardizes features by removing the mean and scaling to unit variance.
   - **Use Case**: Useful for algorithms like SVM or KNN which are sensitive to the scale of the data.
   - **Code Example**:
     ```python
     from sklearn.preprocessing import StandardScaler
     scaler = StandardScaler()
     X_scaled = scaler.fit_transform(X)
     ```

2. **MinMaxScaler**
   - **Purpose**: Transforms features by scaling each feature to a given range (default is 0 to 1).
   - **Use Case**: Commonly used for algorithms requiring normalized input like neural networks.
   - **Code Example**:
     ```python
     from sklearn.preprocessing import MinMaxScaler
     scaler = MinMaxScaler()
     X_scaled = scaler.fit_transform(X)
     ```

3. **Normalizer**
   - **Purpose**: Normalizes samples individually to have unit norm.
   - **Use Case**: Often used in text classification or clustering to handle sparse data.
   - **Code Example**:
     ```python
     from sklearn.preprocessing import Normalizer
     normalizer = Normalizer()
     X_normalized = normalizer.fit_transform(X)
     ```

4. **LabelEncoder**
   - **Purpose**: Encodes target labels with value between 0 and n_classes-1.
   - **Use Case**: Useful for converting categorical labels to numeric values.
   - **Code Example**:
     ```python
     from sklearn.preprocessing import LabelEncoder
     encoder = LabelEncoder()
     y_encoded = encoder.fit_transform(y)
     ```

5. **OneHotEncoder**
   - **Purpose**: Encodes categorical integer features as a one-hot numeric array.
   - **Use Case**: Useful for converting categorical features to a format that can be provided to ML algorithms.
   - **Code Example**:
     ```python
     from sklearn.preprocessing import OneHotEncoder
     encoder = OneHotEncoder()
     X_encoded = encoder.fit_transform(X)
     ```

6. **Imputer (SimpleImputer)**
   - **Purpose**: Imputes missing values using a specified strategy (mean, median, most frequent, etc.).
   - **Use Case**: Useful for datasets with missing values.
   - **Code Example**:
     ```python
     from sklearn.impute import SimpleImputer
     imputer = SimpleImputer(strategy='mean')
     X_imputed = imputer.fit_transform(X)
     ```

### Model Selection

1. **GridSearchCV**
   - **Purpose**: Exhaustive search over specified parameter values for an estimator.
   - **Use Case**: Finds the best hyperparameters for a model.
   - **Code Example**:
     ```python
     from sklearn.model_selection import GridSearchCV
     param_grid = {'param1': [1, 10], 'param2': [0.1, 0.01]}
     grid_search = GridSearchCV(estimator, param_grid, cv=5)
     grid_search.fit(X, y)
     ```

2. **RandomizedSearchCV**
   - **Purpose**: Randomized search on hyperparameters.
   - **Use Case**: More efficient than GridSearchCV when the parameter space is large.
   - **Code Example**:
     ```python
     from sklearn.model_selection import RandomizedSearchCV
     param_dist = {'param1': [1, 10], 'param2': [0.1, 0.01]}
     random_search = RandomizedSearchCV(estimator, param_dist, cv=5, n_iter=10)
     random_search.fit(X, y)
     ```

3. **train_test_split**
   - **Purpose**: Splits arrays or matrices into random train and test subsets.
   - **Use Case**: Used to evaluate the performance of a model.
   - **Code Example**:
     ```python
     from sklearn.model_selection import train_test_split
     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
     ```

4. **cross_val_score**
   - **Purpose**: Evaluates a score by cross-validation.
   - **Use Case**: Provides a more reliable estimate of model performance by averaging over multiple splits.
   - **Code Example**:
     ```python
     from sklearn.model_selection import cross_val_score
     scores = cross_val_score(estimator, X, y, cv=5)
     ```

### Feature Selection

1. **SelectKBest**
   - **Purpose**: Selects features according to the k highest scores.
   - **Use Case**: Useful for reducing the dimensionality of the data.
   - **Code Example**:
     ```python
     from sklearn.feature_selection import SelectKBest, f_classif
     selector = SelectKBest(f_classif, k=10)
     X_selected = selector.fit_transform(X, y)
     ```

2. **RFE (Recursive Feature Elimination)**
   - **Purpose**: Recursively removes features and builds a model on remaining attributes.
   - **Use Case**: Identifies the most important features for model accuracy.
   - **Code Example**:
     ```python
     from sklearn.feature_selection import RFE
     selector = RFE(estimator, n_features_to_select=10)
     X_selected = selector.fit_transform(X, y)
     ```

### Dimensionality Reduction

1. **PCA (Principal Component Analysis)**
   - **Purpose**: Reduces the dimensionality of the data while retaining most of the variance.
   - **Use Case**: Helps in visualizing high-dimensional data and speeding up ML algorithms.
   - **Code Example**:
     ```python
     from sklearn.decomposition import PCA
     pca = PCA(n_components=2)
     X_pca = pca.fit_transform(X)
     ```

2. **LDA (Linear Discriminant Analysis)**
   - **Purpose**: Finds the linear combinations of features that best separate classes.
   - **Use Case**: Useful for classification tasks.
   - **Code Example**:
     ```python
     from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
     lda = LDA(n_components=1)
     X_lda = lda.fit_transform(X, y)
     ```

### Pipeline

1. **Pipeline class**
   - **Purpose**: Chains multiple steps into a single unit.
   - **Use Case**: Ensures that all processing steps are applied consistently during training and testing.
   - **Code Example**:
     ```python
     from sklearn.pipeline import Pipeline
     pipeline = Pipeline([
         ('scaler', StandardScaler()),
         ('classifier', SVC())
     ])
     pipeline.fit(X_train, y_train)
     ```

### Model Evaluation

1. **accuracy_score**
   - **Purpose**: Computes the accuracy classification score.
   - **Use Case**: Evaluates the proportion of correctly classified samples.
   - **Code Example**:
     ```python
     from sklearn.metrics import accuracy_score
     accuracy = accuracy_score(y_true, y_pred)
     ```

2. **precision_score**
   - **Purpose**: Computes the precision.
   - **Use Case**: Evaluates the proportion of true positives among the predicted positives.
   - **Code Example**:
     ```python
     from sklearn.metrics import precision_score
     precision = precision_score(y_true, y_pred)
     ```

3. **recall_score**
   - **Purpose**: Computes the recall.
   - **Use Case**: Evaluates the proportion of true positives among the actual positives.
   - **Code Example**:
     ```python
     from sklearn.metrics import recall_score
     recall = recall_score(y_true, y_pred)
     ```

4. **f1_score**
   - **Purpose**: Computes the F1 score, the harmonic mean of precision and recall.
   - **Use Case**: Provides a balance between precision and recall.
   - **Code Example**:
     ```python
     from sklearn.metrics import f1_score
     f1 = f1_score(y_true, y_pred)
     ```

5. **confusion_matrix**
   - **Purpose**: Computes the confusion matrix.
   - **Use Case**: Provides a detailed breakdown of true positives, false positives, true negatives, and false negatives.
   - **Code Example**:
     ```python
     from sklearn.metrics import confusion_matrix
     cm = confusion_matrix(y_true, y_pred)
     ```

6. **roc_curve**
   - **Purpose**: Computes the receiver operating characteristic (ROC) curve.
   - **Use Case**: Evaluates the performance of a binary classifier at different threshold settings.
   - **Code Example**:
     ```python
     from sklearn.metrics import roc_curve
     fpr, tpr, thresholds = roc_curve(y_true, y_score)
     ```

7. **auc (Area Under the Curve)**
   - **Purpose**: Computes the area under the ROC curve.
   - **Use Case**: Provides a single metric to evaluate the performance of a binary classifier.
   - **Code Example**:
     ```python
     from sklearn.metrics import auc
     roc_auc = auc(fpr, tpr)
     ```
