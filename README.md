# A). Decentral Smart Grid Control (DSGC)

The file Smart-Grid-nominal-power-study.ipynb located within the ML_Project/ml_project_notebook/ directory evaluates Decentral Smart Grid Control (DSGC) systems. It models stability and power dynamics across decentralized energy networks.
# Objectives of this notebook

The core technical and pedagogical objectives of this notebook include:

* **Decentralized Grid Physics:** Understanding the 4-node star architecture (comprising one central power supply source and three localized consumer nodes).
* **Stability Optimization:** Modeling how nominal power production, consumption fluctuations, and participant response times alter AC grid frequency stability.
* **Price Elasticity Mechanics:** Evaluating economic-engineering feedback loops, specifically tracking how consumers alter their energy footprints in response to real-time grid price adjustments.
* **Advanced Feature Interactions:** Extracting non-linear operational bounds where structural load balances break, causing systemic power grid failure.
# Detailed Summary of the Notebook

The notebook is divided into structured phases following a production-grade machine learning workflow:

### 1. Theoretical Setup & Math Framework

The notebook starts by implementing the **Decentral Smart Grid Control (DSGC)** mathematical framework. Instead of relying on a rigid, top-down traditional distribution schema, it analyzes a modern two-way smart framework where price dictates load balance. Stability is treated as a function of grid frequency deviations: if production and consumption drift out of sync, the system risks cascading brownouts.

### 2. Feature & Attribute Analysis

The notebook utilizes the standard 14-attribute smart grid stability dataset (typically sourced from Vadim Arzamasov / Kaggle). It breaks down twelve structural predictors to isolate their impact on nominal power:

* **Tau ($\tau_1$ to $\tau_4$):** Reaction time coefficients for each network participant adjusting to price signals.
* **Power ($p_1$ to $p_4$):** Nominal power values. $p_1$ represents the primary generator source (always positive), while $p_2, p_3, p_4$ handle consumer nodes (always negative, indicating consumption).
* **Gamma ($\gamma_1$ to $\gamma_4$):** Price elasticity parameters mapping participant sensitivity to electricity price variations.
### 3. Statistical Analysis & Feature Engineering

The notebook processes the feature space to separate target definitions. It evaluates how variances in the continuous root features map to system traits:

* **Mathematical Target Balancing:** Inspecting the continuous stability root differential (`stab`) and its binary classification counterpart (`stabf`: stable vs. unstable).
* **Multicollinearity Checks:** Evaluating cross-correlations between the nominal power consumption levels of the consumer nodes ($p_2, p_3, p_4$) and their physical impact on overall network resilience.
### 4. Baseline Modeling & Behavior Insights

By evaluating how the system balances nominal power constraints against pricing elasticity, this study establishes reference criteria for predictive classifiers. It highlights that **participant response times ($\tau$)** and **load volumes ($p$)** serve as the leading indicators for capturing critical thresholds where localized grids flip from operational equilibria to unstable structural failures.

# B). diagnostics for linear regression using the classic Auto MPG dataset

The file auto-mpg-regression-outliers-residuals.ipynb located within the ML_Project/ml_project_notebook/ directory focuses on classical diagnostics for linear regression using the classic Auto MPG dataset. It serves as an end-to-end practical study on how anomalies and residual patterns impact standard Ordinary Least Squares (OLS) models.

# Objectives of this notebook

The core statistical and diagnostic objectives of this notebook include:

* **Residual Analysis Mechanics:** Learning how to visually and statistically evaluate residuals to verify standard OLS assumptions (linearity, homoscedasticity, and normality).
* **Outlier & Leverage Identification:** Detecting data points with extreme feature values (high leverage) or extreme target errors (outliers) that skew predictive metrics.
* **Influential Observation Metrics:** Calculating quantitative mathematical metrics like Cook’s Distance and DFBETAS to identify specific data points that disproportionately alter model coefficients.
* **Model Remediation:** Understanding when and how to safely transform variables or prune specific anomalies to build stable, reliable regressions.
# Detailed Summary of the Notebook

The notebook walks through a rigorous workflow designed to critique and improve a baseline linear model:

### 1. Exploratory Data Analysis & Baseline OLS

* **Data Prep:** The notebook loads vehicle specifications (such as cylinders, displacement, horsepower, weight, and acceleration) to predict fuel efficiency (`mpg`).
* **Initial Model Fitting:** A baseline multiple linear regression model is trained using OLS. The initial summary parameters ($R^2$, adjusted $R^2$, and $t$-statistics) are generated to establish a benchmarking reference.
### 2. Residual Diagnostics

The notebook systematically evaluates the errors of the baseline model to check for violations of OLS assumptions:

* **Non-linearity & Heteroscedasticity:** Plots of *Residuals vs. Fitted Values* are generated. Curvatures in this plot signal missing polynomial terms (non-linearity), while a funnel-like expansion signals changing error variance (heteroscedasticity).
* **Non-Normality:** Quantile-Quantile (Q-Q) plots compare the sample residual distribution against a theoretical normal distribution to identify heavy tails or skewed error distributions.
### 3. High Leverage and Outlier Detection

The notebook isolates unusual observations by separating their geometric position in the feature space from their actual target errors:

* **Leverage (Hat Matrix):** Computes the hat values ($h_{ii}$) to flag vehicles with extreme or rare technical specifications that exert high geometric leverage over the orientation of the regression line.
* **Studentized Residuals:** Residuals are divided by their estimated standard deviation. Data points with studentized residuals exceeding $\pm2$ or $\pm3$ are flagged as statistical outliers.
### 4. Quantifying Influential Observations

To bridge the gap between leverage and outliers, the notebook calculates metrics to pinpoint observations that single-handedly shift the model's coefficients:

* **Cook's Distance ($D_i$):** Measures the aggregate shift in *all* predicted values when a specific vehicle is deleted from the training pool. Points exceeding the classic threshold of $4/n$ are flagged for review.
* **DFBETAS:** Examines exactly how much an individual regression coefficient ($\beta_j$) changes when a specific outlier is excluded, isolating exactly which vehicle features are most sensitive to anomalous data.
### 5. Model Refinement & Comparison

* The notebook demonstrates the downstream effect of removing highly influential, corrupted, or unrepresentative data points.
* By refitting the OLS model on the filtered data, it illustrates how diagnostic pruning stabilizes standard errors, rectifies residual distributions, and yields a generalized, trustworthy regression equation.

# C). Ridge Regression:
The file classical-ridge-regression-for-housing.ipynb located within the ML_Project/ml_project_notebook/ directory focuses on mitigating overfitting and handling multicollinearity using L2 regularization. It provides a step-by-step framework for transitioning from Ordinary Least Squares (OLS) to Ridge Regression when working with highly correlated spatial and structural housing attributes.
# Objectives of this notebook

The core statistical and implementation objectives of this notebook include:

* **Regularization Mechanics:** Understanding the mathematical theory of adding an L2 penalty term (squared magnitude of coefficients) to the OLS loss function.
* **Multicollinearity Mitigation:** Learning how Ridge regression stabilizes coefficient estimates and reduces variance when feature vectors are collinear.
* **Hyperparameter Tuning ($\alpha$ / $\lambda$):** Implementing validation loops (such as $k$-fold cross-validation) to locate the optimal regularization strength.
* **Bias-Variance Trade-off:** Visualizing and assessing how introducing a small amount of bias into the parameter estimates drops the overall model variance.
# Detailed Summary of the Notebook

The notebook follows a systematic applied machine learning pipeline to compare baseline models against their regularized variants:

### 1. Exploratory Data Analysis & Feature Scaling

* **Dataset Ingestion:** The notebook processes housing indicators (typically variables like square footage, bedrooms, bathrooms, and location metrics) to map real estate market valuations.
* **Standardization:** Because Ridge regression penalizes the squared magnitude of coefficients, features on larger scales (e.g., total square feet) will be unfairly penalized compared to smaller scales (e.g., bedroom count). The notebook scales the feature matrix $X$ using standard scores ($z$-scores) before entering the model framework.
### 2. The Limits of Baseline OLS

* **Unconstrained Fitting:** An initial standard multiple linear regression model is trained.
* **Variance Evaluation:** The notebook inspects the baseline's susceptibility to multicollinearity. When predictors track too closely together, standard errors inflate, making the resulting OLS model overly sensitive to random noise in the training subset.
### 3. Transition to Ridge Regression (L2 Regularization)

The notebook introduces the Ridge cost minimization objective function:

$$\text{Loss} = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + \alpha \sum_{j=1}^{m} \beta_j^2$$

* **Shrinkage Action:** It illustrates how the penalty multiplier ($\alpha$ or $\lambda$) shrinks the magnitude of the $\beta$ coefficients asymptotically toward zero. This prevents any single feature from capturing disproportionate leverage over the system.
### 4. Hyperparameter Tuning and Cross-Validation

* **Alpha Path Inspection:** The notebook plots the *regularization path*, showing how individual feature coefficients drop as $\alpha$ increases from $0$ to large values.
* **Validation:** Utilizing tools like `RidgeCV` or manual $k$-fold cross-validation loops, it calculates evaluation parameters (such as Root Mean Squared Error (RMSE) or $R^2$) across a spectrum of alpha possibilities to select the parameter that yields the highest generalization on test data.

# D). comprehensive data science workflow on housing project
housing-project-statistical-data-analysis.ipynb Primary Learning Goal, Design, build, and optimize an end-to-end predictive statistical framework using multi-variable regression models.

## Detailed Summary & Analysis Methods

The `housing-project-statistical-data-analysis.ipynb` notebook implements a comprehensive data science workflow to extract insights and predict housing market valuations. It leverages an analytical approach structured across the following methodologies:

### 1. Preprocessing & Descriptive Analysis

* **Statistical Summaries**: Employs structural functions (e.g., `describe()`, `info()`) to map out numeric types, dataset scope, and missing entries.
* **Feature Scale Check**: Assesses the necessity of scaling metrics to standardise independent variables across differing units.

### 2. Visual & Structural EDA

* **Scatterplot Matrix & Pair Plots**: Visualises multidimensional feature spaces to map out initial linear or non-linear trend directions (mental visualization).
* **Trend Analysis**: Evaluates the immediate visual impact of continuous features (such as spatial size or regional location indicators) on the target housing valuations.

### 3. Data Cleansing & Outlier Filtering

* **Tukey’s IQR Method**: Computes the Interquartile Range ($\text{IQR} = Q3 - Q1$) to identify anomalies. Data points falling below $Q1 - 1.5 \times \text{IQR}$ or exceeding $Q3 + 1.5 \times \text{IQR}$ are isolated and pruned.
* **Boxplot Verification**: Uses distribution boxplots to visually cross-verify the presence, volume, and elimination of extreme market anomalies.
### 4. Multicollinearity & Diagnostic Checks

* **Predictor Independence**: Checks for high inter-correlation between separate independent features. This safeguards the stability of the regression coefficients against confounding variances.
* **Feature Interaction Mapping**: Checks distribution matrices to guarantee that predictive elements accurately capture individual market signals.
### 5. Inferential & Predictive Regression Modeling

* **Supervised Linearity Mapping**: Employs Simple and Multiple Linear Regression models ($\text{Price} = \beta_0 + \beta_1 X_1 + \dots + \epsilon$) to establish explicit mathematical relationships.
* **Validation Benchmarking**: Splits datasets systematically into separate training sets (model learning phase) and validation test partitions.
* **Goodness-of-Fit Verification**: Measures the ultimate predictive success through explicit statistical metrics:
    * **Coefficient of Determination ($R^2$)**: Quantifies the percentage of target housing variance explained by the features.
    * **Mean Absolute Error (MAE) & Root Mean Squared Error (RMSE)**: Pinpoints error variances and penalises larger deviations to reflect real-world financial miscalculations.

# E). Design, train, and optimize Deep Learning neural networks  on smart grid:
Design, train, and optimize Deep Learning neural networks for regression tasks, specifically focusing on grid power prediction and automated hyperparameter tuning frameworks.
## Detailed Summary & Hyperparameter Tuning Methods

The `neural-models-for-grid-power.ipynb` notebook shifts away from traditional statistical models to explore non-linear relationship modeling using Artificial Neural Networks (ANNs) for smart-grid nominal power tracking.

### 1. Core Objectives & Workflow

* **Problem Domain**: Predict nominal power outputs within a smart-grid infrastructure using system parameters.
* **Pipeline Structure**: Features typical deep learning steps including data preparation, tensor reshaping, network compilation, validation splitting, and evaluation tracking.

### 2. Hyperparameter Tuning & Modeling Methods

To find the most efficient model architecture and avoid manual trial-and-error, the notebook implements automated structural optimizations:
* **Framework Utilities**: Uses tuning wrappers (such as `Keras Tuner` or custom `scikit-learn` grid search hooks) to systematically iterate through architectural variations.
* **Search Space Optimization**:
    * **Layer Depth & Width**: Dynamic testing of the number of hidden layers and the number of neurons per layer (e.g., searching between 32 and 512 nodes).
    * **Activation Functions**: Benchmarking non-linear activators like `ReLU` or `LeakyReLU` to prevent vanishing gradient issues.
    * **Optimization & Learning Rates**: Tuning optimizers (like `Adam` or `RMSprop`) alongside variable learning rates (e.g., $10^{-2}$ to $10^{-4}$) to guarantee smooth loss convergence.
    * **Regularization Over Dropout**: Adjusting dropout percentages (e.g., 0.1 to 0.5) and weight decay ($L_2$ regularization) dynamically to curb overfitting on grid noise.
### 3. Convergence & Training Evaluation

* **Loss Trackers**: Uses Mean Squared Error (MSE) or Mean Absolute Error (MAE) as target loss objectives during structural search iterations.
* **Early Stopping Integration**: Utilizes callback hooks to halt tuning trials early if validation loss plateaus, preserving computing resources.

# F). Outlier-detection with robust mathematical filtering:
Understand and implement robust mathematical filtering algorithms to identify, isolate, and remove anomalies in real estate datasets before statistical modeling.
## Detailed Summary & Mathematical Approach

The `outlier-detection-in-house-prices.ipynb` notebook focuses heavily on the diagnostic side of data science. It demonstrates how extreme mathematical deviations distort linear trends and how to systematically clean them.

### 1. Core Objectives & Workflow

* **Problem Domain**: Eliminating distribution skewness caused by extreme real estate valuations (luxury estates, data errors, or forced sales).
* **Pipeline Structure**: Features baseline data distribution checks, parallel mathematical outlier filtering, and a final look at how these actions change regression stability.
### 2. Mathematical Approaches for Outlier Detection

The notebook applies specific statistical frameworks to measure and flag extreme values:

* **Z-Score Method (Parametric)**:
    * Assumes a roughly Gaussian (normal) distribution of house prices.
    * Calculates the distance of a data point from the mean using standard deviations:

$$Z = \frac{x - \mu}{\sigma}$$

    * *Where*: $x$ is the asset value, $\mu$ is the population mean, and $\sigma$ is the standard deviation.
    * Points with an absolute value $|Z| > 3$ are flagged as statistical outliers.
* **Tukey’s Interquartile Range (IQR) Method (Non-Parametric)**:
    * Does not assume normal distribution, making it highly effective for skewed real estate data.
    * Calculates the spread between the 75th percentile ($Q_3$) and the 25th percentile ($Q_1$):

$$\text{IQR} = Q_3 - Q_1$$

    * Establishes lower and upper mathematical boundaries:

$$\text{Lower Bound} = Q_1 - 1.5 \times \text{IQR}$$

$$\text{Upper Bound} = Q_3 + 1.5 \times \text{IQR}$$

    * Data points outside this envelope are systematically pruned.
* **Mahalanobis Distance (Multivariate Tracking)**:
    * Looks beyond single-column limits to find structural outliers across multiple dimensions at once (e.g., a tiny 1-bedroom house costing $5 million).
    * Evaluates distance while accounting for the covariance matrix ($\Sigma$) of the features:

$$D_M(x) = \sqrt{(x - \mu)^T \Sigma^{-1} (x - \mu)}$$
### 3. Impact Assessment

* **Skewness & Kurtosis Check**: Measures the asymmetry and peakedness of the price distributions before and after applying the formulas.
* **Residual Variance Improvement**: Demonstrates how shrinking your dataset to normal mathematical bounds decreases error variance in ordinary least squares (OLS) estimations.

# G).  Generalized Linear Models (GLMs)
 Understand and apply Generalized Linear Models (GLMs) to transcend ordinary linear regressions, mapping target boundaries mathematically to handle non-Gaussian errors or constrained outputs.

## Detailed Summary & Mathematical Approach

The `prediction-with-glm-statistical-analysis.ipynb` notebook implements robust parametric modeling using the GLM framework (typically powered by `statsmodels.api.GLM`). This notebook addresses scenarios where standard Ordinary Least Squares (OLS) assumptions fall short—specifically when error residuals do not display homoscedasticity or a strict normal distribution.

### 1. Core Objectives & Workflow

* **Problem Domain**: Modeling target conditions bounded by non-linear constraints (such as non-negative valuations or bounded probabilities) using structured real estate or risk features.
* **Pipeline Structure**: Encompasses structural data mapping, defining exponential family targets, maximum likelihood estimations, and diagnostic deviance checks.
### 2. Mathematical Foundations of GLM

Unlike classic linear algorithms ($Y = X\beta + \epsilon$), GLMs relax structural boundaries by defining three strict mathematical components:

* **The Random Component (Target Conditional Distribution)**:
    The conditional distribution of the target variable $Y$, given predictors $X$, must belong to the Exponential Dispersion Family. Its probability density function is written mathematically as:

$$f(y; \theta, \phi) = \exp \left( \frac{y\theta - b(\theta)}{a(\phi)} + c(y, \phi) \right)$$

    * *Where*: $\theta$ represents the natural (canonical) parameter, $\phi$ is the dispersion parameter, and $b(\theta)$ is a log-partition function tracking the conditional mean ($\mu = E[Y] = b'(\theta)$). Common target choices configured inside the repository include Gaussian, Gamma, or Poisson profiles.
* **The Systematic Component (Linear Predictor)**:
    Combines independent variables into a standardized vector notation, mapping individual weights to input metrics:

$$\eta = X^T\beta = \beta_0 + \beta_1 X_1 + \dots + \beta_k X_k$$

* **The Link Function ($g(\cdot)$)**:
    Connects the systematic linear component $\eta$ directly to the expected target mean $\mu$:

$$\eta = g(\mu) \implies \mu = g^{-1}(X^T\beta)$$
The notebook leverages specific canonical links depending on data structure goals:

*   **Log Link ($g(\mu) = \ln(\mu)$)**: Used for continuous non-negative pricing or count outputs to ensure exponentiated structural parameters ($\mu = e^{X^T\beta}$) are strictly positive.
*   **Logit Link ($g(\mu) = \ln(\frac{\mu}{1-\mu})$)**: Standard for binary/fractional probabilities mapping outputs explicitly inside an open $[0, 1]$ boundary.

### 3. Mathematical Optimization & Diagnostics

*   **Iteratively Reweighted Least Squares (IRLS)**: Since non-Gaussian GLMs often do not yield exact closed-form analytical solutions, parameters ($\hat{\beta}$) are optimized using numerical Maximum Likelihood Estimation (MLE) iteratively via Newton-Raphson or IRLS routines.
*   **Goodness-of-Fit Deviance**: Rather than tracking standard residual variance sums, the algorithm determines structural errors using Deviance ($D$), calculating log-likelihood deviations ($L$) relative to a perfect saturated model ($L_{sat}$):

$$D = -2 \left( \ell(\hat{\beta}) - \ell_{sat} \right)$$

*   **Pearson Residual Scale Check**: Scaled to check variations across fluctuating conditional targets:

$$r_P = \frac{y_i - \hat{\mu}_i}{\sqrt{\text{Var}(\hat{\mu}_i)}}$$

# H). Parametric regression modeling:
Master parametric regression modeling and structural hypothesis testing to evaluate smart-grid load constraints and predict continuous nominal power values.
## Detailed Summary & Mathematical Approach

The `statistical-models-for-grid-power.ipynb` notebook implements a comprehensive statistical framework using traditional linear estimators (typically via `statsmodels.api.OLS` and `scikit-learn`). It serves as a rigorous parametric baseline for smart-grid nominal power tracking, ensuring that structural data properties satisfy core econometric assumptions.

### 1. Core Objectives & Workflow

*   **Problem Domain**: Predict continuous electrical grid nominal power output using structural operational parameters (e.g., system thermal metrics, node variables, or supply load capacity).
*   **Pipeline Structure**: Features comprehensive data parsing, structural diagnostic checks, Ordinary Least Squares (OLS) coefficient optimization, and statistical validation reporting.
### 2. Mathematical Modeling Framework

The notebook approaches grid power prediction using classical parametric equations rather than black-box algorithms, maintaining full structural interpretability:

*   **Multiple Linear Regression (MLR)**:
    Models the nominal power output $Y$ as a linear combination of system predictors $X_i$ paired with a Gaussian error term:

$$Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \dots + \beta_k X_k + \epsilon$$

    *Where*: $\beta_0$ represents the intercept, $\beta_i$ represent the calculated feature effects (coefficients), and $\epsilon$ represents the structural error vector.

*   **Ordinary Least Squares (OLS) Estimation**:
    The parameter vector $\hat{\beta}$ is computed analytically by minimizing the Sum of Squared Residuals (SSR):
$$\min_{\beta} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 \implies \hat{\beta} = (X^T X)^{-1} X^T Y$$

This requires the matrix $(X^T X)$ to be non-singular (invertible), highlighting the importance of the notebook's multicollinearity tracking.

### 3. Gauss-Markov Diagnostic Testing

To guarantee that the calculated OLS parameters represent the **Best Linear Unbiased Estimator (BLUE)**, the pipeline tests foundational mathematical assumptions:

*   **Homoscedasticity Assessment**:
    Checks that the variance of the error terms remains constant across all predicted power ranges:

$$\text{Var}(\epsilon_i | X) = \sigma^2$$
This is validated mathematically via the **Breusch-Pagan** or **White test**, checking if residual squares correlate with the input features.

*   **Normality of Residuals**:
    Ensures error profiles follow a standard Gaussian curve ($\epsilon \sim N(0, \sigma^2)$) to preserve valid hypothesis testing. This is measured using the **Jarque-Bera** statistic, which tests distribution shape deviations:

$$JB = \frac{n}{6} \left( S^2 + \frac{(K - 3)^2}{4} \right)$$

    *Where*: $S$ represents sample skewness and $K$ represents sample kurtosis.

*   **Multicollinearity Flagging via VIF**:
    Calculates the Variance Inflation Factor (VIF) for each grid parameter to ensure independent feature signals are not distorted by cross-correlations:

$$\text{VIF}_i = \frac{1}{1 - R_i^2}$$
Features yielding a $\text{VIF} > 5$ to $10$ are flagged for removal to protect coefficient stability.

### 4. Inferential Significance Testing

*   **Student’s t-Test**: Evaluates individual feature impact by asserting the null hypothesis $H_0: \beta_i = 0$ against the calculated standard error ($SE$):

$$t = \frac{\hat{\beta}_i}{SE(\hat{\beta}_i)}$$

*   **F-Test (Overall Fit)**: Evaluates whether the collection of variables predicts nominal grid capacity significantly better than a simple mean baseline model.

# I). : Apply Principal Component Analysis (PCA) 
 Apply Principal Component Analysis (PCA) to resolve multi-variable correlation instabilities and eliminate multicollinearity within vehicle performance datasets.
 ## Detailed Summary & Mathematical Approach

The `visual-pca-multicollinearity-in-auto-mpg.ipynb` notebook implements an unsupervised dimensionality reduction pipeline. It directly targets datasets like the classic *Auto MPG* dataset, where engineering attributes (e.g., cylinders, displacement, horsepower, weight) suffer from massive inter-correlations that destabilize Ordinary Least Squares (OLS) regression models.

### 1. Core Objectives & Workflow

*   **Problem Domain**: Mitigating standard error inflation and unstable parameter estimation caused by high multicollinearity among physical engine design features.
*   **Pipeline Structure**: Encompasses data standardization, covariance matrix formulation, eigenvalue decomposition, variance interpretation visualization, and a final transformed regression mapping.

### 2. Mathematical Framework: Principal Component Analysis (PCA)

To extract orthogonal (uncorrelated) signals, the pipeline systematically shifts the coordinate system of the original features through specific matrix algebra steps:

*   **Z-Score Feature Standardization**:
    Because PCA is highly sensitive to the scaling and units of features, data columns are centered and scaled to have a mean ($\mu$) of 0 and a standard deviation ($\sigma$) of 1:

$$Z = \frac{X - \mu}{\sigma}$$

*   **Covariance Matrix Computation**:
    The algorithm constructs an $n \times n$ covariance matrix ($\Sigma$) from the scaled matrix $Z$ to quantify the structural linkages between every pair of features:

$$\Sigma = \frac{1}{m - 1} Z^T Z$$
    *Where*: $m$ is the total number of vehicles/samples, and $n$ is the number of continuous engine metrics.

*   **Eigendecomposition (Spectral Mapping)**:
    The notebook solves the fundamental characteristic equation to break the covariance structure down into its core directional components:

$$\Sigma v = \lambda v \implies (\Sigma - \lambda I)v = 0$$

    *   **Eigenvalues ($\lambda_i$)**: Scalar measures representing the amount of data variance captured along each newly derived axis.
    *   **Eigenvectors ($v_i$)**: Orthogonal directional vectors (loadings) used to project the original features into the new coordinate space.

*   **Dimensionality Selection & Variance Distribution**:
    The script calculates the Explained Variance Ratio (EVR) for each principal component to choose how many dimensions to keep while minimizing information loss:
$$\text{EVR}_i = \frac{\lambda_i}{\sum_{j=1}^{n} \lambda_j}$$

This distribution is plotted on a visual Scree Plot, allowing engineers to retain the top $k$ components (where $k < n$) that capture the vast majority of the total dataset variance.

### 3. Resolving Multicollinearity for Regression

*   **Orthogonal Projection Transformation**:
    The original standardized data is multiplied by the chosen feature weight matrix ($W_k$) containing the top $k$ eigenvectors:

$$\text{Scores} = Z \cdot W_k$$
*   **Eliminating Covariance**:
    By design, the cross-correlation between these newly generated component vectors drops to absolute zero ($\text{Cov}(PC_i, PC_j) = 0$ for all $i \neq j$).

*   **Variance Inflation Factor (VIF) Optimization**:
    Running a regression model on these uncorrelated principal components reduces every single feature's VIF score down to a perfect baseline score of 1.0. This completely eliminates multi-variable instability while preserving strong predictive power for forecasting vehicle fuel efficiency (MPG).
 
### 5. Performance Benchmarking

* **Final Comparisons:** The regularized Ridge model is evaluated side-by-side with the unconstrained OLS model across test sets.
* **Key Insight:** The notebook highlights that while Ridge regression incurs slightly higher bias on the training split, it yields a major drop in testing error (RMSE), making it a much more robust option for real-world real estate deployment.

