# Machine Learning for Heart Disease Risk Prediction

This project investigates the prediction of heart disease using a dataset from Kaggle by comparing the performance of various machine learning models, including a custom neural network implemented in PyTorch. The primary goal was to understand which model architectures and hyperparameters yield the best performance and to provide insights into the strengths and limitations of each approach in the context of healthcare. <br />

Heart disease is a leading cause of mortality worldwide, and early detection is crucial for improving patient outcomes. With the growing availability of patient data, there is an opportunity to leverage machine learning and neural networks to develop predictive models that can assist clinicians in making more informed decisions. This project was undertaken to explore the potential of these models in accurately predicting heart disease, with a particular focus on how different algorithms perform and how their predictions can be understood and trusted by healthcare professionals. <br />

The analysis revealed that the neural network model, after tuning its hyperparameters, achieved a test accuracy of 86.89%, which is comparable to traditional machine learning models like Logistic Regression and SVM. The neural network showed slight improvements in handling the complexities of the dataset, particularly in terms of precision and recall. However, the trade-offs between model interpretability and accuracy were evident. For instance, while the neural network performed well, its "black-box" nature could limit its applicability in clinical settings where transparency is critical. <br />

These findings underscore the importance of model selection in healthcare applications, where both accuracy and interpretability are paramount. While neural networks offer powerful predictive capabilities, their complexity and lack of transparency may not always be suitable for real-world clinical environments. This project highlights the need for a balanced approach, where simpler models may be favored in scenarios requiring clear justifications for predictions. <br />

The neural network portion of the project is divided into the following key sections: <br />

(1) Data Exploration and Preprocessing: Initial analysis of the dataset to understand the distribution of features and the relationships between them, followed by data normalization and splitting into training and test sets. <br />
(2) Model Development: Implementation of a neural network using PyTorch, with detailed exploration of different architectures and hyperparameter tuning via GridSearchCV. <br />
(3) Model Evaluation: Comprehensive evaluation of the model's performance using metrics such as accuracy, precision, recall, and ROC-AUC, supported by visual tools like confusion matrices, learning curves, and calibration curves. <br />
(4) Comparison with Other Models: A comparison of the neural network's performance against traditional machine learning models implemented using scikit-learn. <br />
(5) Conclusions: A discussion on the implications of the findings, particularly the trade-offs between model accuracy and interpretability, and recommendations for future work in this area. <br />

This project extensively compares a variety of traditional machine learning models for heart disease prediction, including: <br />

(1) Logistic Regression (Full Model): This serves as a baseline model using all available features. <br />
(2) Logistic Regression (Reduced Model): A streamlined version of the model using a subset of features selected through variance thresholding. <br />
(3) Decision Tree: An interpretable model that provides insight into decision-making processes but is prone to overfitting. <br />
(4) Naive Bayes: A probabilistic model that makes strong independence assumptions between features. <br />
(5) Linear Discriminant Analysis (LDA): A model that assumes normal distribution of features and equal covariance across classes. <br />
(6) Quadratic Discriminant Analysis (QDA): An extension of LDA allowing for different covariance structures per class. <br />
(7) Support Vector Machine (SVM): A robust classifier particularly effective in high-dimensional spaces. <br />
(8) K Nearest Neighbors (KNN): A simple yet powerful model based on feature similarity. <br />
(9) Kernel Logistic Regression: A more complex model that uses kernel functions to handle non-linear relationships in the data. <br />

The goal of incorporating multiple models was to explore the trade-offs between simplicity, interpretability, and performance. By evaluating both linear models like Logistic Regression and more complex models like SVM and Kernel Logistic Regression, this project aimed to uncover the strengths and weaknesses of each approach when applied to a clinical dataset. This comprehensive comparison ensures that the final recommendations are well-rounded, considering not only accuracy but also the interpretability and computational efficiency of each method. <br />

Some of the key findings include: <br />

• Logistic Regression (Full and Reduced): These models provided strong performance with high accuracy and AUC scores, making them reliable and interpretable options. The reduced model slightly decreased in performance but was simpler and more efficient. <br />
• Decision Tree: While interpretable, the Decision Tree model struggled with overfitting, leading to lower overall performance. <br />
• Naive Bayes and LDA: Both models performed well, with LDA slightly outperforming Naive Bayes in terms of precision and recall, highlighting LDA's effectiveness in handling the dataset's linear separability. <br />
• QDA: This model offered a good balance between complexity and performance, handling non-linear relationships better than LDA but at the cost of slightly reduced interpretability. <br />
• SVM: This model matched the performance of Logistic Regression, with high accuracy and robustness, especially in high-dimensional spaces. <br />
• KNN: This model performed less well, highlighting its sensitivity to the choice of neighbors and its limitations in higher dimensions. <br />
• Kernel Logistic Regression: While it showed promise, this model was computationally intensive and did not outperform simpler models, making it less practical for this dataset. <br />

These findings illustrate that while more complex models like Kernel Logistic Regression and SVM can offer high performance, simpler models like Logistic Regression or LDA often provide similar results with greater interpretability and lower computational cost. In a clinical setting, where interpretability is crucial, this suggests that traditional methods like Logistic Regression may be preferable unless the additional complexity can be justified by significantly improved performance. <br />

This section of the project is structured as follows: <br />

(1) Data Preprocessing and Feature Selection: Variance thresholding is applied to create a reduced feature set, followed by scaling. <br />
(2) Model Training and Evaluation: Each model is trained on the full and reduced datasets, and performance metrics are calculated. <br />
(3) Model Comparison: A comprehensive comparison of model metrics including accuracy, precision, recall, F1-score, and AUC is presented through visualizations. <br />
(4) Conclusions and Recommendations: Based on the comparative analysis, recommendations for the most suitable models in different scenarios are provided. <br />

The dataset is available in the CSV file Heart_Disease.csv or at https://www.kaggle.com/datasets/krishujeniya/heart-diseae/data





