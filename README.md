# Machine Learning for Heart Disease Risk Prediction

**Introduction** <br />
This project investigates the prediction of heart disease using a dataset from Kaggle by comparing the performance of various machine learning models, including a custom neural network implemented in PyTorch. The primary goal was to understand which model architectures and hyperparameters yield the best performance and to provide insights into the strengths and limitations of each approach in the context of healthcare. <br />

**Purpose** <br />
Heart disease is a leading cause of mortality worldwide, and early detection is crucial for improving patient outcomes. With the growing availability of patient data, there is an opportunity to leverage machine learning and neural networks to develop predictive models that can assist clinicians in making more informed decisions. This project was undertaken to explore the potential of these models in accurately predicting heart disease, with a particular focus on how different algorithms perform and how their predictions can be understood and trusted by healthcare professionals. <br />

**Significance** <br />
The analysis revealed that the neural network model, after tuning its hyperparameters, achieved a test accuracy of 86.89%, which is comparable to traditional machine learning models like Logistic Regression and SVM. The neural network showed slight improvements in handling the complexities of the dataset, particularly in terms of precision and recall. However, the trade-offs between model interpretability and accuracy were evident. For instance, while the neural network performed well, its "black-box" nature could limit its applicability in clinical settings where transparency is critical. <br />

These findings underscore the importance of model selection in healthcare applications, where both accuracy and interpretability are paramount. While neural networks offer powerful predictive capabilities, their complexity and lack of transparency may not always be suitable for real-world clinical environments. This project highlights the need for a balanced approach, where simpler models may be favored in scenarios requiring clear justifications for predictions. <br />

**Project Overview** <br />
**Data Exploration and Preprocessing:** <br />
Initial analysis of the dataset was conducted to understand the distribution of features and the relationships between them. The data was then normalized and split into training and test sets. <br />

**Model Development:** <br />
A neural network was implemented using PyTorch, with detailed exploration of different architectures and hyperparameter tuning via GridSearchCV. <br />

**Model Evaluation:** <br />
Comprehensive evaluation of the model's performance was performed using metrics such as accuracy, precision, recall, and ROC-AUC. Visual tools like confusion matrices, learning curves, and calibration curves were used to support the evaluation. <br />

**Comparison with Other Models:** <br />
The neural network's performance was compared against traditional machine learning models implemented using scikit-learn, including Logistic Regression, Decision Tree, Naive Bayes, LDA, QDA, SVM, KNN, and Kernel Logistic Regression. <br />

**Conclusions:** <br />
The project discusses the implications of the findings, particularly the trade-offs between model accuracy and interpretability, and provides recommendations for future work in this area. <br />

**Final Model** <br />
The best-performing model, after hyperparameter tuning, was a neural network with a test accuracy of 86.89%. Despite its high accuracy, the neural network's lack of interpretability suggests that in clinical settings, more interpretable models like Logistic Regression might be preferable unless the additional accuracy justifies the complexity.

**Findings/Conclusion** <br />
• Logistic Regression: Provided strong performance with high accuracy and AUC scores, making it a reliable and interpretable option. <br />
• Decision Tree: Struggled with overfitting, leading to lower overall performance. <br />
• Naive Bayes and LDA: Both performed well, with LDA slightly outperforming Naive Bayes. <br />
• QDA: Handled non-linear relationships better than LDA but with reduced interpretability. <br />
• SVM: High accuracy and robustness, especially in high-dimensional spaces. <br />
• KNN: Performed less well, highlighting limitations in higher dimensions. <br />
• Kernel Logistic Regression: Computationally intensive and did not outperform simpler models. <br />

These findings suggest that while complex models like SVM and Kernel Logistic Regression offer high performance, simpler models like Logistic Regression or LDA often provide similar results with greater interpretability and lower computational cost. In a clinical setting, where interpretability is crucial, traditional methods like Logistic Regression may be preferable unless significantly improved performance justifies the additional complexity. <br />

The dataset is available in the CSV file Heart_Disease.csv or at https://www.kaggle.com/datasets/krishujeniya/heart-diseae/data





