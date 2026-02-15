a. Problem Statement:
The goal is to understand and model the factors that influence student academic performance and predict student exam outcomes 
using machine learning models based on various features like hours studied, attendance, parental involvement, access to resources, sleep hours and previous scores etc.

A student is classified as:
Pass (target = 1) if exam score ≥ 35
Fail (target = 0) if exam score < 35
==================================================================================================
b. Dataset Description:
Dataset Name: StudentPerformanceFactors
Source: Kaggle
Type: Binary Classification
Number of instances: >500
Number of features: >12

Target column: target
The dataset contains student-related features such as:
| Feature                        | Description                                    |                    
| ------------------------------ | ---------------------------------------------- | 
| Hours_Studied                  | Number of hours a student studied              |                    
| Attendance                     | Percentage of classes attended                 |                    
| Parental_Involvement           | Level of parental support (Low/Medium/High)    |                    
| Access_to_Resources            | Access to study materials or tech              |                    
| Extracurricular_Activities     | Whether the student participates or not        |                    
| Sleep_Hours                    | Average daily sleep hours                      |                    
| Previous_Scores                | Past academic performance                      |                    
| Motivation_Level               | Self‑reported level of motivation              |                    
| Internet_Access                | Yes/No – access to internet                    |                    
| Tutoring_Sessions              | Number of tutoring sessions attended           |                    
| Family_Income                  | Income group of student’s family               |                    
| Teacher_Quality                | Perceived teacher quality                      |                    
| School_Type                    | Type of school attended                        |                    
| Peer_Influence                 | Influence level of peers                       |                    
| Physical_Activity              | Hours of physical activities                   |                    
| Learning_Disabilities          | Presence of learning challenges                |                    
| Parental_Education_Level       | Education level of parents                     |                    
| Distance_from_Home             | Distance from home to school                   |                    
| Gender                         | Student’s gender                               |                    
| Exam_Score                     | Final exam result (target variable)            |

===========================================================================================================
c. Models Used:
The following Machine Learning classification models were implemented:
1.Logistic Regression
2.Decision Tree Classifier
3.K-Nearest Neighbor Classifier
4.Naive Bayes Classifier - Gaussian or Multinomial
5.Ensemble Model - Random Forest
6.Ensemble Model - XGBoost

Evaluation Metrics Used:
The following evaluation metrics were used:
1.Accuracy
2.Precision
3.Recall
4.F1 Score
5.Matthews Correlation Coefficient (MCC)
6.AUC Score

Model Performance Comparison Table:

| ML Model Name       | Accuracy | Precision | Recall | F1 Score|   MCC  |   AUC  |
| ------------------- | -------- | --------- | ------ | ------- | ------ | ------ |
| Logistic Regression | 0.9932   |  0.9932   | 1.0000 | 0.9966  | 0.4249 | 0.9757 |
| Decision Tree       | 0.9856   |  0.9939   | 0.9916 | 0.9927  | 0.2346 | 0.6322 |
| K-Nearest Neighbor  | 0.9932   |  0.9939   | 0.9992 | 0.9966  | 0.4498 | 0.9041 |
| Naive Bayes         | 0.9932   |  0.9954   | 0.9977 | 0.9966  | 0.5297 | 0.9940 |
| Random Forest       | 0.9924   |  0.9924   | 1.0000 | 0.9962  | 0.3004 | 0.9723 |
| XGBoost             | 0.9955   |  0.9970   | 0.9985 | 0.9977  | 0.7013 | 0.9969 |

Model Performance Observations:

| ML Model Name            | Observation about model performance                                                                                                                                                                                                           |
| ------------------------ |-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Logistic Regression      | Logistic Regression achieved the high accuracy (0.9932) and perfect recall (1.0). Good F1 and AUC. MCC moderate (0.4249), so correlation between predictions and true labels is okay but not the best.                                        |
| Decision Tree            | Decision Tree achieved lowest accuracy (0.9856) and AUC (0.6322). F1 slightly lower. MCC very low (0.2346). Likely overfits and struggles to generalize.                                                                                      |
| K-Nearest Neighbor       | KNN achieved high accuracy (0.9932) and F1 (0.9966). Recall nearly perfect (0.9992). MCC moderate (0.4498). Overall good, but not the strongest across all metrics.                                                                           |
| Naive Bayes              | Naive Bayes achieved high accuracy (0.9932), precision (0.9954), recall (0.9977), and F1 (0.9966). MCC strong (0.5297) and AUC high (0.9940). Very balanced and robust model.                                                                 |
| Random Forest (Ensemble) | Random Forest achieved accuracy very high (0.9924) with perfect recall (1.0). F1 high (0.9962). MCC moderate (0.3004) and AUC good (0.9723). Performs well, but correlation between predictions and labels not very strong.                   |
| XGBoost (Ensemble)       | XGBoost showed the best overall performance among all models. It achieved highest accuracy (0.9955), precision (0.9970), recall (0.9985), F1 (0.9977), MCC (0.7013), and AUC (0.9969). Excels across all metrics; very reliable and balanced. |

