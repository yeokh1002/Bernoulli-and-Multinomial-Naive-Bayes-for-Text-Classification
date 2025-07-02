# Bernoulli-and-Multinomial-Naive-Bayes-for-Text-Classification

Text classification is a fundamental problem in natural language processing (NLP) with applications in
spam detection, sentiment analysis, and topic categorization. In this project, we focus on classifying
Reddit posts and comments into their respective subreddits: Ottawa, Geneva, Canberra, and Boston.
Given a text sample, our goal is to predict the subreddit it originated from based on learned patterns
in the training data.

To achieve this, we implemented and compared several machine learning models: a Bernoulli
Naive Bayes classifier (implemented from scratch), a Multinomial Naive Bayes classifier (also from
scratch) and additional classifiers from the Scikit-learn library (BernoulliNB, LogisticRegression,
and DecisionTreeClassifier). The dataset consists of labeled Reddit posts and comments, which
were transformed using our custom preprocessing function. We experimented with different feature
subsets, hyperparameters, and vectorization techniques to achieve the best performance, as evaluated
using k-fold cross-validation. The final model was then used to generate predictions on the test set
for submission to a Kaggle competition.
