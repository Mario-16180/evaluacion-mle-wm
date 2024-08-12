import wandb
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

def train_model(X_train, X_test, y_train, y_test, gamma, max_depth, scale_pos_weight):
    # Wandb initialization
    wandb.init(project='WalmartChallenge',config={"model": "XGBoost", 
                                                  "gamma": gamma, "max_depth": max_depth, 
                                                  "scale_pos_weight": scale_pos_weight})
    # Model training
    model = XGBClassifier(gamma=gamma, max_depth=max_depth, scale_pos_weight=scale_pos_weight)
    model.fit(X_train, y_train)
    # Model evaluation
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    # Logging the metrics
    wandb.log({"accuracy": accuracy, "f1": f1})
    # Saving the model
    model.save_model('models/model.bin')
    # Wandb finishing
    wandb.finish()
    return model, accuracy, f1