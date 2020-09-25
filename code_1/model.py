from features_creation import *
import joblib
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import AdaBoostClassifier


if __name__ == "__main__":
    df_train = pd.read_csv("resources/df_train.csv", index_col=0)
    df_test = pd.read_csv("resources/df_test.csv", index_col=0)

    target_name = "ContrabandIndicator"

    # drop duplicates
    is_duplicated = df_train.duplicated()
    df_train = df_train[is_duplicated == False]

    # divide dataset in features(X) and target(y)
    X_train = df_train.drop(columns=[target_name])
    y_train = df_train[target_name]

    X_test = df_test.drop(columns=[target_name])
    y_test = df_test[target_name]

    # target_encoding
    target = Target()
    y_train = target.target_encoding(y_train)

    y_test = target.target_encoding(y_test)

    features_cleaning = joblib.load("resources/pipeline_features.pickle")

    X_train = features_cleaning.fit_transform(X_train, y_train)

    X_test = features_cleaning.transform(X_test)

    smote = SMOTE()

    X_train, y_train = smote.fit_resample(X_train, y_train)

    adc = AdaBoostClassifier(random_state=0)

    adc.fit(X_train, y_train)
    y_pred = adc.predict(X_test)
    y_perc = adc.predict_proba(X_test)

    #save model
    joblib.dump(adc, 'resources/model.pickle')

    #Model Performance
    score = classification_report(y_test, y_pred, labels=[0, 1])
    print(score)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    cm = "tn: " + str(tn) + "fp: " + str(fp) + "fn: " + str(fn) + "tp: " + str(tp)
    print(cm)

    file = open("resources/final_model_results.txt", 'w')
    file.write(str(score))
    file.write("\n")
    file.write(cm)
    file.close()

    df_test_with_prediction = copy.copy(df_test)
    df_test_with_prediction["prediction"] = y_pred
    df_test_with_prediction["probability"] = y_perc[:, 1]

    df_test_with_prediction.to_csv("resources/df_test_with_prediction.csv")







