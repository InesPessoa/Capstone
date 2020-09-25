from imblearn.over_sampling import SMOTE
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix

from features_creation import *

if __name__ == "__main__":
    df_train = pd.read_csv("resources/df_train.csv", index_col=0)
    df_test = pd.read_csv("resources/df_test.csv", index_col=0)
    df_production = pd.read_csv("resources/production_data.csv", index_col=0).drop(columns=["prediction"])

    #training the model with just serached data - serached vechicles are suspcicious
    df_train = df_train[df_train["VehicleSearchedIndicator"]==True]
    df_test_f = df_test[df_test["VehicleSearchedIndicator"]==True]

    target_name = "ContrabandIndicator"

    # drop duplicates
    is_duplicated_train = df_train.duplicated()
    df_train = df_train[is_duplicated_train == False]

    is_duplicated_test = df_test.duplicated()
    df_test = df_test[is_duplicated_test == False]

    is_duplicated_test_f = df_test_f.duplicated()
    df_test_f = df_test_f[is_duplicated_test_f == False]


    # divide dataset in features(X) and target(y)
    X_train = df_train.drop(columns=[target_name])
    y_train = df_train[target_name]

    X_test = df_test.drop(columns=[target_name])
    y_test = df_test[target_name]

    X_test_f = df_test_f.drop(columns=[target_name])
    y_test_f = df_test_f[target_name]

    X_production = df_production.drop(columns=[target_name])

    # target_encoding
    target = Target()
    y_train = target.target_encoding(y_train)

    y_test = target.target_encoding(y_test)
    y_test_f = target.target_encoding(y_test_f)

    features_cleaning =  Pipeline(
        [('clean_text_sr', CleanText("StatuteReason")),
         ('clean_text_irc', CleanText("InterventionReasonCode")),
         ('clean_text_iln', CleanText("InterventionLocationName")),
         ('set_others_statute_reason', SetOthers(feature_name="StatuteReason", threshold=10)),
         ('set_others_InterventionReasonCode', SetOthers(feature_name="InterventionReasonCode", threshold=10)),
         ('set_others_InterventionLocationName', SetOthers(feature_name="InterventionLocationName", threshold=10)),
         ('time_features', TimeFeatures(feature_name="InterventionDateTime", month=False, weekday=False, hour=True)),
         ('per_iln', PercentageEncoder(feature_name="InterventionLocationName")),
         ('per_sr', PercentageEncoder(feature_name="StatuteReason")),
         ('boo_ri', BooleanEncoder(feature_name="ResidentIndicator")),
         ('boo_tri', BooleanEncoder(feature_name="TownResidentIndicator")),
         ('encode_sac', EncodeSAC(feature_name="SearchAuthorizationCode")),
         ('per_irc', PercentageEncoder(feature_name="InterventionReasonCode")),
         ('drop_columns', DropColumns(feature_names=[#"SearchAuthorizationCode",
                                                     "ReportingOfficerIdentificationID",
                                                     "Department Name",
                                                     "VehicleSearchedIndicator",
                                                     "SubjectAge",
                                                     "SubjectEthnicityCode",
                                                     "SubjectRaceCode",
                                                     "SubjectSexCode"
                                                    ]))
         ])


    X_train = features_cleaning.fit_transform(X_train, y_train)

    X_test = features_cleaning.transform(X_test)
    X_test_f = features_cleaning.transform(X_test_f)

    df_cleaned = copy.copy(X_test)
    df_cleaned["target"] = y_test
    df_cleaned = df_cleaned.dropna()

    X_test = df_cleaned.drop(columns=["target"])
    y_test = df_cleaned["target"]

    X_production = features_cleaning.transform(X_production)

    smote = SMOTE()

    X_train, y_train = smote.fit_resample(X_train, y_train)

    adc = AdaBoostClassifier(random_state=0)

    adc.fit(X_train, y_train)
    y_pred = adc.predict(X_test)
    y_pred_f = adc.predict(X_test_f)

    y_pred_production = adc.predict(X_production)
    y_pred_production_prob = adc.predict_proba(X_production)


    #Model Performance all data
    score = classification_report(y_test, y_pred, labels=[0, 1])
    print(score)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    cm = "tn: " + str(tn) + "fp: " + str(fp) + "fn: " + str(fn) + "tp: " + str(tp)
    print(cm)

    file = open("resources/final_model_sac_results_sv_all_test.txt", 'w')
    file.write(str(score))
    file.write("\n")
    file.write(cm)
    file.close()

    # Model Performance filtered data
    score = classification_report(y_test_f, y_pred_f, labels=[0, 1])
    print(score)
    tn, fp, fn, tp = confusion_matrix(y_test_f, y_pred_f).ravel()
    cm = "tn: " + str(tn) + "fp: " + str(fp) + "fn: " + str(fn) + "tp: " + str(tp)
    print(cm)

    file = open("resources/final_model_results_sv_filtered_test.txt", 'w')
    file.write(str(score))
    file.write("\n")
    file.write(cm)
    file.close()

    df_test_with_prediction_production = copy.copy(df_production)
    df_test_with_prediction_production["prediction"] = y_pred_production

    df_test_with_prediction_production["probability"] = y_pred_production_prob[:, 1]


    df_test_with_prediction_production.to_csv("resources/df_test_with_prediction_production_data.csv")

    df_test_with_prediction = copy.copy(df_test_f)
    df_test_with_prediction["prediction"] = y_pred_f

    df_test_with_prediction.to_csv("resources/df_test_filtered_with_prediction_with_sac.csv")







