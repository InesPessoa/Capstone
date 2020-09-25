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

    features_cleaning =  Pipeline(
        [('clean_text_sr', CleanText("StatuteReason")),
         ('clean_text_irc', CleanText("InterventionReasonCode")),
         ('clean_text_iln', CleanText("InterventionLocationName")),
         ('set_others_statute_reason', SetOthers(feature_name="StatuteReason", threshold=100)),
         ('set_others_InterventionReasonCode', SetOthers(feature_name="InterventionReasonCode", threshold=100)),
         ('set_others_InterventionLocationName', SetOthers(feature_name="InterventionLocationName", threshold=100)),
         ('time_features', TimeFeatures(feature_name="InterventionDateTime", month=False, weekday=False, hour=True)),
         ('per_iln', PercentageEncoder(feature_name="InterventionLocationName")),
         ('per_sr', PercentageEncoder(feature_name="StatuteReason")),
         ('boo_ri', BooleanEncoder(feature_name="ResidentIndicator")),
         ('boo_tri', BooleanEncoder(feature_name="TownResidentIndicator")),
         #('encode_sac', EncodeSAC(feature_name="SearchAuthorizationCode")),
         ('per_irc', PercentageEncoder(feature_name="InterventionReasonCode")),
         ('drop_columns', DropColumns(feature_names=["SearchAuthorizationCode",
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

    smote = SMOTE()

    X_train, y_train = smote.fit_resample(X_train, y_train)

    adc = AdaBoostClassifier(random_state=0)

    adc.fit(X_train, y_train)
    y_pred = adc.predict(X_test)


    #Model Performance
    score = classification_report(y_test, y_pred, labels=[0, 1])
    print(score)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    cm = "tn: " + str(tn) + "fp: " + str(fp) + "fn: " + str(fn) + "tp: " + str(tp)
    print(cm)

    file = open("resources/final_model_without_sac_results.txt", 'w')
    file.write(str(score))
    file.write("\n")
    file.write(cm)
    file.close()

    df_test_with_prediction = copy.copy(df_test)
    df_test_with_prediction["prediction"] = y_pred

    df_test_with_prediction.to_csv("resources/df_test_with_prediction_without_sac.csv")







