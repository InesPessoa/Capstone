from imblearn.over_sampling import SMOTE
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix

from features_creation import *

if __name__ == "__main__":
    df_train = pd.read_csv("resources/df_train.csv", index_col=0)
    df_test = pd.read_csv("resources/df_test.csv", index_col=0)
    df_production = pd.read_csv("resources/production_data.csv", index_col=0)

    is_duplicated_train = df_train.duplicated()
    df_train = df_train[is_duplicated_train==False]
    df_train_0 = df_train[df_train["SearchAuthorizationCode"] == "N"]
    df_train_1 = df_train[df_train["SearchAuthorizationCode"] != "N"]

    is_duplicated_test = df_test.duplicated()
    df_test = df_test[is_duplicated_test == False]
    df_test_0 = df_test[df_test["SearchAuthorizationCode"] == "N"]
    df_test_1 = df_test[df_test["SearchAuthorizationCode"] != "N"]

    is_duplicated_prod = df_production.duplicated()
    df_production = df_production[is_duplicated_prod == False]
    df_production = df_production.dropna()
    df_prod_0 = df_production[df_production["SearchAuthorizationCode"] == "N"]
    df_prod_1 = df_production[df_production["SearchAuthorizationCode"] != "N"]


    target_name = "ContrabandIndicator"


    # divide dataset in features(X) and target(y)
    X_train_1 = df_train_1.drop(columns=[target_name])
    y_train_1 = df_train_1[target_name]
    X_train_0 = df_train_0.drop(columns=[target_name])
    y_train_0 = df_train_0[target_name]

    X_test_1 = df_test_1.drop(columns=[target_name])
    y_test_1 = df_test_1[target_name]
    X_test_0 = df_test_0.drop(columns=[target_name])
    y_test_0 = df_test_0[target_name]

    X_prod_1 = df_prod_1.drop(columns=[target_name])
    y_prod_1 = df_prod_1[target_name]
    X_prod_0 = df_prod_0.drop(columns=[target_name])
    y_prod_0 = df_prod_0[target_name]

    # target_encoding
    target = Target()
    y_train_1 = target.target_encoding(y_train_1)
    y_train_0 = target.target_encoding(y_train_0)

    y_test_1 = target.target_encoding(y_test_1)
    y_test_0 = target.target_encoding(y_test_0)

    y_prod_1 = target.target_encoding(y_prod_1)
    y_prod_0 = target.target_encoding(y_prod_0)

    features_cleaning_1 =  Pipeline(
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

    features_cleaning_0 = Pipeline(
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
         # ('encode_sac', EncodeSAC(feature_name="SearchAuthorizationCode")),
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


    X_train_1 = features_cleaning_1.fit_transform(X_train_1, y_train_1)
    X_test_1 = features_cleaning_1.transform(X_test_1)
    X_prod_1 = features_cleaning_1.transform(X_prod_1)

    X_train_0 = features_cleaning_0.fit_transform(X_train_0, y_train_0)
    X_test_0 = features_cleaning_0.transform(X_test_0)
    X_prod_0 = features_cleaning_0.transform(X_prod_0)

    smote_1 = SMOTE()
    X_train_1, y_train_1 = smote_1.fit_resample(X_train_1, y_train_1)
    smote_0 = SMOTE()
    X_train_0, y_train_0 = smote_0.fit_resample(X_train_0, y_train_0)

    adc_1 = AdaBoostClassifier(random_state=0)
    adc_0 = AdaBoostClassifier(random_state=0)

    adc_1.fit(X_train_1, y_train_1)
    y_pred_1 = adc_1.predict(X_test_1)
    y_prod_pred_1 = adc_1.predict(X_prod_1)

    adc_0.fit(X_train_0, y_train_0)
    y_pred_0 = adc_0.predict(X_test_0)
    y_prod_pred_0 = adc_0.predict(X_prod_0)

    y_pred = list(y_pred_0) + list(y_pred_1)
    y_test = list(y_test_0) + list(y_test_1)

    y_prod = list(y_prod_0) + list(y_prod_1)
    y_prod_pred = list(y_prod_pred_0) + list(y_prod_pred_1)

   # y_pred_production = adc.predict(X_production)


    #Model Performance
    score = classification_report(y_test, y_pred, labels=[0, 1])
    print(score)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    cm = "tn: " + str(tn) + "fp: " + str(fp) + "fn: " + str(fn) + "tp: " + str(tp)
    print(cm)

    # Model Performance
    score = classification_report(y_test_0, y_pred_0, labels=[0, 1])
    print(score)
    tn, fp, fn, tp = confusion_matrix(y_test_0, y_pred_0).ravel()
    cm = "tn: " + str(tn) + "fp: " + str(fp) + "fn: " + str(fn) + "tp: " + str(tp)
    print(cm)

    # Model Performance
    score = classification_report(y_test_1, y_pred_1, labels=[0, 1])
    print(score)
    tn, fp, fn, tp = confusion_matrix(y_test_1, y_pred_1).ravel()
    cm = "tn: " + str(tn) + "fp: " + str(fp) + "fn: " + str(fn) + "tp: " + str(tp)
    print(cm)

    score = classification_report(y_prod, y_prod_pred, labels=[0, 1])
    print(score)
    tn, fp, fn, tp = confusion_matrix(y_prod, y_prod_pred).ravel()
    cm = "tn: " + str(tn) + "fp: " + str(fp) + "fn: " + str(fn) + "tp: " + str(tp)
    print(cm)

    # df_test_with_prediction_production = copy.copy(df_production)
    # df_test_with_prediction_production["prediction"] = y_pred_production
    #
    # #df_test_with_prediction.to_csv("resources/df_test_with_prediction_without_sac_production_data.csv")
    #
    # df_test_with_prediction_production.to_csv("resources/df_test_with_prediction_without_sac_production_data.csv")







