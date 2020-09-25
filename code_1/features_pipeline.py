import pickle
from features_creation import *
from sklearn.externals import joblib
import json

if __name__ == "__main__":

    df_train = pd.read_csv("resources/df_train.csv", index_col=0)

    target_name = "ContrabandIndicator"

    #drop duplicates
    is_duplicated = df_train.duplicated()
    df_train = df_train[is_duplicated==False]

    #divide dataset in features(X) and target(y)
    X_train = df_train.drop(columns=[target_name])
    y_train = df_train[target_name]

    #target_encoding
    target = Target()
    y_train = target.target_encoding(y_train)



    features_creation = Pipeline(
        [('clean_st', CleanText("StatuteReason")),
         ('clean_irc', CleanText("InterventionReasonCode")),
         ('clean_iln', CleanText("InterventionLocationName")),
         ('set_others_statute_reason', SetOthers(feature_name="StatuteReason", threshold=100)),
         ('set_others_InterventionReasonCode', SetOthers(feature_name="InterventionReasonCode", threshold=100)),
         ('set_others_InterventionLocationName', SetOthers(feature_name="InterventionLocationName", threshold=100)),
         ('time_features', TimeFeatures(feature_name="InterventionDateTime", month=False, weekday=False, hour=True)),
         ('per_iln', PercentageEncoder(feature_name="InterventionLocationName")),
         ('per_sr', PercentageEncoder(feature_name="StatuteReason")),
         ('boo_ri', BooleanEncoder(feature_name="ResidentIndicator")),
         ('boo_tri', BooleanEncoder(feature_name="TownResidentIndicator")),
         ('encode_sac', EncodeSAC(feature_name="SearchAuthorizationCode")),
         ('per_irc', PercentageEncoder(feature_name="InterventionReasonCode")),
         ('drop_columns', DropColumns(feature_names=[
                                                     "ReportingOfficerIdentificationID",
                                                     "Department Name",
                                                     "VehicleSearchedIndicator",
                                                     "SubjectAge",
                                                     "SubjectEthnicityCode",
                                                     "SubjectRaceCode",
                                                     "SubjectSexCode"
                                                    ]))
         ])


    features_creation.fit(X_train, y_train)

    with open('resources/columns.json', 'w') as fh:
        json.dump(X_train.columns.tolist(), fh)

    with open('resources/dtypes.pickle', 'wb') as fh:
        pickle.dump(X_train.dtypes, fh)

    joblib.dump(features_creation, 'resources/pipeline_features.pickle')



