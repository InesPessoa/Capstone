import pandas as pd
import copy
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

class CleanText(BaseEstimator, TransformerMixin):

    def __init__(self, feature_name):
        self.feature_name = feature_name
        self.values = None

    def clean_text(self, value):
        value_ = copy.copy(value).lower()
        value_ = value_.replace(" ", "")

        value_ = value_.replace("\n", "")
        value_ = value_.replace("\t", "")

        if self.feature_name == "InterventionLocationName":
            if ("at" in value) and (value_.replace("at", "@").split("@")[0] in self.values):
                value_ = value_.replace("at", "@")
            if "@" in value_:
                value_ = value_.split("@")[0]
            if "/" in value_:
                value_ = value_.split("/")[0]
            value_ = re.sub("[\W]", "", value_)
            value_ = self.clean_intervention_location_name(value_)
            return value_
        elif self.feature_name == "Department Name":
            value_ = value_.replace("@", "at")
            value_ = re.sub("[\W]", "", value_)
            value_ = self.clean_department_name(value_)
            return value_
        else:
            value_ = re.sub("[\W]", "", value_)
            return value_

    def clean_intervention_location_name(self, value):
        value_ = copy.copy(value)
        if value_ in ["", "none", "na", "nan"]:
            return None
        elif "unknown" in value_:
            return None
        elif value_ in ["othertown", "ve", "venue", "yourcity", "st", "street", "dr", "drive", "ville"]:
            return "others"
        elif value_[-6:] == "street":
            return value_[:-6]
        elif value_[-2:] == "st":
            return value_[:-2]
        elif value_[-5:] == "drive":
            return value_[:-5]
        elif value_[-2:] == "dr":
            return value_[:-2]
        elif value_[-5:] == "ville":
            return value_[:-5]
        elif value_[-5:] == "venue":
            return value_[:-5]
        elif value_[-2:] == "ve":
            return value_[:-2]
        else:
            return value_

    def clean_department_name(self, value):
        value_ = copy.copy(value)
        if "police" in value_:
            return value_.replace("police", "")
        else:
            return value_

    def split_text(self, value):
        value_ = copy.copy(value)
        value_ = value_.lower().replace(" ", "")
        if "@" in value_:
            value_ = value_.split("@")[0]
        if "/" in value_:
            value_ = value_.split("/")[0]
        return value_


    def fit(self, X, y=None):
        X_ = copy.copy(X)
        self.values = X_[self.feature_name].astype("str").apply(self.split_text).unique()
        return self

    def transform(self, X):
        X_ = copy.copy(X)
        X_[self.feature_name] = X_[self.feature_name].astype("str")
        X_[self.feature_name] = X_[self.feature_name].apply(lambda x: self.clean_text(x))
        return X_


class SetOthers(BaseEstimator, TransformerMixin):

    def __init__(self, feature_name, threshold):
        self.feature_name = feature_name
        self.threshold = threshold
        self.values = None

    def get_values(self, X):
        number_ocurrencies = X[self.feature_name].value_counts()
        values = number_ocurrencies[number_ocurrencies > self.threshold].index
        return values

    def clean_value(self, value):
        if value in self.values:
            return value
        else:
            return "other"

    def fit(self, X, y=None):
        self.values = self.get_values(X)
        return self

    def transform(self, X):
        X_ = copy.copy(X)
        X_[self.feature_name] = X_[self.feature_name].astype("str")
        X_[self.feature_name] = X_[self.feature_name].apply(lambda x: self.clean_value(x))
        return X_

class TimeFeatures(BaseEstimator, TransformerMixin):

    def __init__(self, feature_name, time_format="%m/%d/%Y %H:%M:%S", month=True, weekday=True, hour=True):
        self.feature_name = feature_name
        self.time_format = time_format
        if month == True:
            self.month = True
        else:
            self.month = False
        if weekday == True:
            self.weekday = True
        else:
            self.weekday = False
        if hour == True:
            self.hour = True
        else:
            self.hour = False

    def clean_timestamp(self, value):
        if value[-2:] == "AM" and value[11:13] == "12":
            return value[:11] + "00" + value[13:-3]

        # remove the AM
        elif value[-2:] == "AM":
            return value[:-3]

        # Checking if last two elements of time
        # is PM and first two elements are 12
        elif value[-2:] == "PM" and value[11:13] == "12":
            return value[:-3]

        else:
            # add 12 to hours and remove PM
            return value[:11] + str(int(value[11:13]) + 12) + value[13:-3]

    def create_features(self, X):
        X_ = copy.copy(X)

        if self.weekday == True:
            X_["weekday"] = X_[self.feature_name].dt.weekday
        if self.hour == True:
            X_["hour"] = X_[self.feature_name].dt.hour
        if self.month == True:
            X_["month"] = X_[self.feature_name].dt.month

        return X_

    def create_cyclical_feature(self, X, name, period):
        X_ = copy.copy(X)
        X_['sin_' + name] = np.sin(2 * np.pi * X_[name] / period)
        X_['cos_' + name] = np.cos(2 * np.pi * X_[name] / period)
        X_ = X_.drop(columns=[name])
        return X_

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_ = copy.copy(X)
        X_[self.feature_name] = X_[self.feature_name].apply(self.clean_timestamp)
        X_[self.feature_name] = X_[self.feature_name].apply(lambda x: pd.to_datetime(x, format=self.time_format))
        X_ = self.create_features(X_)
        if self.month == True:
            X_ = self.create_cyclical_feature(X_, "month", 12)
        if self.weekday == True:
            X_ = self.create_cyclical_feature(X_, "weekday", 7)
        if self.hour == True:
            X_ = self.create_cyclical_feature(X_, "hour", 24)
        X_ = X_.drop(columns=[self.feature_name])
        return X_


class DropColumns(BaseEstimator, TransformerMixin):

    def __init__(self, feature_names):
        self.feature_names = feature_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_ = copy.copy(X)
        for i in self.feature_names:
            if i in X.columns:
                X_ = X_.drop(columns=i)
        return X_


class BooleanEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, feature_name):
        self.feature_name = feature_name
        self.le = LabelEncoder()

    def invalid_data(self, value):
        if value:
            return True
        else:
            return False

    def fit(self, X, y=None):
        X_ = copy.copy(X)
        X_[self.feature_name] = X_[self.feature_name].apply(self.invalid_data)
        self.le.fit(X[self.feature_name].astype("str"))
        return self

    def transform(self, X):
        X_ = copy.copy(X)
        X_[self.feature_name] = X_[self.feature_name].apply(self.invalid_data)
        X_[self.feature_name] = self.le.transform(X_[self.feature_name].astype("str"))
        return X_

class PercentageEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, feature_name):
        self.feature_name = feature_name
        self.percentages = None

    def percentage_of_positives(self, group):
        true_positives = group[(group.target == True)].shape[0]
        n_vehicles_stopped = group.shape[0]
        return round(true_positives/n_vehicles_stopped, 2)

    def fit(self, X, y=None):
        df_ = copy.copy(X)
        df_["target"] = y
        df_[self.feature_name] = df_[self.feature_name].astype("str")
        results_serie = df_.groupby(by=self.feature_name).apply(self.percentage_of_positives)
        self.percentages = results_serie.to_dict()
        return self

    def transform(self, X):
        X_ = copy.copy(X)
        X_[self.feature_name] = X_[self.feature_name].astype("str").map(self.percentages)
        return X_


class EncodeSAC(BaseEstimator, TransformerMixin):

    def __init__(self, feature_name):
        self.feature_name = feature_name

    def encode_value(self, value):
        if value in ["O", "C", "I"]:
            return 1
        else:
            return 0

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_ = copy.copy(X)
        X_[self.feature_name] = X_[self.feature_name].astype("str").apply(self.encode_value)
        return X_

class CleanAge(BaseEstimator, TransformerMixin):

    def __init__(self, feature_name):
        self.feature_name = feature_name

    def fit(self, X, y=None):
        return self

    def clean_age(self, value):
        if value < 18:
            return "Youth"
        elif value >= 18 and value <=35:
            return "Young Adult"
        elif value  >= 36 and value <=55:
            return "Adult"
        else:
            return "Senior"

    def transform(self, X):
        X_ = copy.copy(X)
        X_[self.feature_name] = X_[self.feature_name].apply(self.clean_age)
        return X_

class EncodeSACRegAmplitude(BaseEstimator, TransformerMixin):

    def __init__(self, feature_name, threshold, power):
        self.feature_name = feature_name
        self.amplitude_by_dn = None
        self.threshold = threshold
        self.power = power

    def calculate_precision(self, group, threshold=0):
        true_positives = group[(group.VehicleSearchedIndicator == True) & (group.ContrabandIndicator == True)].shape[0]
        n_vehicle_searched = group[(group.VehicleSearchedIndicator == True)].shape[0]
        if n_vehicle_searched > threshold:
            precision = true_positives/n_vehicle_searched
            return precision
        else:
            return 0

    def one_minus_amplitude(sef, row):
        row_ = copy.copy(row)
        n_nulls = len(row[pd.isna(row)])
        len_row = len(row)
        n_not_nulls = len_row - n_nulls
        row_ = row_.dropna()
        diffs = []
        if n_not_nulls > 1:
            for i in range(0, len(row_)-1):
                for j in range(i + 1, len(row_)):
                    diffs.append(abs(row_[i] - row_[j]))
            value = 1 - np.max(diffs)
            return value
        else:
            return 0

    def evaluate_bias_by_group(self, df, group, threshold=0):
        police_src = df.groupby(by=group).apply(lambda x: self.calculate_precision(x, self.threshold))
        result = police_src.unstack(level=[1, 2, 3, 4])
        result["amplitude"] = result.apply(self.one_minus_amplitude, axis=1)
        self.amplitude_by_dn = result["amplitude"]


    def encode_value(self, value):
        if value in ["O", "C", "I"]:
            return 1
        else:
            return 0

    def apply_regularization(self, row):
        return (row[self.feature_name] * self.amplitude_by_dn.loc[row["Department Name"]]) ** self.power

    def fit(self, X, y=None):

        X_ = copy.copy(X)
        X_["ContrabandIndicator"] = y
        self.evaluate_bias_by_group(df=X_, group=["Department Name", "SubjectRaceCode", "SubjectEthnicityCode",
                                                  "SubjectAge", "SubjectSexCode"], threshold=self.threshold)
        return self

    def transform(self, X):
        X_ = copy.copy(X)
        X_[self.feature_name] = X_[self.feature_name].astype("str").apply(self.encode_value)
        X_[self.feature_name] = X_.apply(self.apply_regularization, axis=1)
        return X_

class EncodeSACRegPrecision(BaseEstimator, TransformerMixin):

    def __init__(self, feature_name, threshold, power):
        self.feature_name = feature_name
        self.precision_by_dn = None
        self.threshold = threshold
        self.power = power

    def calculate_precision(self, group, threshold=0):
        true_positives = group[(group.VehicleSearchedIndicator == True) & (group.ContrabandIndicator == True)].shape[0]
        n_vehicle_searched = group[(group.VehicleSearchedIndicator == True)].shape[0]
        if n_vehicle_searched > threshold:
            precision = true_positives/n_vehicle_searched
            return precision
        else:
            return 0

    def evaluate_bias_by_group(self, df, group):
        self.precision_by_dn = df.groupby(by=group).apply(lambda x: self.calculate_precision(x, self.threshold))

    def encode_value(self, value):
        if value in ["O", "C", "I"]:
            return 1
        else:
            return 0

    def apply_regularization(self, row):
        if (row["Department Name"], row["SubjectRaceCode"], row["SubjectEthnicityCode"], row["SubjectAge"], row["SubjectSexCode"]) in list(self.precision_by_dn.index.values):
            return (row[self.feature_name] * self.precision_by_dn.loc[row["Department Name"],
                                                                      row["SubjectRaceCode"],
                                                                      row["SubjectEthnicityCode"],
                                                                      row["SubjectAge"],
                                                                      row["SubjectSexCode"]
                                                   ]) ** self.power
        else:
            return 0

    def fit(self, X, y=None):

        X_ = copy.copy(X)
        X_["ContrabandIndicator"] = y
        self.evaluate_bias_by_group(df=X_, group=["Department Name", "SubjectRaceCode", "SubjectEthnicityCode",
                                                  "SubjectAge", "SubjectSexCode"])
        return self

    def transform(self, X):
        X_ = copy.copy(X)
        X_[self.feature_name] = X_[self.feature_name].astype("str").apply(self.encode_value)
        X_[self.feature_name] = X_.apply(self.apply_regularization, axis=1)
        return X_

class Target():

    def target_encoding(self, y):
        y_ = copy.copy(y)
        y_ = y_.astype("str").map({"False": 0, "True": 1})
        return y_


if __name__ == "__main__":
    df = pd.read_csv("resources/train.csv")

    target_name = "ContrabandIndicator"

    df_train, df_test = train_test_split(df,
                                         test_size=0.2,
                                         random_state=100,
                                         shuffle=True,
                                         stratify=df[target_name])
    #save train and test csv
    df_train.to_csv("resources/df_train.csv")
    df_test.to_csv("resources/df_test.csv")

    #drop duplicates
    is_duplicated = df.duplicated()
    df = df[is_duplicated == False]

    is_duplicated_train = df_train.duplicated()
    df_train = df_train[is_duplicated_train == False]

    #divide dataset in features(X) and target(y)
    X = df.drop(columns=[target_name])
    y = df[target_name]

    X_train = df_train.drop(columns=[target_name])
    y_train = df_train[target_name]

    #target_encoding
    target = Target()
    y_train = target.target_encoding(y_train)

    features_creation = Pipeline(
        [('clean_text_iln', CleanText(feature_name="InterventionLocationName")),
         ('clean_text_sr', CleanText(feature_name="StatuteReason")),
         ('clean_text_irc', CleanText(feature_name="InterventionReasonCode")),
         ('set_others_InterventionLocationName', SetOthers(feature_name="InterventionLocationName", threshold=100)),
         ('set_others_statute_reason', SetOthers(feature_name="StatuteReason", threshold=100)),
         ('set_others_InterventionReasonCode', SetOthers(feature_name="InterventionReasonCode", threshold=100)),
         ('time_features', TimeFeatures(feature_name="InterventionDateTime", month=True, weekday=True, hour=True)),
         ('per_iln', PercentageEncoder(feature_name="InterventionLocationName")),
         ('per_sr', PercentageEncoder(feature_name="StatuteReason")),
         ('boo_ri', BooleanEncoder(feature_name="ResidentIndicator")),
         ('boo_tri', BooleanEncoder(feature_name="TownResidentIndicator")),
         ('encoder_sac', EncodeSAC(feature_name="SearchAuthorizationCode")),
         ('per_irc', PercentageEncoder(feature_name="InterventionReasonCode")),
         ('drop_columns', DropColumns(feature_names=["VehicleSearchedIndicator",
                                                      "ReportingOfficerIdentificationID",
                                                      "Department Name",
                                                      "SubjectAge",
                                                      "SubjectEthnicityCode",
                                                      "SubjectRaceCode",
                                                      "SubjectSexCode"]))
         ])

    #df_train

    features_creation.fit(X_train, y_train)

    X_train = features_creation.transform(X_train)

    df_train_cleaned = copy.copy(X_train)

    df_train_cleaned[target_name] = y_train

    df_train_cleaned.to_csv("resources/df_train_cleaned.csv")

    #df

    features_creation.fit(X, y)

    X = features_creation.transform(X)

    df_cleaned = copy.copy(X)

    df_cleaned[target_name] = y

    df_cleaned.to_csv("resources/df_cleaned.csv")


