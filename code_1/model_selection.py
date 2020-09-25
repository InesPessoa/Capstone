from imblearn.ensemble import RUSBoostClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import cross_val_score
from features_creation import *
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
import json
from statistics import mean

if __name__ == "__main__":
    df_train = pd.read_csv("resources/df_train.csv", index_col=0)

    target_name = "ContrabandIndicator"

    # drop duplicates
    is_duplicated = df_train.duplicated()
    df_train = df_train[is_duplicated == False]

    # divide dataset in features(X) and target(y)
    X_train = df_train.drop(columns=[target_name])
    y_train = df_train[target_name]

    # target_encoding
    target = Target()
    y_train = target.target_encoding(y_train)

    pkl_filename = "resources/pipeline_features.pickle"

    features_cleaning = joblib.load(pkl_filename)

    X_train = features_cleaning.fit_transform(X_train, y_train)

    #note: pipeline feature_cleaning in an ideal way should also be trainned inside cross validation

    ##########################################RANDOM FOREST#########################################

    models_results = dict()

    print("Random Forest with undersampling")

    rfc = RandomForestClassifier(random_state=0, n_jobs=-1)
    rus = RandomUnderSampler()

    model_rfc = ImbPipeline([
        ('oversampling', rus),
        ('classification', rfc)
    ])
    score = mean(cross_val_score(model_rfc, X_train, y_train, cv=5, scoring='f1_macro'))

    print("score:", score)

    models_results["Random Forest with undersampling"] = copy.copy(score)

    print("Random Forest with oversampling")

    rfc = RandomForestClassifier(random_state=0, n_jobs=-1)
    ros = RandomOverSampler()

    model_rfc = ImbPipeline([
        ('oversampling', ros),
        ('classification', rfc)
    ])

    score = mean(cross_val_score(model_rfc, X_train, y_train, cv=5, scoring='f1_macro'))

    print("score:", score)

    models_results["Random Forest with oversampling"] = copy.copy(score)

    print("Random Forest with balanced_subsample")

    rfc = RandomForestClassifier(class_weight="balanced_subsample", random_state=0, n_jobs=-1)

    score = mean(cross_val_score(rfc, X_train, y_train, cv=5, scoring='f1_macro'))

    print("score:", score)

    models_results["Random Forest with balanced_subsample"] = copy.copy(score)

    print("Random Forest with balanced")

    rfc = RandomForestClassifier(class_weight="balanced", random_state=0, n_jobs=-1)

    score = mean(cross_val_score(rfc, X_train, y_train, cv=5, scoring='f1_macro'))

    print("score:", score)

    models_results["Random Forest with balanced"] = copy.copy(score)

    print("Random Forest with smote")

    rfc = RandomForestClassifier(random_state=0, n_jobs=-1)
    smote = SMOTE()

    model_rfc = ImbPipeline([
        ('smote', smote),
        ('classification', rfc)
    ])

    score = mean(cross_val_score(model_rfc, X_train, y_train, cv=5, scoring='f1_macro'))

    print("score:", score)

    models_results["Random Forest with smote"] = copy.copy(score)

    #############################################AdaBoost#######################################

    print("AdaBoost with undersampling")

    adc = AdaBoostClassifier(random_state=0)
    rus = RandomUnderSampler()

    model_adc = ImbPipeline([
        ('undersampling', rus),
        ('classification', adc)
    ])
    score = mean(cross_val_score(model_adc, X_train, y_train, cv=5, scoring='f1_macro', n_jobs=-1))

    print("score:", score)

    models_results["AdaBoost with undersampling"] = copy.copy(score)

    print("AdaBoost with oversampling")

    adc = AdaBoostClassifier(random_state=0)
    ros = RandomOverSampler()

    model_adc = ImbPipeline([
        ('oversampling', ros),
        ('classification', adc)
    ])
    score = mean(cross_val_score(model_adc, X_train, y_train, cv=5, scoring='f1_macro', n_jobs=-1))

    print("score:", score)

    models_results["AdaBoost with oversampling"] = copy.copy(score)

    print("AdaBoost with smote")

    adc = AdaBoostClassifier(random_state=0)
    smote = SMOTE()

    model_adc = ImbPipeline([
        ('smote', smote),
        ('classification', adc)
    ])
    score = mean(cross_val_score(model_adc, X_train, y_train, cv=5, scoring='f1_macro', n_jobs=-1))

    print("score:", score)

    models_results["AdaBoost with smote"] = copy.copy(score)

    #############################################GradientBoosting#######################################
    print("GradientBoosting with undersampling")

    grad = GradientBoostingClassifier(random_state=0)
    rus = RandomUnderSampler()

    model_adc = ImbPipeline([
        ('undersampling', rus),
        ('classification', grad)
    ])
    score = mean(cross_val_score(model_adc, X_train, y_train, cv=5, scoring='f1_macro', n_jobs=-1))

    print("score:", score)

    models_results["GradientBoosting with undersampling"] = copy.copy(score)

    print("GradientBoosting with oversampling")

    grad = GradientBoostingClassifier(random_state=0)
    ros = RandomOverSampler()

    model_adc = ImbPipeline([
        ('oversampling', ros),
        ('classification', grad)
    ])
    score = mean(cross_val_score(model_adc, X_train, y_train, cv=5, scoring='f1_macro', n_jobs=-1))

    print("score:", score)

    models_results["GradientBoosting with oversampling"] = copy.copy(score)

    print("GradientBoosting with smote")

    grad = GradientBoostingClassifier(random_state=0)
    smote = SMOTE()

    model_adc = ImbPipeline([
        ('smote', smote),
        ('classification', grad)
    ])
    score = mean(cross_val_score(model_adc, X_train, y_train, cv=5, scoring='f1_macro', n_jobs=-1))

    print("score:", score)

    models_results["GradientBoosting with smote"] = copy.copy(score)


    with open('resources/selection_results.json', 'w') as fh:
        json.dump(models_results, fh)