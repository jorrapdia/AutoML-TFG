import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import StandardScaler
from tpot.builtins import StackingEstimator, ZeroCount
from xgboost import XGBClassifier
from sklearn.preprocessing import FunctionTransformer
from copy import copy

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: 0.8561798797964284
exported_pipeline = make_pipeline(
    make_union(
        make_pipeline(
            make_union(
                FunctionTransformer(copy),
                StackingEstimator(estimator=XGBClassifier(learning_rate=0.5, max_depth=1, min_child_weight=12, n_estimators=100, n_jobs=1, subsample=0.15000000000000002, verbosity=0))
            ),
            ZeroCount()
        ),
        make_union(
            make_union(
                StackingEstimator(estimator=XGBClassifier(learning_rate=0.5, max_depth=5, min_child_weight=12, n_estimators=100, n_jobs=1, subsample=0.15000000000000002, verbosity=0)),
                make_union(
                    FunctionTransformer(copy),
                    FunctionTransformer(copy)
                )
            ),
            FunctionTransformer(copy)
        )
    ),
    StandardScaler(),
    StackingEstimator(estimator=SGDClassifier(alpha=0.001, eta0=0.1, fit_intercept=False, l1_ratio=0.0, learning_rate="constant", loss="hinge", penalty="elasticnet", power_t=0.0)),
    SelectPercentile(score_func=f_classif, percentile=65),
    XGBClassifier(learning_rate=0.1, max_depth=9, min_child_weight=19, n_estimators=100, n_jobs=1, subsample=0.35000000000000003, verbosity=0)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
