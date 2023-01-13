from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier

import preprocess as pp
import hydra


def pipeline(config):
    match_pipe = Pipeline(

        [
            ('date_extraction', pp.DateExtraction(date_variables='trans_date_trans_time')),

            ('drop_fatures',
             pp.DropUnecessaryFeatures(variables_to_drop=config.variables.drop_features)),

            ('categorical_encoder_onehot',
             pp.CategoricalEncoder(variables=config.variables.categorical_onehot)),

            ('categorical_encoder_ordinal', ColumnTransformer(transformers=[
                ("oe", OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1),
                 list(config.variables.categorical_ordinal))],
                remainder="passthrough"))
            ,
            # ('feature hashing',
            #   FeatureHasher(n_features=10, input_type='string')),

            # ('log_transformer',
            #    pp.LogTransformer()),

            ('scaler', MinMaxScaler()),

            ('classifier', DecisionTreeClassifier(max_depth=config.pipeline.decisiontree.parameters.max_depth,
                                                  criterion=config.pipeline.decisiontree.parameters.criterion))
        ]
    )

    return match_pipe
