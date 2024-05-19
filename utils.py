import pandas as pd

clf_models = ['lr', 'knn', 'rf', 'dt', 'svm', 'xgboost', 'lightgbm', 'catboost']
impute_strategy = ['mode', 'mean', 'zero', 'min', 'max', 'drop']
encoding_methods = ["Label encoding", "One-hot encoding"]

def average(df, column):
    numeric_data = pd.to_numeric(df[column], errors='coerce')
    return numeric_data.mean()

def get_min(df, column):
    numeric_data = pd.to_numeric(df[column], errors='coerce')
    return numeric_data.min()

def get_max(df, column):
    numeric_data = pd.to_numeric(df[column], errors='coerce')
    return numeric_data.max()

def mode(df, column):
    return df[column].mode().iloc[0]

def one_hot_encode(df, column_name):
    if column_name in df.columns:
        df_encoded = pd.get_dummies(df.loc[:, column_name], prefix=column_name, dtype=int)
        df.drop(column_name, axis=1, inplace=True)
        df = pd.concat([df, df_encoded], axis=1)
        print("One-hot encoding applied to column ", column_name, " successfully.")
        print(df)
        return df
    else:
        print(f"Column '{column_name}' not found in the DataFrame.")

def label_encode(df, column_name):
    if column_name in df.columns:
        categories = df[column_name].unique()
        mapping = {category: index for index, category in enumerate(categories)}
        df[column_name] = df[column_name].map(mapping)
        print(f"Label encoding applied to column '{column_name}' successfully.")
    else:
        print(f"Column '{column_name}' not found in the DataFrame.")

def impute(df, column, strategy='mode'):
    if strategy == 'mean':
        df[column].fillna(average(df, column), inplace=True)
    elif strategy == 'mode':
        df[column].fillna(mode(df, column), inplace=True)
    elif strategy == 'min':
        df[column].fillna(get_min(df, column), inplace=True)
    elif strategy == 'max':
        df[column].fillna(get_max(df, column), inplace=True)
    elif strategy == 'zero':
        df[column].fillna(0, inplace=True)
    else:
        raise Exception("Unsupported impute strategy. Please select one of the following: ", impute_strategy)
    

def read_file(path):
        # Read the uploaded file as bytes
        # file_contents = file.getvalue()

        # # Determine the MIME type of the file
        # mime_type = magic.Magic(mime=True).from_buffer(file_contents)
        format = path.split('.')[-1]
        if format == 'csv':
            try:
                df = pd.read_csv(path)
            except Exception as e:
                raise Exception('Error while trying reading csv file, make sure your file is valid', e)

        elif format == 'xlsx' or format == 'excel':
            try:
                df = pd.read_excel(path)
            except Exception as e:
                raise Exception('Error while trying reading excel file, make sure your file is valid. ', e)

        elif format == 'json':
            try:
                df = pd.read_json(path)
            except Exception as e:
                raise Exception('Error while trying reading json file, make sure your file is valid', e)

        elif format == 'parquet':
            try:
                df = pd.read_parquet(path)
            except Exception as e:
                raise Exception('Error while trying reading Parquet file, make sure your file is valid', e)

        elif format == 'pickle' or format == 'pkl':
            try:
                df = pd.read_pickle(path)
            except Exception as e:
                raise Exception('Error while trying reading pickle file, make sure your file is valid', e)

        elif format == 'feather':
            try:
                df = pd.read_feather(path)
            except Exception as e:
                raise Exception('Error while trying reading feather file, make sure your file is valid', e)

        elif format == 'stata' or format == 'dta':
            try:
                df = pd.read_stata(path)
            except Exception as e:
                raise Exception('Error while trying reading stata file, make sure your file is valid', e)

        elif format == 'html':
            try:
                df = pd.read_html(path)
            except Exception as e:
                raise Exception('Error while trying reading html file, make sure your file is valid', e)

        else:
            raise Exception("Unsupported File Format")
        return df
