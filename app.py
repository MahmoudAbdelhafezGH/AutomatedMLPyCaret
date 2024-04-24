import utils
import tempfile
import os
import pandas as pd
import streamlit as st
from classifiers import Classifier, classify
from regressors import Regressor, regress

# from pycaret.regression import get_model_names as get_regression_models

clf_models = ['lr', 'knn', 'rf', 'dt', 'svm', 'xgboost', 'lightgbm', 'catboost']
impute_strategy = ['mode', 'mean', 'zero', 'min', 'max', 'drop']
encoding_methods = ["Label encoding", "One-hot encoding"]
    
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

# Function to get or create SessionState object
def get_session_state():
    if "dataset" not in st.session_state:
        st.session_state.dataset = pd.DataFrame()

# Initialize session state
get_session_state()



with st.sidebar:
    st.image("MLMind.png")
    st.title("AutoStreamML")
    choice = st.radio("Navigation", ["Upload", "Preprocessing", "Profiling", "Train Model"])

if choice == "Upload":
    st.title("Upload your data for modelling!")
    file = st.file_uploader("Upload your Dataset Here")
    if file:
        temp_dir = tempfile.mkdtemp()
        path = os.path.join(temp_dir, file.name)
        with open(path, "wb") as f:
            f.write(file.getvalue())
        dataset = read_file(path)
        st.session_state.dataset = pd.concat([st.session_state.dataset, dataset], ignore_index=True)

if choice == "Preprocessing":        
    st.title("Preprocess of Data")
    if st.session_state.dataset.empty:
        st.subheader("Your Data is empty. Please upload a dataset.")
    else:
        st.subheader("Do you want to drop any column?")
        columns_to_drop = st.multiselect("Select Columns to drop", st.session_state.dataset.columns, key = "Drop-Multiselect-1")
        if st.button("Drop"):
            st.session_state.dataset.drop(columns_to_drop, axis = 1, inplace = True)
            st.write("Data after removing columns:")
        st.dataframe(st.session_state.dataset)

        st.subheader("Handle Missing numerical Values:")
        selected_impute_method_numeric = st.selectbox("Choose a method to handle numeric values:", impute_strategy)
        dataframe_to_impute = st.session_state.dataset
        if st.button("Impute Numeric Values"):
            if selected_impute_method_numeric == 'drop':
                for column in dataframe_to_impute.columns:
                    if pd.api.types.is_numeric_dtype(dataframe_to_impute[column]):
                        dataframe_to_impute.dropna(subset = column, inplace = True)
            elif selected_impute_method_numeric is not None:
                for column in dataframe_to_impute.columns:
                    if pd.api.types.is_numeric_dtype(dataframe_to_impute[column]):
                        utils.impute(dataframe_to_impute, column, selected_impute_method_numeric)
            st.session_state.dataset = dataframe_to_impute
        st.dataframe(st.session_state.dataset)

        st.subheader("Handle Missing categorical Values:")
        selected_impute_method_categorical = st.selectbox("Choose a method to handle categorical values:", ['drop', 'mode'])
        dataframe_to_impute = st.session_state.dataset
        if st.button("Impute Categorical Values"):
            if selected_impute_method_numeric == 'drop':
                for column in dataframe_to_impute.columns:
                    if not pd.api.types.is_numeric_dtype(dataframe_to_impute[column]):
                        dataframe_to_impute.dropna(subset = column, inplace = True)
            elif selected_impute_method_numeric is not None:
                for column in dataframe_to_impute.columns:
                    if not pd.api.types.is_numeric_dtype(dataframe_to_impute[column]):
                        utils.impute(dataframe_to_impute, column, selected_impute_method_numeric)
            st.session_state.dataset = dataframe_to_impute
        st.dataframe(st.session_state.dataset)

        st.subheader("Encode categorical Data:")
        df_to_encode = st.session_state.dataset
        key_id = 0
        for column in df_to_encode.columns:
            if not pd.api.types.is_numeric_dtype(df_to_encode[column]):
                st.write("For the following Column choose an Encoding Method:", column)
                selected_encoding_method = st.selectbox(
                    "Choose a method to handle categorical values:", 
                    encoding_methods, key = "Encoding-select-box-" + str(key_id)
                )
                key_id += 1
                if st.button("Encode", key = "Encode-Button-" + str(key_id)):
                    if selected_encoding_method == 'Label encoding':
                        utils.label_encode(df_to_encode, column)
                        st.dataframe(df_to_encode)
                    else:
                        column_to_encode = df_to_encode[column]                        
                        one_hot = pd.get_dummies(column_to_encode)
                        df_to_encode = df_to_encode.drop(column, axis = 1)
                        df_to_encode = df_to_encode.join(one_hot)
                        st.dataframe(one_hot)
        st.session_state.dataset = df_to_encode
        st.dataframe(st.session_state.dataset)

if choice == "Profiling":
    st.title("Automated Exploratory Data Analysis")
    if not st.session_state.dataset.empty:
        dataset = st.session_state.dataset
        st.write("Dataset:", dataset)
        # profile = ProfileReport(dataset, title="Profiling Report")    
        st.write("Details:")
        st.write("Dataset Head:", dataset.head())
        st.write(f"Number of Columns: {dataset.shape[0]}")
        st.write(f"Number of rows   : {dataset.shape[1]}")
        st.subheader("Info:")
        st.write(f"{dataset.info()}")
        st.subheader("Description:")
        columns_to_analyze = st.multiselect("Which Columns do you want to analyze?", dataset.columns)
        dataframe_to_analyze = dataset[columns_to_analyze]
        if not dataframe_to_analyze.empty:
            df = dataframe_to_analyze.describe(include='all').fillna("").astype("str")
            st.dataframe(df)
    else:
        st.write("No dataset uploaded yet. Please upload a dataset first.")

if choice == "Train Model":
    st.title("Models Training")
    dataset = st.session_state.dataset
    target = st.selectbox("Select your Target: ", dataset.columns, key='select-box-1')
    clf = Classifier()
    reg = Regressor()
    # Check if all values are zeros and ones
    if all(value in [0, 1] for value in dataset[target].unique()):
        task_type = 'Classification'
    else:
        # If not all values are zeros and ones, check if it's numeric for regression
        if pd.api.types.is_numeric_dtype(dataset[target].dtype):
            task_type = 'Regression'
        else:
            task_type = 'Classification'

    st.write(f"This is a {task_type} task.")
    # Get list of available classification models
    if task_type == 'Classification':
        classify(dataset, target)
    elif task_type == 'Regression':
        regress(dataset, target)
