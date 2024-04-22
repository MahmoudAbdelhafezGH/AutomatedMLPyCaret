import tempfile
import os
import pandas as pd
from pycaret.datasets import get_data
import streamlit as st
from classifiers import Classifier, classify
from regressors import Regressor, regress

# from pycaret.regression import get_model_names as get_regression_models

clf_models = ['lr', 'knn', 'rf', 'dt', 'svm', 'xgboost', 'lightgbm', 'catboost']

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
    choice = st.radio("Navigation", ["Upload", "Profiling", "Train Model"])

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
        df = dataset.describe(include='all').fillna("").astype("str")
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
    
    # if st.button("Train"):
    #     clf = Classifier()
    #     exp1 = clf.setup(data=dataset, target=target)
    #     setup_df = clf.pull()
    #     st.write("The setup of the Training:")
    #     st.dataframe(setup_df)
    #     if "All" in model_types:
    #         model_types = clf_models
    #     best_model = clf.compare_models(model_types)
    #     compare_df = clf.pull()
    #     st.info("The evaluated Models:")
    #     st.dataframe(compare_df)
    #     clf.save_model(best_model, "best_model")
    #     with open("best_model.pkl", "rb") as f:
    #         st.download_button("Download the model", f, "trained_model.pkl")
