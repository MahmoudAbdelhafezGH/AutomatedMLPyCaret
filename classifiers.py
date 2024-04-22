from pycaret.classification import *
import streamlit as st
import pandas as pd

def classify(dataset, target):
    setup(data=dataset, target=target)
    all_models = models()
    st.write("Available Classifiers", all_models)
    model_types = st.multiselect("Select Models for Training: ", all_models['Name'] ,key='classification-key-1')
    if st.button("Train", key='classification-button-1'):
        # Iterate over each name in model_types["Name"] column
        for name in model_types:
            models_ids = all_models[all_models["Name"] == name].index.tolist()
            st.write('Selected Models:', models_ids)
        
        setup_df = pull()
        st.write("The setup of the Training:", setup_df)
        st.dataframe(setup_df)
        best_model = compare_models(models_ids)
        compare_df = pull()
        st.info("The evaluated Models:")
        st.dataframe(compare_df)
        save_model(best_model, "best_model")
        with open("best_model.pkl", "rb") as f:
            st.download_button("Download the model", f, "trained_model.pkl", key='classification-dbutton-1')

class Classifier():
    def __init__(self, session_id = 444):
        self.session_id = session_id
        self.exp = None

    def setup(self, data, target, profile = True ):
        setup(data=data, target=target, session_id = self.session_id, profile = True)

    def create_model(self, model):
        create_model(model)

    def evaluate(self, model):
        evaluate_model(model)

    def tune_model(self, model):
        tune_model(model)

    def compare_models(self, models):
        compare_models(include = models)

    def ensemble_model(self, models):
        ensemble_model(models)

    def finalize_model(self, model):
        finalize_model(model)

    def save_model(self, model, model_name="BestsEver"):
        save_model(model = model, model_name=model_name)

    def pull(self):
        pull()

    def load_model(self, model_name):
        load_model(model_name)

    def models(self):
        models()