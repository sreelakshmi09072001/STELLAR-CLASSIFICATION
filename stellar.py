import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass
from typing import Dict, List, Any
import logging
import xgboost as xgb


@dataclass
class StellarFeatures:
    alpha: float  # Right Ascension
    u_filter: float  # Ultraviolet
    g_filter: float  # Green
    r_filter: float  # Red
    i_filter: float  # Near Infrared
    z_filter: float  # Infrared
    redshift: float
    plate_id: int
    mjd: int  # Modified Julian Date
    run_id: int


class StellarClassificationApp:
    CLASS_MAPPING = {0: "Galaxy", 1: "Quasar", 2: "Star"}

    def __init__(self):
        self.setup_logging()
        self.setup_page_config()
        self.load_models()
        self.initialize_session_state()

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def setup_page_config(self):
        st.set_page_config(
            page_title="Stellar Classification Explorer",
            page_icon="⭐",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        # Add custom CSS
        st.markdown("""
            <style>
            .stButton>button {
                width: 100%;
                background-color: #4CAF50;
                color: white;
            }
            .sidebar .sidebar-content {
                background-color: #f0f2f6;
            }
            </style>
            """, unsafe_allow_html=True)

    def load_models(self):
        try:
            self.scaler = self._load_pickle("scaler.sav")
            self.model = self._load_pickle("xb_model.sav")
            self.logger.info("Models loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading models: {str(e)}")
            st.error(f"Failed to load machine learning models. Error: {str(e)}")
            st.stop()

    @staticmethod
    def _load_pickle(filename: str) -> Any:
        with open(filename, 'rb') as f:
            return pickle.load(f)

    def initialize_session_state(self):
        if 'predictions' not in st.session_state:
            st.session_state.predictions = []
        if 'feature_history' not in st.session_state:
            st.session_state.feature_history = []

    def get_feature_inputs(self) -> StellarFeatures:
        with st.sidebar:
            st.header("📊 Stellar Object Parameters")

            features = StellarFeatures(
                alpha=st.number_input('Right Ascension (ALPHA)',
                                      format='%f',
                                      step=0.0001,
                                      help="The right ascension of the stellar object"),
                u_filter=st.number_input('Ultraviolet (U Filter)',
                                         format='%f',
                                         step=0.0001,
                                         help="Ultraviolet magnitude measurement"),
                g_filter=st.number_input('Green (G Filter)',
                                         format='%f',
                                         step=0.0001),
                r_filter=st.number_input('Red (R Filter)',
                                         format='%f',
                                         step=0.0001),
                i_filter=st.number_input('Near Infrared (I Filter)',
                                         format='%f',
                                         step=0.0001),
                z_filter=st.number_input('Infrared (Z Filter)',
                                         format='%f',
                                         step=0.0001),
                redshift=st.number_input('Redshift',
                                         format='%f',
                                         step=0.0001,
                                         help="Measure of the object's relative motion"),
                plate_id=st.number_input('Plate ID',
                                         format='%d',
                                         help="Identifier for the observation plate"),
                mjd=st.number_input('Modified Julian Date',
                                    format='%d',
                                    help="Date of observation in MJD format"),
                run_id=st.number_input('Run ID',
                                       format='%d',
                                       help="Identifier for the observation run")
            )
            return features

    def predict_stellar_class(self, features: StellarFeatures) -> str:
        try:
            feature_values = [
                features.alpha, features.u_filter, features.g_filter,
                features.r_filter, features.i_filter, features.z_filter,
                features.redshift, features.plate_id, features.mjd, features.run_id
            ]
            scaled_features = self.scaler.transform([feature_values])
            prediction = self.model.predict(scaled_features)[0]
            return self.CLASS_MAPPING.get(prediction, "Unknown")
        except Exception as e:
            self.logger.error(f"Prediction error: {str(e)}")
            return "Error in prediction"

    def display_results(self, features: StellarFeatures, prediction: str):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📝 Input Features")
            feature_df = pd.DataFrame({
                'Parameter': [
                    'Right Ascension', 'Ultraviolet', 'Green', 'Red',
                    'Near Infrared', 'Infrared', 'Redshift', 'Plate ID',
                    'Modified Julian Date', 'Run ID'
                ],
                'Value': [
                    features.alpha, features.u_filter, features.g_filter,
                    features.r_filter, features.i_filter, features.z_filter,
                    features.redshift, features.plate_id, features.mjd,
                    features.run_id
                ]
            })
            st.dataframe(feature_df, use_container_width=True)

        with col2:
            st.subheader("🔮 Prediction Results")
            st.markdown(f"""
            <div style='padding: 20px; background-color: #f0f8ff; border-radius: 10px;'>
                <h3 style='color: #1e88e5;'>Stellar Class: {prediction}</h3>
            </div>
            """, unsafe_allow_html=True)

    def visualization_section(self):
        st.header("📊 Stellar Classification Insights")

        if not st.session_state.feature_history:
            st.info("Make predictions to see visualizations of the data!")
            return

        # Create visualization from prediction history
        history_df = pd.DataFrame(st.session_state.feature_history)

        # Spectral features visualization
        fig1 = px.line(
            history_df,
            y=['u_filter', 'g_filter', 'r_filter', 'i_filter', 'z_filter'],
            title='Spectral Features Across Predictions',
            labels={'value': 'Magnitude', 'variable': 'Filter'}
        )
        st.plotly_chart(fig1)

        # Redshift distribution
        fig2 = px.histogram(
            history_df,
            x='redshift',
            title='Redshift Distribution',
            nbins=20
        )
        st.plotly_chart(fig2)

    def main(self):
        st.title("⭐ Stellar Classification Explorer")
        st.markdown("""
        This application uses machine learning to classify stellar objects based on their 
        observed properties. Enter the parameters in the sidebar to get started.
        """)

        features = self.get_feature_inputs()

        if st.sidebar.button("🔍 Predict Stellar Class"):
            prediction = self.predict_stellar_class(features)
            self.display_results(features, prediction)

            # Store prediction and features for history
            st.session_state.predictions.append(prediction)
            st.session_state.feature_history.append(features.__dict__)

        if st.session_state.predictions:
            st.sidebar.subheader("🕒 Prediction History")
            for i, pred in enumerate(st.session_state.predictions, 1):
                st.sidebar.markdown(f"{i}. {pred}")

        self.visualization_section()


def main():
    app = StellarClassificationApp()
    app.main()


if __name__ == "__main__":
    main()
