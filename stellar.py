import streamlit as st             #building web interface
import pickle                      #loading pre-trained models.
import pandas as pd                #data handling
import plotly.express as px        #data visualization
from dataclasses import dataclass  #structuring data
from typing import Any
import logging                     #logging events and errors.
import xgboost as xgb              #logging events and errors.


@dataclass
class StellarFeatures:
    u_filter: float  # Ultraviolet
    g_filter: float  # Green
    r_filter: float  # Red
    i_filter: float  # Near Infrared
    z_filter: float  # Infrared
    redshift: float
    plate_id: int
    mjd: int  # Modified Julian Date
    run_id: int
    fiber_id: int


class StellarClassificationApp:
    CLASS_MAPPING = {0: "Galaxy", 1: "Quasar", 2: "Star"}

    def __init__(self):
        self.setup_logging()
        self.setup_page_config()
        self.load_models()
        self.initialize_session_state()

    def setup_logging(self):
        logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def setup_page_config(self):
        st.set_page_config(
            page_title="Stellar Classification Explorer",
            page_icon="⭐",
            layout="wide",
            initial_sidebar_state="expanded"
        )
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
            .prediction-box {
                padding: 20px;
                background-color: #f0f8ff;
                border-radius: 10px;
                margin: 10px 0;
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

            # Create columns for better organization
            col1, col2 = st.columns(2)

            with col1:
                u_filter = st.number_input('Ultraviolet (U Filter)')
                r_filter = st.number_input('Red (R Filter)')
                z_filter = st.number_input('Infrared (Z Filter)')
                plate_id = st.number_input('Plate ID')
                run_id = st.number_input('Run ID')

            with col2:
                g_filter = st.number_input('Green (G Filter)')
                i_filter = st.number_input('Near Infrared (I Filter)')
                redshift = st.number_input('Redshift')
                mjd = st.number_input('Modified Julian Date')
                fiber_id = st.number_input('Fiber ID')

            return StellarFeatures(u_filter=u_filter,g_filter=g_filter,r_filter=r_filter,i_filter=i_filter,z_filter=z_filter,redshift=redshift,plate_id=plate_id,mjd=mjd,run_id=run_id,fiber_id=fiber_id)

    def predict_stellar_class(self, features: StellarFeatures) -> str:
        try:
            feature_values = [
                features.u_filter, features.g_filter,
                features.r_filter, features.i_filter, features.z_filter,
                features.redshift, features.plate_id, features.mjd,
                features.run_id, features.fiber_id
            ]
            scaled_features = self.scaler.transform([feature_values])
            prediction = self.model.predict(scaled_features)[0]
            return self.CLASS_MAPPING.get(prediction, "Unknown")
        except Exception as e:
            self.logger.error(f"Prediction error: {str(e)}")
            return "Error in prediction", None

    def display_results(self, features: StellarFeatures, prediction_result: tuple):
        prediction= prediction_result
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📝 Input Features")
            feature_df = pd.DataFrame({
                'Parameter': [
                    'Ultraviolet', 'Green', 'Red',
                    'Near Infrared', 'Infrared', 'Redshift', 'Plate ID',
                    'Modified Julian Date', 'Run ID', 'Fiber ID'
                ],
                'Value': [
                    features.u_filter, features.g_filter,
                    features.r_filter, features.i_filter, features.z_filter,
                    features.redshift, features.plate_id, features.mjd,
                    features.run_id, features.fiber_id
                ]
            })
            st.dataframe(feature_df, use_container_width=True)

        with col2:
            st.subheader("🔮 Prediction Results")
            st.markdown(f"""
            <div class="prediction-box">
                <h3 style='color: #1e88e5;'>Stellar Class: {prediction}</h3>
            </div>
            """, unsafe_allow_html=True)


    def visualization_section(self):
        if not st.session_state.feature_history:
            st.info("Make predictions to see visualizations of the data!")
            return

        st.header("📊 Stellar Classification Insights")

        # Create visualization from prediction history
        history_df = pd.DataFrame(st.session_state.feature_history)

        col1, col2 = st.columns(2)

        with col1:
            # Spectral features visualization
            fig1 = px.line(
                history_df,
                y=['u_filter', 'g_filter', 'r_filter', 'i_filter', 'z_filter'],
                title='Spectral Features Across Predictions',
                labels={'value': 'Magnitude', 'variable': 'Filter'}
            )
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            # Redshift distribution
            fig2 = px.histogram(
                history_df,
                x='redshift',
                title='Redshift Distribution',
                nbins=20
            )
            st.plotly_chart(fig2, use_container_width=True)


    def main(self):
        st.title("⭐ Stellar Classification Explorer")
        st.markdown("""
        This application uses machine learning to classify stellar objects based on their 
        observed properties. Enter the parameters in the sidebar to get started.

        ### Features:
        - Spectral measurements across five filters (U, G, R, I, Z)
        - Redshift measurements
        - Observation metadata (Plate ID, MJD, Run ID, Fiber ID)
        """)

        features = self.get_feature_inputs()

        if st.sidebar.button("🔍 Predict Stellar Class"):
            prediction_result = self.predict_stellar_class(features)
            self.display_results(features, prediction_result)

            # Store prediction and features for history
            st.session_state.predictions.append(prediction_result[0])
            st.session_state.feature_history.append(features.__dict__)

        self.visualization_section()


if __name__ == "__main__":
    app = StellarClassificationApp()
    app.main()
