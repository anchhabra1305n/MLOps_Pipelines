
import streamlit as st
import pandas as pd
import mlflow
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
import base64  # Add this import statement

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

def download_predictions(output):
    # Convert predictions to a DataFrame
    #df = pd.DataFrame(predictions, columns=["Predictions"])

    # Create a CSV file from the DataFrame
    csv = output.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">Download Predictions CSV File</a>'
    return href

# Function to evaluate model performance
def evaluate_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return accuracy, precision, recall, f1

# Function to compute historical metrics
def compute_historical_metrics():
    # Define your logic to compute historical metrics here
    pass

# Define the layout of the app
def main():
    st.title("Model Monitoring and Churn Prediction")

    # Display initial message
    #st.write("Please wait while the application initializes...")

    # Option selection
    option = st.sidebar.selectbox("Select an option:", ["Select one of the below option","Model Monitoring", "Churn Model Prediction"])

    # Render selected option
    if option == "Select one of the below option":
        select_monitoring()
    elif option == "Model Monitoring":
        render_model_monitoring()
    elif option == "Churn Model Prediction":
        render_churn_prediction()

# Function to render model monitoring section
def select_monitoring():
    st.header("Please select one of the options listed in side bar")
def render_model_monitoring():
    st.header("Model Monitoring")
    # Load current data and model
    st.sidebar.header("Upload CSV Data")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        current_data = df
        X = current_data.drop(["churn"], axis=1)
        Y = current_data["churn"]
        X = pd.get_dummies(X, drop_first=True)

        artifact_uri = "runs:/2956f1fcd7b74ed9be2d85df3b554ddc/model"
        model = mlflow.pyfunc.load_model(artifact_uri)
        # Make predictions using the model
        predicted_labels = model.predict(X)
        # Evaluate current model performance
        accuracy, precision, recall, f1 = evaluate_model(Y, predicted_labels)
        # Load historical performance metrics
        # Fetch the run ID using the artifact URI
        run_id = mlflow.get_run(artifact_uri.split("/")[-2]).info.run_id

        # Fetch the run using the run ID
        run = mlflow.get_run(run_id)

        # Retrieve metrics from the run
        metrics = run.data.metrics

        # Extract historical accuracy, precision, recall, and F1 scores
        historical_accuracy = metrics.get("accuracy")
        historical_precision = metrics.get("precision")
        historical_recall = metrics.get("recall")
        historical_f1 = metrics.get("f1")


        # Display current and historical metrics
        st.write("Current Performance Metrics:")
        st.write(f"Accuracy: {accuracy}")
        st.write(f"Precision: {precision}")
        st.write(f"Recall: {recall}")
        st.write(f"F1 Score: {f1}")

        st.write("Historical Performance Metrics:")
        st.write(f"Accuracy: {historical_accuracy}")
        st.write(f"Precision: {historical_precision}")
        st.write(f"Recall: {historical_recall}")
        st.write(f"F1 Score: {historical_f1}")

        # Compare current performance with historical performance
        if accuracy < historical_accuracy:
            st.warning("Warning: Current accuracy is lower than historical accuracy!")

# Function to render churn model prediction section
def render_churn_prediction():
    st.header("Churn Model Prediction App")
    artifact_uri = "runs:/d94ed5e555614b0faea846186ed68abe/model"
    model = mlflow.pyfunc.load_model(artifact_uri)

    # Upload CSV file through Streamlit
    st.sidebar.header("Upload CSV Data")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        st.sidebar.subheader("Uploaded Data:")
        df = pd.read_csv(uploaded_file)
        st.sidebar.write(df)

        # Make predictions using the loaded model
        predictions = model.predict(df)
        my_predictions = pd.DataFrame(predictions)
        my_predictions.columns = ['churn']
        output = df.join(my_predictions)
        # Display predictions
        st.subheader("Churn Predictions:")
        #output.write(predictions)
        # Allow user to download predictions as CSV
        st.write("")
        st.write("Download predictions as CSV:")
        st.markdown(download_predictions(output), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
