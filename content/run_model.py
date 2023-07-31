import streamlit as st
import pandas as pd
import content.predict as predict

# Function to read SMILES from different file formats
def read_smiles_from_file(file):
    extension = file.name.split(".")[-1].lower()
    if extension == "txt":
        # Explicitly decode the bytes content with the appropriate encoding
        smiles_list = file.read().decode("utf-8").split("\n")
        smiles_list = [smiles.strip() for smiles in smiles_list if smiles.strip()]
    elif extension == "csv":
        # Explicitly decode the bytes content with the appropriate encoding
        df = pd.read_csv(file, encoding="utf-8")
        smiles_list = df["SMILES"].tolist()
        smiles_list = [smiles.strip() for smiles in smiles_list]
    else:
        raise ValueError("Unsupported file format.")
    return smiles_list

def render():
    st.title("SMILES Converter to CSV")

    # Add a dropdown for user choice
    user_choice = st.selectbox("Choose an option:", ("File Upload", "Enter SMILES String"))

    if user_choice == "File Upload":
        uploaded_file = st.file_uploader("Upload a file:", type=["txt", "csv"])

        if uploaded_file is not None:
            try:
                smiles_input = read_smiles_from_file(uploaded_file)  # Make the file content a list named smiles_input
                df_uploaded_smiles = pd.DataFrame({"Uploaded SMILES": smiles_input})
                st.write("Uploaded SMILES:")
                st.dataframe(df_uploaded_smiles)

            except Exception as e:
                st.error(f"Error occurred: {e}")

    elif user_choice == "Enter SMILES String":
        smiles_input = st.text_area("Enter SMILES String", placeholder="Enter Comma Separated SMILES String here...")
        smiles_input = [smiles.strip() for smiles in smiles_input.split(",")]  # Convert to a list

        df_uploaded_smiles = pd.DataFrame({"Uploaded SMILES": smiles_input})
        st.write("Uploaded SMILES:")
        st.dataframe(df_uploaded_smiles)


    

    if st.button("Predict"):
        if uploaded_file:
            if smiles_input: 
                try:
                    df_predictions = predict.main(smiles_input)  # Predict using the user-provided SMILES
                    st.write("Predicted Values:")
                    st.write(df_predictions)

                    st.download_button("Download Predictions", df_predictions.to_csv(index=False), file_name="predictions.csv", mime="text/csv")


                except Exception as e:
                    st.error(f"Error occurred during prediction: {e}")
            else:
                st.warning("Please enter a valid SMILES string for prediction.")
        else:
            st.warning("Please upload a file for prediction.")

if __name__ == "__main__":
    render()