import streamlit as st
import pandas as pd
from content.predict import main as predict_main
from content.mc_dropout import mc_dropout_main
from content.predict import functionals_dictionary
import deepchem as dc
from rdkit import Chem
from deepchem.feat import MolGraphConvFeaturizer

# Function to read SMILES from different file formats


def read_smiles_from_file(file):
    extension = file.name.split(".")[-1].lower()
    if extension == "txt":
        smiles_list = file.read().decode("utf-8").split("\n")
        smiles_list = [smiles.strip()
                       for smiles in smiles_list if smiles.strip()]
    elif extension == "csv":
        df = pd.read_csv(file, encoding="utf-8")
        smiles_list = df["SMILES"].tolist()
        smiles_list = [smiles.strip() for smiles in smiles_list]
    else:
        raise ValueError("Unsupported file format.")
    return smiles_list


def render():
    if 'smiles_input' not in st.session_state:
        st.session_state.smiles_input = []

    st.title("Predict from single SMILES or CSV")

    st.write("Below, you can enter any valid SMILES string and DELFI will predict the score for the molecule(s) given.\nPlease note that DELFI has been trained on QM8 dataset, which contains organic molecules with up to 8 CONF atoms.\nAny prediction on molecules that significantly differ from the ones contained in the training test should be carefully benchmarked")
    
    # Add a dropdown for user choice
    user_choice = st.selectbox(
        "Choose an option:", ("Enter SMILES String", "File Upload"))
    
    mc_checkbox = st.checkbox(
        "Perform Monte Carlo Predictions for Uncertainty Estimation")

    if user_choice == "File Upload":
        uploaded_file = st.file_uploader("Upload a file:", type=["txt", "csv"])

        if uploaded_file is not None:
            try:
                st.session_state.smiles_input = read_smiles_from_file(
                    uploaded_file)
                df_uploaded_smiles = pd.DataFrame(
                    {"Uploaded SMILES": st.session_state.smiles_input})
                st.write("Uploaded SMILES:")
                st.dataframe(df_uploaded_smiles)

            except Exception as e:
                st.error(f"Error occurred: {e}")

    elif user_choice == "Enter SMILES String":
        # Checkbox to choose Monte Carlo predictions
        

        st.session_state.smiles_input = st.text_area(
            "Enter SMILES String", placeholder="Enter Comma Separated SMILES String here...")
        st.session_state.smiles_input = [
            smiles.strip() for smiles in st.session_state.smiles_input.split(",")]

        df_uploaded_smiles = pd.DataFrame(
            {"Uploaded SMILES": st.session_state.smiles_input})
        st.write("Uploaded SMILES:")
        st.dataframe(df_uploaded_smiles)

    

    if st.button("Predict"):
        if st.session_state.smiles_input:
            try:
                filtered_smiles, predictions = predict_main(
                    st.session_state.smiles_input)
                df_predictions = pd.concat([pd.DataFrame(filtered_smiles, columns=['SMILES']), pd.DataFrame(
                    predictions, columns=functionals_dictionary.keys())], axis=1)
                st.write("Predicted Values:")
                st.write(df_predictions)

                st.download_button("Download Predictions", df_predictions.to_csv(
                    index=False), file_name="predictions.csv", mime="text/csv")

                # Monte Carlo Dropout for uncertainty estimation
                if mc_checkbox:
                    st.write(
                        "Calculating uncertainty with Monte Carlo Dropout over 50 iterations...")
                    try:
                        # Convert SMILES strings to RDKit molecules
                        molecules = [Chem.MolFromSmiles(smiles) for smiles in filtered_smiles]

                        # Create a ConvMolFeaturizer
                        featurizer = MolGraphConvFeaturizer()

                        # Convert RDKit molecules to ConvMol objects
                        conv_mols = [featurizer.featurize([mol])[0] for mol in molecules]
                        
                        df_mc_predictions = dc.data.NumpyDataset(X=conv_mols, y=predictions)
                        mc_predictions = mc_dropout_main(df_mc_predictions)
                        
                        st.write("Monte Carlo Predictions:")
                        reordered_mc_predictions = [[prediction[0] for prediction in mc_predictions],[prediction[1] for prediction in mc_predictions]]
                        df_predictions = pd.concat([pd.DataFrame(filtered_smiles, columns=['SMILES']), pd.DataFrame(reordered_mc_predictions, columns=[functional+" (mean, variance)" for functional in functionals_dictionary.keys()])], axis=1)
                        st.write(df_predictions)
                    except Exception as e:
                        st.error(
                            f"Error occurred during uncertainty calculation: {e}")

            except:
                st.error(
                    "Make sure that you have entered at least one valid SMILES string.")

        else:
            st.warning("Please enter a valid SMILES string for prediction.")


if __name__ == "__main__":
    render()
