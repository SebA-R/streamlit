import pandas as pd
import streamlit as st
import os
from visualize import (
    filter_numeric_columns,
    visualize_heatmap,
    visualize_histogram,
    visualize_filtered_histogram
)

@st.cache_resource
def load_data(path):
    data = pd.read_csv(path)
    return data

dependent_visualisations = ["Histogram", "Heatmap"]

def main():
    st.title("TD-DFT functionals scoring system")

    # File uploader to allow users to upload their own CSV file
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    # Get the path of the CSV file
    if uploaded_file is not None:
        csv_path = os.path.join(os.getcwd(), uploaded_file.name)
        # Save the uploaded file to the current directory
        with open(csv_path, "wb") as f:
            f.write(uploaded_file.read())
    else:
        csv_path = os.path.join(os.path.dirname(__file__), "scores_without_NaN.csv")

    # Read the CSV file
    df = load_data(csv_path)

    default_limit = 100

    visualization_types = st.multiselect(
        "Select visualization types",
        ["Heatmap", "Histogram", "Filtered Histogram"]
    )

    # Get the total number of rows in the DataFrame
    total_rows = df.shape[0]

    # Get the row_limit value from the sidebar slider
    row_limit = st.sidebar.slider("Row Limit", 0, total_rows, default_limit, key='row_limit')

    # Initialize columns as None
    columns = None

    for ell in dependent_visualisations:
        if ell in visualization_types:
            # Select the columns for visualization
            columns = st.multiselect("Select columns for visualization", df.columns[3:])
            break

    if columns:
        # Create sliders for numeric columns
        sliders = {}
        for column in columns:
            if pd.api.types.is_numeric_dtype(df[column]):
                sliders[column] = st.slider(
                    f"Only display {column} results above:",
                    float(df[column].min()),
                    float(df[column].max()),
                    key=f'slider_{column}'  # Add a unique key for each slider
                )

        # Filter the data based on slider values
        filtered_df = df[filter_numeric_columns(df, columns)]
        for column, value in sliders.items():
            filtered_df = filtered_df[filtered_df[column] >= value]

        else:
            filtered_df = df  # No column selected, show the entire DataFrame

        if "Filtered Histogram" in visualization_types:
            # Get the threshold value from the sidebar slider
            threshold = st.sidebar.slider("Threshold", 0, 100, 0, key='threshold')
            # Display the filtered histogram using the entire DataFrame (df) and threshold
            visualize_filtered_histogram(df, threshold)
        if not filtered_df.empty:
            # Slice the filtered dataframe based on the row_limit value
            filtered_df = filtered_df.head(row_limit)

            if "Heatmap" in visualization_types:
                visualize_heatmap(filtered_df, columns)

            if "Histogram" in visualization_types:
                visualize_histogram(filtered_df, columns)

        else:
            st.write("No data to display.")
    else:
        st.write("No columns selected for visualization.")

if __name__ == "__main__":
    main()
