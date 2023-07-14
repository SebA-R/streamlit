import pandas as pd
import streamlit as st
import os
from visualize import (
    filter_numeric_columns,
    visualize_bar_chart,
    visualize_line_chart,
    visualize_scatter_plot,
    visualize_radar_chart,
    visualize_heatmap,
    visualize_histogram,
    visualize_box_plot,
    visualize_violin_plot,
    visualize_area_plot,
    visualize_stacked_bar_plot,
)


@st.cache_resource
def load_data(path):
    data = pd.read_csv(path)
    return data


def main():
    st.title("Molecule Data Visualizer")

    # Get the path of the CSV file
    csv_path = os.path.join(os.path.dirname(__file__), "scores_without_NaN.csv")

    # Read the CSV file
    df = load_data(csv_path)

    default_limit = 100

    # Get the row_limit value from the sidebar slider
    row_limit = st.sidebar.slider("Row Limit", 10, 21228, default_limit, key='row_limit')

    # Select the columns for visualization
    columns = st.multiselect("Select columns for visualization", df.columns[3:])

    if len(columns) > 0:
        # Create sliders for numeric columns
        selected_columns = st.columns(len(columns))
        sliders = {}
        for i, column in enumerate(columns):
            if pd.api.types.is_numeric_dtype(df[column]):
                sliders[column] = selected_columns[i].slider(
                    f"Only display {column} results above:",
                    float(df[column].min()),
                    float(df[column].max()),
                    key=f'slider_{column}'  # Add a unique key for each slider
                )

        # Filter the data based on slider values
        filtered_df = df[filter_numeric_columns(df, columns)]
        for column, value in sliders.items():
            filtered_df = filtered_df[filtered_df[column] >= value]

        if not filtered_df.empty:
            # Slice the filtered dataframe based on the row_limit value
            filtered_df = filtered_df.head(row_limit)

            visualization_types = st.multiselect(
                "Select visualization types",
                ["Bar Chart", "Line Chart", "Scatter Plot", "Radar Chart", "Heatmap", "Histogram", "Box Plot", "Violin Plot", "Area Plot", "Stacked Bar Plot"]
            )

            if "Bar Chart" in visualization_types:
                visualize_bar_chart(filtered_df, columns, "empty for now")

            if "Line Chart" in visualization_types:
                visualize_line_chart(filtered_df, columns)

            if "Scatter Plot" in visualization_types:
                visualize_scatter_plot(filtered_df, columns)

            if "Radar Chart" in visualization_types:
                visualize_radar_chart(filtered_df, columns)

            if "Heatmap" in visualization_types:
                visualize_heatmap(filtered_df, columns)

            if "Histogram" in visualization_types:
                visualize_histogram(filtered_df, columns)

            if "Box Plot" in visualization_types:
                visualize_box_plot(filtered_df, columns)

            if "Violin Plot" in visualization_types:
                visualize_violin_plot(filtered_df, columns)

            if "Area Plot" in visualization_types:
                visualize_area_plot(filtered_df, columns)

            if "Stacked Bar Plot" in visualization_types:
                visualize_stacked_bar_plot(filtered_df, columns)
        else:
            st.write("No data to display.")
    else:
        st.write("No columns selected for visualization.")


if __name__ == "__main__":
    main()
