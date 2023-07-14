import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import os


@st.cache_resource
def load_data(path):
    data = pd.read_csv(path, nrows=100)
    return data


def main():
    st.title("Molecule Data Visualizer")

    # Get the path of the CSV file
    csv_path = os.path.join(os.path.dirname(__file__), "scores_without_NaN.csv")

    # Read the CSV file
    df = load_data(csv_path)

    default_limit = 100

    row_limit = st.slider("Row Limit", 10, 1000, default_limit)

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
                )

        # Filter the data based on slider values
        filtered_df = df.copy()
        for column, value in sliders.items():
            filtered_df = filtered_df[filtered_df[column] >= value]

        if not filtered_df.empty:
            # Visualize the data
            filtered_df = filtered_df.head(row_limit)

            visualization_types = st.multiselect(
                "Select visualization types", ["Bar Chart", "Line Chart", "Scatter Plot", "Radar Chart", "Heatmap"]
            )

            if "Bar Chart" in visualization_types:
                visualize_bar_chart(filtered_df, columns)

            if "Line Chart" in visualization_types:
                visualize_line_chart(filtered_df, columns)

            if "Scatter Plot" in visualization_types:
                visualize_scatter_plot(filtered_df, columns)

            if "Radar Chart" in visualization_types:
                visualize_radar_chart(filtered_df, columns)

            if "Heatmap" in visualization_types:
                visualize_heatmap(filtered_df, columns)
        else:
            st.write("No data to display.")
    else:
        st.write("No columns selected for visualization.")


def visualize_bar_chart(df, columns):
    st.subheader("Bar Charts")
    for column in columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            st.write(f"### {column} Distribution")
            plt.figure(figsize=(8, 6))
            x = df["Name"]
            y = df[column]
            plt.bar(x, y, alpha=0.7)
            plt.xlabel("Name")
            plt.ylabel(column)
            plt.xticks(rotation=90)
            plt.tight_layout()
            st.pyplot(plt)
            st.write("---")


def visualize_line_chart(df, columns):
    st.subheader("Line Charts")
    for column in columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            st.write(f"### {column} Trend")
            plt.figure(figsize=(8, 6))
            plt.plot(df["Name"], df[column])
            plt.xlabel("Name")
            plt.ylabel(column)
            plt.xticks(rotation=90)
            plt.tight_layout()
            st.pyplot(plt)
            st.write("---")


def visualize_scatter_plot(df, columns):
    st.subheader("Scatter Plots")
    for column in columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            st.write(f"### {column} vs Name")
            plt.figure(figsize=(8, 6))
            plt.scatter(df[column], df["Name"])
            plt.xlabel(column)
            plt.ylabel("Name")
            plt.xticks(rotation=90)
            plt.tight_layout()
            st.pyplot(plt)
            st.write("---")


def visualize_radar_chart(df, columns):
    st.subheader("Radar Charts")
    for column in columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            st.write(f"### {column} Radar Chart")
            data = df.loc[:, ["Name", column]].set_index("Name").T
            fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(polar=True))
            labels = data.columns
            num_vars = len(labels)
            angles = [(n / float(num_vars) * 2 * 3.14) for n in range(num_vars)]
            angles += angles[:1]
            for i, (name, values) in enumerate(data.iterrows()):
                values = values.tolist()
                values += values[:1]
                ax.plot(angles, values, linewidth=1, linestyle="solid", label=name)
                ax.fill(angles, values, alpha=0.25)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(labels)
            ax.set_yticks([20, 40, 60, 80, 100])
            ax.set_yticklabels(["20", "40", "60", "80", "100"])
            ax.spines["polar"].set_visible(False)
            ax.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))
            plt.tight_layout()
            st.pyplot(plt)
            st.write("---")


def visualize_heatmap(df, columns):
    st.subheader("Heatmap")
    plt.figure(figsize=(10, 6))
    heatmap_df = df[columns].corr()
    sns.heatmap(heatmap_df, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    st.pyplot(plt)
    st.write("---")


if __name__ == "__main__":
    main()
