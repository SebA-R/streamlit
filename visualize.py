import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import streamlit as st


def filter_numeric_columns(df, columns):
    return [column for column in columns if pd.api.types.is_numeric_dtype(df[column])]


def visualize_bar_chart(df, columns, x_column):
    st.subheader("Bar Charts")
    for column in columns:
        st.write(f"### {column} Distribution")
        plt.figure(figsize=(8, 6))
        x = df[x_column]
        y = df[column]
        plt.bar(x, y, alpha=0.7)
        plt.xlabel(x_column)
        plt.ylabel(column)
        plt.xticks(rotation=90)
        plt.tight_layout()
        st.pyplot(plt)
        st.write("---")


def visualize_line_chart(df, columns, x_column):
    st.subheader("Line Charts")
    for column in columns:
        st.write(f"### {column} Trend")
        plt.figure(figsize=(8, 6))
        plt.plot(df[x_column], df[column])
        plt.xlabel(x_column)
        plt.ylabel(column)
        plt.xticks(rotation=90)
        plt.tight_layout()
        st.pyplot(plt)
        st.write("---")


def visualize_scatter_plot(df, columns, x_column):
    st.subheader("Scatter Plots")
    for column in columns:
        st.write(f"### {column} vs {x_column}")
        plt.figure(figsize=(8, 6))
        plt.scatter(df[column], df[x_column])
        plt.xlabel(column)
        plt.ylabel(x_column)
        plt.xticks(rotation=90)
        plt.tight_layout()
        st.pyplot(plt)
        st.write("---")


def visualize_radar_chart(df, columns, x_column):
    st.subheader("Radar Charts")
    for column in columns:
        st.write(f"### {column} Radar Chart")
        data = df.loc[:, [x_column, column]].set_index(x_column).T
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


def visualize_histogram(df, columns):
    st.subheader("Histogram")
    for column in columns:
        st.write(f"### {column} Distribution")
        plt.figure(figsize=(8, 6))
        plt.hist(df[column], bins="auto", alpha=0.7)
        plt.xlabel(column)
        plt.ylabel("Count")
        plt.tight_layout()
        st.pyplot(plt)
        st.write("---")


def visualize_box_plot(df, columns):
    st.subheader("Box Plot")
    for column in columns:
        st.write(f"### {column} Box Plot")
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=df[column])
        plt.xlabel(column)
        plt.tight_layout()
        st.pyplot(plt)
        st.write("---")


def visualize_violin_plot(df, columns):
    st.subheader("Violin Plot")
    for column in columns:
        st.write(f"### {column} Violin Plot")
        plt.figure(figsize=(8, 6))
        sns.violinplot(x=df[column])
        plt.xlabel(column)
        plt.tight_layout()
        st.pyplot(plt)
        st.write("---")


def visualize_area_plot(df, columns, x_column):
    st.subheader("Area Plot")
    for column in columns:
        st.write(f"### {column} Area Plot")
        plt.figure(figsize=(8, 6))
        plt.fill_between(df[x_column], df[column], alpha=0.7)
        plt.xlabel(x_column)
        plt.ylabel(column)
        plt.xticks(rotation=90)
        plt.tight_layout()
        st.pyplot(plt)
        st.write("---")


def visualize_stacked_bar_plot(df, columns, x_column):
    st.subheader("Stacked Bar Plot")
    plt.figure(figsize=(8, 6))
    bottom = None
    for column in columns:
        plt.bar(df[x_column], df[column], bottom=bottom, alpha=0.7, label=column)
        if bottom is None:
            bottom = df[column]
        else:
            bottom += df[column]
    plt.xlabel(x_column)
    plt.ylabel("Value")
    plt.xticks(rotation=90)
    plt.legend()
    plt.tight_layout()
    st.pyplot(plt)
    st.write("---")
