"""
Interactive Features Module for ML Capstone App
Adds user interactivity for bonus marks
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def interactive_prediction_regression(model, X, scaler, feature_names):
    """Interactive prediction interface for regression"""
    st.subheader("🎯 Make Your Own Prediction")
    st.markdown("Adjust the sliders to see real-time predictions!")
    
    # Create input sliders for each feature
    user_input = {}
    cols = st.columns(2)
    
    for idx, feature in enumerate(feature_names):
        col = cols[idx % 2]
        with col:
            # Get min, max, and mean from the data
            min_val = float(X[feature].min())
            max_val = float(X[feature].max())
            mean_val = float(X[feature].mean())
            
            user_input[feature] = st.slider(
                f"**{feature}**",
                min_value=min_val,
                max_value=max_val,
                value=mean_val,
                help=f"Range: {min_val:.2f} - {max_val:.2f}"
            )
    
    # Create prediction button
    if st.button("🔮 Predict Score", type="primary"):
        # Prepare input
        input_df = pd.DataFrame([user_input])
        
        # Make prediction
        try:
            prediction = model.predict(input_df)[0]
            
            # Display prediction with animation
            st.markdown("---")
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        padding: 30px; border-radius: 15px; text-align: center;
                        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);'>
                <h2 style='color: white; margin: 0;'>Predicted Average Score</h2>
                <h1 style='color: white; margin: 10px 0; font-size: 3em;'>{prediction:.2f}</h1>
                <p style='color: white; margin: 0; opacity: 0.9;'>strokes per round</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show input summary
            st.markdown("### 📋 Your Input Summary")
            input_display = pd.DataFrame([user_input]).T
            input_display.columns = ['Value']
            st.dataframe(input_display, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error making prediction: {e}")


def interactive_prediction_classification(model, X, scaler, feature_names):
    """Interactive prediction interface for classification"""
    st.subheader("🎯 Predict Match Outcome")
    st.markdown("Adjust match parameters to predict who will win!")
    
    # Create input sliders for each feature
    user_input = {}
    cols = st.columns(2)
    
    for idx, feature in enumerate(feature_names):
        col = cols[idx % 2]
        with col:
            # Get min, max, and mean from the data
            min_val = float(X[feature].min())
            max_val = float(X[feature].max())
            mean_val = float(X[feature].mean())
            
            user_input[feature] = st.slider(
                f"**{feature}**",
                min_value=min_val,
                max_value=max_val,
                value=mean_val,
                help=f"Range: {min_val:.2f} - {max_val:.2f}"
            )
    
    # Create prediction button
    if st.button("🔮 Predict Winner", type="primary"):
        # Prepare input
        input_df = pd.DataFrame([user_input])
        
        # Make prediction
        try:
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0]
            
            # Display prediction with animation
            st.markdown("---")
            
            if prediction == 1:
                result_text = "Higher Ranked Player WINS! 🏆"
                color_start = "#11998e"
                color_end = "#38ef7d"
            else:
                result_text = "Lower Ranked Player WINS! 🥇"
                color_start = "#667eea"
                color_end = "#764ba2"
            
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, {color_start} 0%, {color_end} 100%);
                        padding: 30px; border-radius: 15px; text-align: center;
                        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);'>
                <h2 style='color: white; margin: 0;'>Prediction Result</h2>
                <h1 style='color: white; margin: 10px 0; font-size: 2.5em;'>{result_text}</h1>
                <p style='color: white; margin: 0; opacity: 0.9;'>Confidence: {max(probability)*100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Probability gauge
            st.markdown("### 📊 Prediction Confidence")
            col1, col2 = st.columns(2)
            col1.metric("Win Probability", f"{probability[1]*100:.1f}%")
            col2.metric("Loss Probability", f"{probability[0]*100:.1f}%")
            
            # Show input summary
            st.markdown("### 📋 Match Parameters Used")
            input_display = pd.DataFrame([user_input]).T
            input_display.columns = ['Value']
            st.dataframe(input_display, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error making prediction: {e}")


def interactive_data_explorer(df, dataset_name):
    """Interactive data exploration widget"""
    st.subheader("🔍 Interactive Data Explorer")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Feature selection
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        selected_feature = st.selectbox("Select Feature to Explore", numeric_cols)
    
    with col2:
        # Chart type
        chart_type = st.selectbox("Visualization Type", 
                                   ["Histogram", "Box Plot", "Violin Plot", "Distribution"])
    
    # Display statistics
    st.markdown(f"### 📊 Statistics for {selected_feature}")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Mean", f"{df[selected_feature].mean():.2f}")
    with col2:
        st.metric("Median", f"{df[selected_feature].median():.2f}")
    with col3:
        st.metric("Std Dev", f"{df[selected_feature].std():.2f}")
    with col4:
        st.metric("Range", f"{df[selected_feature].max() - df[selected_feature].min():.2f}")
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if chart_type == "Histogram":
        ax.hist(df[selected_feature].dropna(), bins=30, color='#667eea', alpha=0.7, edgecolor='black')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    elif chart_type == "Box Plot":
        ax.boxplot(df[selected_feature].dropna(), vert=True, patch_artist=True,
                   boxprops=dict(facecolor='#667eea', alpha=0.7),
                   medianprops=dict(color='red', linewidth=2))
        ax.set_ylabel('Value', fontsize=12, fontweight='bold')
    elif chart_type == "Violin Plot":
        parts = ax.violinplot([df[selected_feature].dropna()], positions=[1], 
                              showmeans=True, showmedians=True)
        for pc in parts['bodies']:
            pc.set_facecolor('#667eea')
            pc.set_alpha(0.7)
        ax.set_ylabel('Value', fontsize=12, fontweight='bold')
    else:  # Distribution
        df[selected_feature].dropna().plot(kind='kde', ax=ax, color='#667eea', linewidth=3)
        ax.fill_between(ax.get_xlim(), 0, 
                        df[selected_feature].dropna().plot(kind='kde').get_lines()[0].get_ydata().max(),
                        alpha=0.3, color='#667eea')
        ax.set_ylabel('Density', fontsize=12, fontweight='bold')
    
    ax.set_xlabel(selected_feature, fontsize=12, fontweight='bold')
    ax.set_title(f'{chart_type} of {selected_feature}', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    
    st.pyplot(fig)
    plt.close()


def interactive_feature_comparison(df):
    """Compare two features interactively"""
    st.subheader("📊 Feature Comparison Tool")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        feature_x = st.selectbox("X-Axis Feature", numeric_cols, key='feat_x')
    with col2:
        feature_y = st.selectbox("Y-Axis Feature", numeric_cols, 
                                 index=min(1, len(numeric_cols)-1), key='feat_y')
    with col3:
        plot_type = st.selectbox("Plot Type", ["Scatter", "Hexbin", "2D Density"])
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if plot_type == "Scatter":
        ax.scatter(df[feature_x], df[feature_y], alpha=0.6, c='#667eea', 
                   edgecolors='k', s=50)
        # Add regression line
        z = np.polyfit(df[feature_x].dropna(), df[feature_y].dropna(), 1)
        p = np.poly1d(z)
        ax.plot(df[feature_x].sort_values(), p(df[feature_x].sort_values()), 
                "r--", linewidth=2, label='Trend Line')
        ax.legend()
    elif plot_type == "Hexbin":
        ax.hexbin(df[feature_x], df[feature_y], gridsize=20, cmap='Blues')
        plt.colorbar(ax.collections[0], ax=ax, label='Count')
    else:  # 2D Density
        from scipy.stats import gaussian_kde
        xy = np.vstack([df[feature_x].dropna(), df[feature_y].dropna()])
        z = gaussian_kde(xy)(xy)
        scatter = ax.scatter(df[feature_x], df[feature_y], c=z, s=50, 
                            cmap='viridis', alpha=0.6, edgecolors='k')
        plt.colorbar(scatter, ax=ax, label='Density')
    
    ax.set_xlabel(feature_x, fontsize=12, fontweight='bold')
    ax.set_ylabel(feature_y, fontsize=12, fontweight='bold')
    ax.set_title(f'{plot_type}: {feature_x} vs {feature_y}', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    
    st.pyplot(fig)
    plt.close()
    
    # Show correlation
    correlation = df[[feature_x, feature_y]].corr().iloc[0, 1]
    st.metric("Correlation Coefficient", f"{correlation:.4f}", 
              help="Ranges from -1 (negative) to +1 (positive)")


def interactive_model_comparison(results_df):
    """Interactive model comparison tool"""
    st.subheader("⚖️ Interactive Model Comparison")
    
    # Let user select models to compare
    available_models = results_df['Model'].tolist()
    selected_models = st.multiselect(
        "Select Models to Compare (2-5 models)",
        available_models,
        default=available_models[:3]
    )
    
    if len(selected_models) < 2:
        st.warning("Please select at least 2 models to compare")
        return
    
    # Filter results
    filtered_df = results_df[results_df['Model'].isin(selected_models)]
    
    # Select metrics to compare
    numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns.tolist()
    selected_metrics = st.multiselect(
        "Select Metrics to Compare",
        numeric_cols,
        default=numeric_cols[:3]
    )
    
    if not selected_metrics:
        st.warning("Please select at least one metric")
        return
    
    # Create comparison visualization
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(selected_models))
    width = 0.8 / len(selected_metrics)
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(selected_metrics)))
    
    for idx, metric in enumerate(selected_metrics):
        values = filtered_df[metric].values
        offset = (idx - len(selected_metrics)/2) * width + width/2
        ax.bar(x + offset, values, width, label=metric, color=colors[idx], alpha=0.8)
    
    ax.set_xlabel('Models', fontsize=12, fontweight='bold')
    ax.set_ylabel('Value', fontsize=12, fontweight='bold')
    ax.set_title('Model Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(selected_models, rotation=45, ha='right')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    # Show comparison table
    st.markdown("### 📋 Detailed Comparison")
    comparison_table = filtered_df[['Model'] + selected_metrics].set_index('Model')
    st.dataframe(comparison_table, use_container_width=True)


def interactive_clustering_explorer(X_scaled, labels, n_clusters):
    """Interactive clustering exploration"""
    st.subheader("🔍 Cluster Explorer")
    
    # Cluster selection
    selected_cluster = st.selectbox(
        "Select Cluster to Analyze",
        range(n_clusters),
        format_func=lambda x: f"Cluster {x}"
    )
    
    # Get cluster data
    cluster_mask = labels == selected_cluster
    cluster_size = np.sum(cluster_mask)
    
    # Display cluster info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Cluster Size", cluster_size)
    with col2:
        st.metric("Percentage", f"{(cluster_size/len(labels)*100):.1f}%")
    with col3:
        st.metric("Cluster ID", selected_cluster)
    
    # Show cluster statistics
    st.markdown("### 📊 Cluster Statistics")
    st.markdown(f"**Cluster {selected_cluster}** contains **{cluster_size}** samples out of **{len(labels)}** total samples.")
    
    # Interactive cluster visualization option
    show_only_cluster = st.checkbox("Show only selected cluster", value=False)
    
    return selected_cluster, show_only_cluster
