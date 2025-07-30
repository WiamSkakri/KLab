import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time

# Configure Streamlit page
st.set_page_config(
    page_title="CNN Execution Time Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API endpoint configuration
API_BASE_URL = "http://ml-api:8000"  # Use service name in Docker
# API_BASE_URL = "http://localhost:8000"  # For local development


def get_available_models():
    """Fetch available models from the API"""
    try:
        response = requests.get(f"{API_BASE_URL}/models")
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Failed to fetch models: {response.status_code}"}
    except requests.RequestException as e:
        return {"error": f"Connection error: {str(e)}"}


def make_prediction(model_name, cnn_config):
    """Make a prediction using the API"""
    try:
        payload = {
            "model_name": model_name,
            "cnn_config": cnn_config
        }
        response = requests.post(f"{API_BASE_URL}/predict", json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Prediction failed: {response.status_code}", "details": response.text}
    except requests.RequestException as e:
        return {"error": f"Connection error: {str(e)}"}


def get_health_status():
    """Get API health status"""
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            return response.json()
        else:
            return {"status": "unhealthy", "error": f"Status code: {response.status_code}"}
    except requests.RequestException as e:
        return {"status": "unhealthy", "error": f"Connection error: {str(e)}"}


def benchmark_model(model_name, iterations=50):
    """Benchmark a model"""
    try:
        response = requests.get(
            f"{API_BASE_URL}/benchmark/{model_name}?iterations={iterations}")
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Benchmark failed: {response.status_code}"}
    except requests.RequestException as e:
        return {"error": f"Connection error: {str(e)}"}

# Main app


def main():
    st.title("🧠 CNN Execution Time Predictor")
    st.markdown(
        "**Predict CNN execution times across different hardware configurations**")

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", [
        "🔮 Prediction",
        "📊 Model Comparison",
        "📈 Performance Analysis",
        "🏥 System Health",
        "🚀 Model Benchmarking"
    ])

    # Check API health
    health = get_health_status()
    if health.get("status") == "healthy":
        st.sidebar.success("✅ API Status: Healthy")
        st.sidebar.metric("Models Loaded", health.get("models_loaded", 0))
    else:
        st.sidebar.error("❌ API Status: Unhealthy")
        st.sidebar.error(health.get("error", "Unknown error"))
        return

    # Get available models
    models_data = get_available_models()
    if "error" in models_data:
        st.error(f"Failed to load models: {models_data['error']}")
        return

    available_models = list(models_data.get("models", {}).keys())

    if page == "🔮 Prediction":
        prediction_page(available_models)
    elif page == "📊 Model Comparison":
        comparison_page(available_models)
    elif page == "📈 Performance Analysis":
        performance_page(available_models)
    elif page == "🏥 System Health":
        health_page(health)
    elif page == "🚀 Model Benchmarking":
        benchmark_page(available_models)


def prediction_page(available_models):
    """Main prediction interface"""
    st.header("🔮 Make Predictions")

    if not available_models:
        st.warning("No models available for prediction")
        return

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("CNN Configuration")

        # Model selection
        selected_model = st.selectbox("Select Model", available_models)

        # CNN parameters
        batch_size = st.slider("Batch Size", min_value=1,
                               max_value=512, value=32, step=1)
        input_channels = st.slider(
            "Input Channels", min_value=1, max_value=512, value=3, step=1)
        input_height = st.slider(
            "Input Height", min_value=32, max_value=2048, value=224, step=32)
        input_width = st.slider(
            "Input Width", min_value=32, max_value=2048, value=224, step=32)
        output_channels = st.slider(
            "Output Channels", min_value=1, max_value=1024, value=64, step=1)
        kernel_size = st.slider(
            "Kernel Size", min_value=1, max_value=15, value=3, step=2)
        stride = st.slider("Stride", min_value=1, max_value=8, value=1, step=1)

        # Quick presets
        st.subheader("Quick Presets")
        preset_col1, preset_col2, preset_col3 = st.columns(3)

        with preset_col1:
            if st.button("ResNet-50"):
                st.session_state.update({
                    'batch_size': 32, 'input_channels': 3, 'input_height': 224,
                    'input_width': 224, 'output_channels': 64, 'kernel_size': 7, 'stride': 2
                })

        with preset_col2:
            if st.button("VGG-16"):
                st.session_state.update({
                    'batch_size': 16, 'input_channels': 3, 'input_height': 224,
                    'input_width': 224, 'output_channels': 64, 'kernel_size': 3, 'stride': 1
                })

        with preset_col3:
            if st.button("MobileNet"):
                st.session_state.update({
                    'batch_size': 64, 'input_channels': 3, 'input_height': 224,
                    'input_width': 224, 'output_channels': 32, 'kernel_size': 3, 'stride': 1
                })

    with col2:
        st.subheader("Prediction Results")

        if st.button("🚀 Predict Execution Time", type="primary"):
            cnn_config = {
                "batch_size": batch_size,
                "input_channels": input_channels,
                "input_height": input_height,
                "input_width": input_width,
                "output_channels": output_channels,
                "kernel_size": kernel_size,
                "stride": stride
            }

            with st.spinner("Making prediction..."):
                result = make_prediction(selected_model, cnn_config)

            if "error" in result:
                st.error(f"Prediction failed: {result['error']}")
            else:
                # Display results
                execution_time = result.get("execution_time_ms", 0)
                st.success("✅ Prediction Complete!")

                # Metrics display
                col_metric1, col_metric2, col_metric3 = st.columns(3)
                with col_metric1:
                    st.metric("Execution Time", f"{execution_time:.2f} ms")
                with col_metric2:
                    st.metric("Model Used", result.get(
                        "model_type", "Unknown"))
                with col_metric3:
                    st.metric(
                        "Throughput", f"{1000/execution_time:.1f} fps" if execution_time > 0 else "N/A")

                # Visualization
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=execution_time,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Execution Time (ms)"},
                    delta={'reference': 100},
                    gauge={
                        'axis': {'range': [None, 500]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgreen"},
                            {'range': [50, 200], 'color': "yellow"},
                            {'range': [200, 500], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 300
                        }
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

                # Configuration summary
                st.subheader("Configuration Summary")
                config_df = pd.DataFrame([cnn_config]).T
                config_df.columns = ["Value"]
                st.dataframe(config_df)


def comparison_page(available_models):
    """Compare multiple models"""
    st.header("📊 Model Comparison")

    if len(available_models) < 2:
        st.warning("Need at least 2 models for comparison")
        return

    # Select models to compare
    selected_models = st.multiselect(
        "Select models to compare", available_models, default=available_models[:3])

    if len(selected_models) < 2:
        st.warning("Please select at least 2 models")
        return

    # Standard test configuration
    st.subheader("Test Configuration")
    test_config = {
        "batch_size": st.slider("Test Batch Size", 1, 128, 32),
        "input_channels": 3,
        "input_height": st.slider("Test Input Height", 32, 512, 224, 32),
        "input_width": st.slider("Test Input Width", 32, 512, 224, 32),
        "output_channels": st.slider("Test Output Channels", 16, 256, 64),
        "kernel_size": 3,
        "stride": 1
    }

    if st.button("🔄 Run Comparison"):
        results = []

        progress_bar = st.progress(0)
        for i, model in enumerate(selected_models):
            with st.spinner(f"Testing {model}..."):
                result = make_prediction(model, test_config)
                if "error" not in result:
                    results.append({
                        "Model": model,
                        "Execution Time (ms)": result.get("execution_time_ms", 0),
                        "Model Type": result.get("model_type", "Unknown"),
                        "Throughput (fps)": 1000/result.get("execution_time_ms", 1)
                    })
                progress_bar.progress((i + 1) / len(selected_models))

        if results:
            df = pd.DataFrame(results)

            # Display results table
            st.subheader("Comparison Results")
            st.dataframe(df)

            # Visualization
            col1, col2 = st.columns(2)

            with col1:
                # Execution time comparison
                fig1 = px.bar(df, x="Model", y="Execution Time (ms)",
                              title="Execution Time Comparison",
                              color="Model Type")
                st.plotly_chart(fig1, use_container_width=True)

            with col2:
                # Throughput comparison
                fig2 = px.bar(df, x="Model", y="Throughput (fps)",
                              title="Throughput Comparison",
                              color="Model Type")
                st.plotly_chart(fig2, use_container_width=True)


def performance_page(available_models):
    """Performance analysis"""
    st.header("📈 Performance Analysis")

    selected_model = st.selectbox(
        "Select model for analysis", available_models)

    # Parameter sweep
    st.subheader("Parameter Sweep Analysis")

    param_to_sweep = st.selectbox("Parameter to sweep", [
        "batch_size", "input_height", "input_width", "output_channels"
    ])

    if param_to_sweep == "batch_size":
        values = [1, 2, 4, 8, 16, 32, 64, 128]
    elif param_to_sweep in ["input_height", "input_width"]:
        values = [32, 64, 128, 224, 256, 384, 512]
    else:  # output_channels
        values = [16, 32, 64, 128, 256, 512]

    if st.button("🔍 Run Parameter Sweep"):
        base_config = {
            "batch_size": 32,
            "input_channels": 3,
            "input_height": 224,
            "input_width": 224,
            "output_channels": 64,
            "kernel_size": 3,
            "stride": 1
        }

        sweep_results = []
        progress_bar = st.progress(0)

        for i, value in enumerate(values):
            config = base_config.copy()
            config[param_to_sweep] = value

            result = make_prediction(selected_model, config)
            if "error" not in result:
                sweep_results.append({
                    param_to_sweep: value,
                    "execution_time_ms": result.get("execution_time_ms", 0)
                })

            progress_bar.progress((i + 1) / len(values))

        if sweep_results:
            df = pd.DataFrame(sweep_results)

            fig = px.line(df, x=param_to_sweep, y="execution_time_ms",
                          title=f"Execution Time vs {param_to_sweep}",
                          markers=True)
            st.plotly_chart(fig, use_container_width=True)


def health_page(health_data):
    """System health monitoring"""
    st.header("🏥 System Health")

    # API Status
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("API Status", health_data.get("status", "Unknown"))
    with col2:
        st.metric(
            "Uptime", f"{health_data.get('uptime_seconds', 0)/3600:.1f}h")
    with col3:
        st.metric("Models Loaded", health_data.get("models_loaded", 0))
    with col4:
        st.metric(
            "Memory Usage", f"{health_data.get('system_info', {}).get('memory_percent', 0):.1f}%")

    # System metrics
    if "system_info" in health_data:
        sys_info = health_data["system_info"]

        col1, col2 = st.columns(2)

        with col1:
            # CPU usage gauge
            fig_cpu = go.Figure(go.Indicator(
                mode="gauge+number",
                value=sys_info.get("cpu_percent", 0),
                title={'text': "CPU Usage (%)"},
                gauge={'axis': {'range': [None, 100]},
                       'bar': {'color': "darkblue"},
                       'steps': [{'range': [0, 50], 'color': "lightgreen"},
                                 {'range': [50, 80], 'color': "yellow"},
                                 {'range': [80, 100], 'color': "red"}]}
            ))
            fig_cpu.update_layout(height=300)
            st.plotly_chart(fig_cpu, use_container_width=True)

        with col2:
            # Memory usage gauge
            fig_mem = go.Figure(go.Indicator(
                mode="gauge+number",
                value=sys_info.get("memory_percent", 0),
                title={'text': "Memory Usage (%)"},
                gauge={'axis': {'range': [None, 100]},
                       'bar': {'color': "darkgreen"},
                       'steps': [{'range': [0, 60], 'color': "lightgreen"},
                                 {'range': [60, 85], 'color': "yellow"},
                                 {'range': [85, 100], 'color': "red"}]}
            ))
            fig_mem.update_layout(height=300)
            st.plotly_chart(fig_mem, use_container_width=True)


def benchmark_page(available_models):
    """Model benchmarking"""
    st.header("🚀 Model Benchmarking")

    selected_model = st.selectbox(
        "Select model to benchmark", available_models)
    iterations = st.slider("Number of iterations", 10, 200, 50)

    if st.button("🏃‍♂️ Run Benchmark"):
        with st.spinner(f"Benchmarking {selected_model} for {iterations} iterations..."):
            result = benchmark_model(selected_model, iterations)

        if "error" in result:
            st.error(f"Benchmark failed: {result['error']}")
        else:
            st.success("✅ Benchmark Complete!")

            # Display metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Avg Latency",
                          f"{result.get('avg_latency_ms', 0):.2f} ms")
            with col2:
                st.metric("Min Latency",
                          f"{result.get('min_latency_ms', 0):.2f} ms")
            with col3:
                st.metric("Max Latency",
                          f"{result.get('max_latency_ms', 0):.2f} ms")
            with col4:
                st.metric(
                    "Throughput", f"{result.get('predictions_per_second', 0):.1f} pred/s")

            # Additional details
            st.subheader("Benchmark Details")
            details = {
                "Model Name": result.get("model_name", "Unknown"),
                "Total Iterations": result.get("iterations", 0),
                "Successful Predictions": result.get("successful_predictions", 0),
                "Success Rate": f"{(result.get('successful_predictions', 0) / result.get('iterations', 1)) * 100:.1f}%",
                "Timestamp": result.get("timestamp", "Unknown")
            }

            details_df = pd.DataFrame([details]).T
            details_df.columns = ["Value"]
            st.dataframe(details_df)


if __name__ == "__main__":
    main()
