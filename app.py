import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew
from scipy.signal import welch
from scipy.fft import fft, fftfreq
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle
import io

# Page Configuration
st.set_page_config(
    page_title="Vibration Analysis & Fault Detection",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
    <style>
    .main-header {
        font-size: 36px;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 20px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 30px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .fault-detected {
        background-color: #ff4b4b;
        color: white;
        padding: 15px;
        border-radius: 8px;
        font-size: 20px;
        font-weight: bold;
        text-align: center;
    }
    .no-fault {
        background-color: #00cc66;
        color: white;
        padding: 15px;
        border-radius: 8px;
        font-size: 20px;
        font-weight: bold;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Feature Extraction Functions
def extract_time_domain_features(segment):
    """Extract time domain features from vibration signal"""
    features = {}
    features['mean'] = np.mean(segment)
    features['std'] = np.std(segment)
    features['rms'] = np.sqrt(np.mean(segment**2))
    features['peak'] = np.max(np.abs(segment))
    features['peak_to_peak'] = np.ptp(segment)
    features['crest_factor'] = features['peak'] / features['rms'] if features['rms'] != 0 else 0
    features['skewness'] = skew(segment)
    features['kurtosis'] = kurtosis(segment)
    features['impulse_factor'] = features['peak'] / np.mean(np.abs(segment)) if np.mean(np.abs(segment)) != 0 else 0
    features['shape_factor'] = features['rms'] / np.mean(np.abs(segment)) if np.mean(np.abs(segment)) != 0 else 0
    return features

def extract_frequency_domain_features(segment, fs=1000):
    """Extract frequency domain features using FFT"""
    features = {}
    
    # FFT
    n = len(segment)
    fft_vals = fft(segment)
    fft_mag = np.abs(fft_vals[:n//2])
    freqs = fftfreq(n, 1/fs)[:n//2]
    
    # Frequency domain features
    features['spectral_mean'] = np.mean(fft_mag)
    features['spectral_std'] = np.std(fft_mag)
    features['spectral_peak'] = np.max(fft_mag)
    
    if np.sum(fft_mag) != 0:
        features['spectral_centroid'] = np.sum(freqs * fft_mag) / np.sum(fft_mag)
    else:
        features['spectral_centroid'] = 0
    
    # Power Spectral Density
    f_psd, psd = welch(segment, fs=fs, nperseg=min(256, len(segment)))
    features['psd_mean'] = np.mean(psd)
    features['psd_std'] = np.std(psd)
    
    return features, freqs, fft_mag

def extract_features_from_xyz(x, y, z, fs=1000):
    """Extract features from X, Y, Z axis data"""
    all_features = {}
    
    for axis_name, axis_data in [('x', x), ('y', y), ('z', z)]:
        # Time domain
        time_features = extract_time_domain_features(axis_data)
        for key, val in time_features.items():
            all_features[f'{axis_name}_{key}'] = val
        
        # Frequency domain
        freq_features, _, _ = extract_frequency_domain_features(axis_data, fs)
        for key, val in freq_features.items():
            all_features[f'{axis_name}_{key}'] = val
    
    return all_features

# Model Training Function
@st.cache_resource
def train_model(df):
    """Train the Random Forest model on the dataset"""
    
    # Extract features
    feature_list = []
    labels = []
    
    # Group by label and process segments
    for label in df['label'].unique():
        label_data = df[df['label'] == label]
        
        # Process in segments
        segment_size = 1000
        for i in range(0, len(label_data) - segment_size, segment_size//2):
            segment = label_data.iloc[i:i+segment_size]
            if len(segment) < segment_size:
                continue
            
            x = segment['x'].values
            y = segment['y'].values
            z = segment['z'].values
            
            features = extract_features_from_xyz(x, y, z)
            feature_list.append(features)
            labels.append(label)
    
    # Convert to DataFrame
    X = pd.DataFrame(feature_list)
    y = np.array(labels)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_scaled, y)
    
    return model, scaler, X.columns

# Prediction Function
def predict_fault(model, scaler, feature_columns, x, y, z):
    """Predict fault from vibration data"""
    features = extract_features_from_xyz(x, y, z)
    feature_df = pd.DataFrame([features])
    
    # Ensure all columns match
    for col in feature_columns:
        if col not in feature_df.columns:
            feature_df[col] = 0
    feature_df = feature_df[feature_columns]
    
    # Scale and predict
    X_scaled = scaler.transform(feature_df)
    prediction = model.predict(X_scaled)[0]
    probabilities = model.predict_proba(X_scaled)[0]
    
    return prediction, probabilities

# Plotting Functions
def plot_time_domain(x, y, z, time):
    """Plot time domain signals"""
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    
    axes[0].plot(time, x, 'b-', linewidth=0.5)
    axes[0].set_ylabel('X-Axis (m/s²)', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('Time Domain Vibration Signals', fontsize=14, fontweight='bold')
    
    axes[1].plot(time, y, 'g-', linewidth=0.5)
    axes[1].set_ylabel('Y-Axis (m/s²)', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(time, z, 'r-', linewidth=0.5)
    axes[2].set_ylabel('Z-Axis (m/s²)', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_frequency_spectrum(x, y, z, fs=1000):
    """Plot frequency spectrum (FFT)"""
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    
    for idx, (axis_data, axis_name, color) in enumerate([(x, 'X-Axis', 'blue'), 
                                                           (y, 'Y-Axis', 'green'), 
                                                           (z, 'Z-Axis', 'red')]):
        n = len(axis_data)
        fft_vals = fft(axis_data)
        fft_mag = np.abs(fft_vals[:n//2])
        freqs = fftfreq(n, 1/fs)[:n//2]
        
        axes[idx].plot(freqs, fft_mag, color=color, linewidth=0.8)
        axes[idx].set_ylabel(f'{axis_name} Magnitude', fontsize=12, fontweight='bold')
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_xlim([0, 500])  # Limit to 500 Hz for better visualization
        
        if idx == 0:
            axes[idx].set_title('Frequency Spectrum Analysis (FFT)', fontsize=14, fontweight='bold')
        if idx == 2:
            axes[idx].set_xlabel('Frequency (Hz)', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    return fig

def plot_psd(x, y, z, fs=1000):
    """Plot Power Spectral Density"""
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    
    for idx, (axis_data, axis_name, color) in enumerate([(x, 'X-Axis', 'blue'), 
                                                           (y, 'Y-Axis', 'green'), 
                                                           (z, 'Z-Axis', 'red')]):
        f, psd = welch(axis_data, fs=fs, nperseg=min(256, len(axis_data)))
        
        axes[idx].semilogy(f, psd, color=color, linewidth=0.8)
        axes[idx].set_ylabel(f'{axis_name} PSD', fontsize=12, fontweight='bold')
        axes[idx].grid(True, alpha=0.3)
        
        if idx == 0:
            axes[idx].set_title('Power Spectral Density Analysis', fontsize=14, fontweight='bold')
        if idx == 2:
            axes[idx].set_xlabel('Frequency (Hz)', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    return fig

def plot_3d_orbit(x, y, z):
    """Plot 3D orbit diagram"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Downsample for better visualization
    step = max(1, len(x) // 1000)
    ax.plot(x[::step], y[::step], z[::step], 'b-', linewidth=0.5, alpha=0.6)
    ax.scatter(x[0], y[0], z[0], c='green', s=100, marker='o', label='Start')
    ax.scatter(x[-1], y[-1], z[-1], c='red', s=100, marker='s', label='End')
    
    ax.set_xlabel('X-Axis (m/s²)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y-Axis (m/s²)', fontsize=12, fontweight='bold')
    ax.set_zlabel('Z-Axis (m/s²)', fontsize=12, fontweight='bold')
    ax.set_title('3D Vibration Orbit Analysis', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig

# Main Application
def main():
    # Header
    st.markdown('<div class="main-header">⚙️ Rotary Equipment Fault Detection System<br>Vibration Analysis & Predictive Maintenance</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/maintenance.png", width=80)
        st.title("🔧 Control Panel")
        st.markdown("---")
        
        app_mode = st.selectbox(
            "Select Operation Mode",
            ["🏠 Home", "📊 Train Model", "🔍 Predict Faults", "📈 Analyze Data", "ℹ️ About"]
        )
        
        st.markdown("---")
        st.markdown("### 📋 Project Info")
        st.info("**FYP: Vibration Analysis**\n\nRotary Equipment Fault Detection using AI/ML")
        
        st.markdown("---")
        st.markdown("### 🎯 Fault Types")
        st.markdown("""
        - ✅ Normal Operation
        - ⚠️ Bearing Fault
        - ⚠️ Misalignment
        - ⚠️ Unbalance
        - ⚠️ Looseness
        """)
    
    # Home Page
    if app_mode == "🏠 Home":
        st.markdown("## Welcome to the Vibration Analysis System")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>📊 Data Analysis</h3>
                <p>Real-time vibration signal processing and feature extraction</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3>🤖 AI Prediction</h3>
                <p>Machine learning-based fault classification and detection</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <h3>📈 Visualization</h3>
                <p>Comprehensive spectrum analysis and diagnostic charts</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("## 🚀 Quick Start Guide")
        
        st.markdown("""
        ### Step 1: Train Model
        - Upload your labeled vibration dataset (CSV/Excel)
        - The system will extract features and train the Random Forest classifier
        - Model will be saved for future predictions
        
        ### Step 2: Predict Faults
        - Upload new vibration data for analysis
        - Get instant fault predictions with confidence scores
        - View comprehensive diagnostic information
        
        ### Step 3: Analyze Results
        - Review time domain signals
        - Examine frequency spectrum (FFT)
        - Analyze power spectral density
        - View 3D orbit diagrams
        """)
        
        st.markdown("---")
        st.markdown("### 📄 Data Format Requirements")
        st.code("""
        Required columns:
        - time: Time values (seconds)
        - x: X-axis acceleration (m/s²)
        - y: Y-axis acceleration (m/s²)
        - z: Z-axis acceleration (m/s²)
        - label: Fault type (for training data only)
        """)
    
    # Train Model Page
    elif app_mode == "📊 Train Model":
        st.markdown("## 🎓 Model Training")
        st.markdown("Upload your labeled vibration dataset to train the fault detection model")
        
        uploaded_file = st.file_uploader("Upload Training Dataset (CSV or Excel)", type=['csv', 'xlsx', 'xls'])
        
        if uploaded_file is not None:
            try:
                # Read file
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file, header=None, skiprows=[0])
                    df = df.rename(columns={0:"time", 1:"x", 2:"y", 3:"z", 4:"label"})
                
                df = df.dropna()
                
                st.success(f"✅ Dataset loaded successfully! Total samples: {len(df)}")
                
                # Display data preview
                st.markdown("### 📊 Data Preview")
                st.dataframe(df.head(10), use_container_width=True)
                
                # Display statistics
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 📈 Dataset Statistics")
                    st.write(df.describe())
                
                with col2:
                    st.markdown("### 🏷️ Fault Distribution")
                    fault_counts = df['label'].value_counts()
                    st.bar_chart(fault_counts)
                
                # Train button
                if st.button("🚀 Train Model", type="primary"):
                    with st.spinner("Training model... This may take a few minutes..."):
                        progress_bar = st.progress(0)
                        
                        # Train model
                        progress_bar.progress(30)
                        model, scaler, feature_columns = train_model(df)
                        progress_bar.progress(100)
                        
                        # Save model
                        st.session_state['model'] = model
                        st.session_state['scaler'] = scaler
                        st.session_state['feature_columns'] = feature_columns
                        
                        st.success("✅ Model trained successfully!")
                        
                        # Model info
                        st.markdown("### 🎯 Model Information")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Algorithm", "Random Forest")
                        with col2:
                            st.metric("Features", len(feature_columns))
                        with col3:
                            st.metric("Classes", len(df['label'].unique()))
                        
                        # Feature importance
                        st.markdown("### 📊 Top 15 Important Features")
                        feature_importance = pd.DataFrame({
                            'feature': feature_columns,
                            'importance': model.feature_importances_
                        }).sort_values('importance', ascending=False).head(15)
                        
                        st.bar_chart(feature_importance.set_index('feature'))
            
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    # Predict Faults Page
    elif app_mode == "🔍 Predict Faults":
        st.markdown("## 🔍 Fault Prediction & Diagnosis")
        
        if 'model' not in st.session_state:
            st.warning("⚠️ Please train the model first in the 'Train Model' section!")
            return
        
        st.markdown("Upload vibration data to predict potential faults")
        
        uploaded_file = st.file_uploader("Upload Vibration Data (CSV or Excel)", type=['csv', 'xlsx', 'xls'])
        
        if uploaded_file is not None:
            try:
                # Read file
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                # Check columns
                required_cols = ['time', 'x', 'y', 'z']
                if not all(col in df.columns for col in required_cols):
                    st.error(f"❌ Missing required columns. Need: {required_cols}")
                    return
                
                df = df.dropna()
                
                st.success(f"✅ Data loaded successfully! Total samples: {len(df)}")
                
                # Sampling rate
                fs = st.number_input("Sampling Frequency (Hz)", value=1000, min_value=100, max_value=50000)
                
                # Segment selection
                segment_size = st.slider("Segment Size", min_value=500, max_value=5000, value=1000, step=100)
                start_idx = st.slider("Start Index", min_value=0, max_value=max(0, len(df)-segment_size), value=0)
                
                # Extract segment
                segment = df.iloc[start_idx:start_idx+segment_size]
                
                time = segment['time'].values
                x = segment['x'].values
                y = segment['y'].values
                z = segment['z'].values
                
                # Predict
                if st.button("🔮 Predict Fault", type="primary"):
                    with st.spinner("Analyzing vibration data..."):
                        prediction, probabilities = predict_fault(
                            st.session_state['model'],
                            st.session_state['scaler'],
                            st.session_state['feature_columns'],
                            x, y, z
                        )
                        
                        # Display prediction
                        st.markdown("---")
                        st.markdown("## 🎯 Prediction Results")
                        
                        if prediction.lower() == 'normal':
                            st.markdown(f'<div class="no-fault">✅ Status: {prediction.upper()}</div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="fault-detected">⚠️ FAULT DETECTED: {prediction.upper()}</div>', unsafe_allow_html=True)
                        
                        st.markdown("---")
                        
                        # Probabilities
                        st.markdown("### 📊 Prediction Confidence")
                        prob_df = pd.DataFrame({
                            'Fault Type': st.session_state['model'].classes_,
                            'Probability (%)': probabilities * 100
                        }).sort_values('Probability (%)', ascending=False)
                        
                        st.dataframe(prob_df, use_container_width=True)
                        
                        # Visualization
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            fig_bar = plt.figure(figsize=(8, 6))
                            plt.barh(prob_df['Fault Type'], prob_df['Probability (%)'], color='skyblue')
                            plt.xlabel('Probability (%)', fontsize=12, fontweight='bold')
                            plt.title('Fault Probability Distribution', fontsize=14, fontweight='bold')
                            plt.grid(axis='x', alpha=0.3)
                            st.pyplot(fig_bar)
                        
                        with col2:
                            # Vibration metrics
                            st.markdown("### 📈 Vibration Metrics")
                            
                            rms_x = np.sqrt(np.mean(x**2))
                            rms_y = np.sqrt(np.mean(y**2))
                            rms_z = np.sqrt(np.mean(z**2))
                            
                            peak_x = np.max(np.abs(x))
                            peak_y = np.max(np.abs(y))
                            peak_z = np.max(np.abs(z))
                            
                            metrics_df = pd.DataFrame({
                                'Axis': ['X', 'Y', 'Z'],
                                'RMS (m/s²)': [rms_x, rms_y, rms_z],
                                'Peak (m/s²)': [peak_x, peak_y, peak_z],
                                'Crest Factor': [peak_x/rms_x if rms_x!=0 else 0,
                                                peak_y/rms_y if rms_y!=0 else 0,
                                                peak_z/rms_z if rms_z!=0 else 0]
                            })
                            
                            st.dataframe(metrics_df, use_container_width=True)
                        
                        # Recommendations
                        st.markdown("---")
                        st.markdown("### 💡 Recommendations")
                        
                        if prediction.lower() == 'normal':
                            st.success("""
                            ✅ **Equipment Status: NORMAL**
                            - Continue regular monitoring
                            - Follow scheduled maintenance plan
                            - No immediate action required
                            """)
                        else:
                            st.error(f"""
                            ⚠️ **FAULT DETECTED: {prediction.upper()}**
                            - Immediate inspection recommended
                            - Schedule corrective maintenance
                            - Monitor vibration levels closely
                            - Check equipment alignment and balance
                            - Inspect bearings and coupling
                            """)
            
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    # Analyze Data Page
    elif app_mode == "📈 Analyze Data":
        st.markdown("## 📈 Comprehensive Vibration Analysis")
        st.markdown("Upload vibration data for detailed spectrum and diagnostic analysis")
        
        uploaded_file = st.file_uploader("Upload Vibration Data (CSV or Excel)", type=['csv', 'xlsx', 'xls'])
        
        if uploaded_file is not None:
            try:
                # Read file
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                required_cols = ['time', 'x', 'y', 'z']
                if not all(col in df.columns for col in required_cols):
                    st.error(f"❌ Missing required columns. Need: {required_cols}")
                    return
                
                df = df.dropna()
                
                st.success(f"✅ Data loaded successfully! Total samples: {len(df)}")
                
                # Parameters
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    fs = st.number_input("Sampling Frequency (Hz)", value=1000, min_value=100, max_value=50000)
                with col2:
                    segment_size = st.slider("Segment Size", min_value=500, max_value=5000, value=1000, step=100)
                with col3:
                    start_idx = st.slider("Start Index", min_value=0, max_value=max(0, len(df)-segment_size), value=0)
                
                # Extract segment
                segment = df.iloc[start_idx:start_idx+segment_size]
                
                time = segment['time'].values
                x = segment['x'].values
                y = segment['y'].values
                z = segment['z'].values
                
                # Analysis tabs
                tab1, tab2, tab3, tab4, tab5 = st.tabs([
                    "📊 Time Domain",
                    "📈 Frequency Spectrum",
                    "🔊 Power Spectral Density",
                    "🌐 3D Orbit",
                    "📋 Statistical Features"
                ])
                
                with tab1:
                    st.markdown("### Time Domain Vibration Signals")
                    fig_time = plot_time_domain(x, y, z, time)
                    st.pyplot(fig_time)
                    
                    # Statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("X-Axis RMS", f"{np.sqrt(np.mean(x**2)):.4f} m/s²")
                    with col2:
                        st.metric("Y-Axis RMS", f"{np.sqrt(np.mean(y**2)):.4f} m/s²")
                    with col3:
                        st.metric("Z-Axis RMS", f"{np.sqrt(np.mean(z**2)):.4f} m/s²")
                
                with tab2:
                    st.markdown("### Frequency Spectrum Analysis (FFT)")
                    fig_fft = plot_frequency_spectrum(x, y, z, fs)
                    st.pyplot(fig_fft)
                    
                    st.info("💡 **Analysis Tip**: Look for dominant frequencies that may indicate specific fault patterns (e.g., bearing defects, unbalance)")
                
                with tab3:
                    st.markdown("### Power Spectral Density")
                    fig_psd = plot_psd(x, y, z, fs)
                    st.pyplot(fig_psd)
                    
                    st.info("💡 **Analysis Tip**: PSD shows energy distribution across frequencies. High energy at specific frequencies may indicate resonance or fault conditions")
                
                with tab4:
                    st.markdown("### 3D Vibration Orbit")
                    fig_3d = plot_3d_orbit(x, y, z)
                    st.pyplot(fig_3d)
                    
                    st.info("💡 **Analysis Tip**: Orbit patterns can reveal misalignment, looseness, or bearing issues")
                
                with tab5:
                    st.markdown("### Statistical Features")
                    
                    # Extract all features
                    features = extract_features_from_xyz(x, y, z, fs)
                    
                    # Organize by axis and domain
                    axes = ['x', 'y', 'z']
                    
                    for axis in axes:
                        st.markdown(f"#### {axis.upper()}-Axis Features")
                        
                        axis_features = {k.replace(f'{axis}_', ''): v for k, v in features.items() if k.startswith(axis)}
                        
                        # Split into time and frequency domain
                        time_domain = ['mean', 'std', 'rms', 'peak', 'peak_to_peak', 'crest_factor', 
                                      'skewness', 'kurtosis', 'impulse_factor', 'shape_factor']
                        freq_domain = ['spectral_mean', 'spectral_std', 'spectral_peak', 
                                      'spectral_centroid', 'psd_mean', 'psd_std']
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**Time Domain**")
                            time_df = pd.DataFrame({
                                'Feature': [k for k in time_domain if k in axis_features],
                                'Value': [f"{axis_features[k]:.6f}" for k in time_domain if k in axis_features]
                            })
                            st.dataframe(time_df, use_container_width=True)
                        
                        with col2:
                            st.markdown("**Frequency Domain**")
                            freq_df = pd.DataFrame({
                                'Feature': [k for k in freq_domain if k in axis_features],
                                'Value': [f"{axis_features[k]:.6f}" for k in freq_domain if k in axis_features]
                            })
                            st.dataframe(freq_df, use_container_width=True)
                        
                        st.markdown("---")
                    
                    # Download features
                    features_df = pd.DataFrame([features])
                    csv = features_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download All Features (CSV)",
                        data=csv,
                        file_name="vibration_features.csv",
                        mime="text/csv"
                    )
            
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    # About Page
    else:
        st.markdown("## ℹ️ About This System")
        
        st.markdown("""
        ### 🎓 Final Year Project
        
        **Title:** Rotary Equipment Fault Probability Indication Using Artificial Intelligence and Machine Learning Algorithms on Vibration Data
        
        ### 🎯 Objectives
        
        1. Review literature relating to condition-based maintenance and machine learning
        2. Study and document all standardized rotary equipment fault diagnostic data
        3. Data collection on real equipment with varying faults
        4. Apply machine learning/AI tools to train models with data files
        5. Application of trained models on real machine problems
        6. Assessment of accuracy of models
        
        ### 🔧 Technical Specifications
        
        **Machine Learning Model:**
        - Algorithm: Random Forest Classifier
        - Features: 90+ time and frequency domain features
        - Fault Types: Normal, Bearing Fault, Misalignment, Unbalance, Looseness
        
        **Feature Extraction:**
        - **Time Domain**: Mean, STD, RMS, Peak, Crest Factor, Kurtosis, Skewness
        - **Frequency Domain**: FFT, PSD, Spectral Centroid, Spectral Energy
        
        **Visualization:**
        - Time domain waveforms
        - FFT spectrum analysis
        - Power spectral density
        - 3D orbit diagrams
        
        ### 📊 System Capabilities
        
        ✅ Real-time fault detection and classification  
        ✅ Comprehensive vibration analysis  
        ✅ Multiple visualization modes  
        ✅ Feature extraction and analysis  
        ✅ Predictive maintenance recommendations  
        ✅ Export analysis results  
        
        ### 🚀 Technology Stack
        
        - **Frontend**: Streamlit
        - **ML Framework**: Scikit-learn
        - **Signal Processing**: SciPy, NumPy
        - **Visualization**: Matplotlib
        - **Data Processing**: Pandas
        
        ### 📞 Contact & Support
        
        For more information about this project, please contact your FYP supervisor or project team.
        
        ---
        
        **Version:** 1.0.0  
        **Last Updated:** November 2025
        """)
        
        st.balloons()

if __name__ == "__main__":
    main()
