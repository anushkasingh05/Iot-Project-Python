import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
import time
import os
from dotenv import load_dotenv

# Import custom modules
from src.data_ingestion import SensorDataIngestion
from src.rag_engine import RAGEngine
from src.predictive_maintenance import PredictiveMaintenance
from src.anomaly_detection import AnomalyDetection
from src.energy_optimization import EnergyOptimization
from src.utils import generate_sample_data, create_sample_manuals

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="IoT Smart Building RAG System",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #2d3748;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
        color: #ffffff;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .alert-high {
        background-color: #742a2a;
        border-left: 4px solid #f56565;
        color: #ffffff;
    }
    .alert-medium {
        background-color: #744210;
        border-left: 4px solid #ed8936;
        color: #ffffff;
    }
    .alert-low {
        background-color: #22543d;
        border-left: 4px solid #48bb78;
        color: #ffffff;
    }
    .alert-card h4 {
        color: #ffffff;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .alert-card p {
        color: #e2e8f0;
        margin-bottom: 0.25rem;
    }
    .alert-card strong {
        color: #ffffff;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

class SmartBuildingRAG:
    def __init__(self):
        self.initialize_components()
        self.setup_data()
    
    def initialize_components(self):
        """Initialize all system components"""
        try:
            self.sensor_ingestion = SensorDataIngestion()
            self.rag_engine = RAGEngine()
            self.predictive_maintenance = PredictiveMaintenance()
            self.anomaly_detection = AnomalyDetection()
            self.energy_optimization = EnergyOptimization()
        except Exception as e:
            st.error(f"Error initializing components: {str(e)}")
    
    def setup_data(self):
        """Setup sample data if not exists"""
        if not os.path.exists("data"):
            os.makedirs("data")
            os.makedirs("data/sensor_data")
            os.makedirs("data/manuals")
            os.makedirs("data/building_specs")
            
            # Generate sample data
            generate_sample_data()
            create_sample_manuals()
    
    def main_dashboard(self):
        """Main dashboard view"""
        st.markdown('<h1 class="main-header">🏢 Smart Building IoT RAG System</h1>', unsafe_allow_html=True)
        
        # Real-time metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="System Health",
                value="92%",
                delta="+2%"
            )
        
        with col2:
            st.metric(
                label="Energy Efficiency",
                value="87%",
                delta="-3%"
            )
        
        with col3:
            st.metric(
                label="Active Alerts",
                value="3",
                delta="-1"
            )
        
        with col4:
            st.metric(
                label="Maintenance Due",
                value="2",
                delta="+1"
            )
        
        # Main content area
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Real-time Monitoring", 
            "🔍 RAG Insights", 
            "🔮 Predictive Maintenance",
            "⚡ Energy Optimization",
            "🚨 Anomaly Detection"
        ])
        
        with tab1:
            self.real_time_monitoring()
        
        with tab2:
            self.rag_insights()
        
        with tab3:
            self.predictive_maintenance_tab()
        
        with tab4:
            self.energy_optimization_tab()
        
        with tab5:
            self.anomaly_detection_tab()
    
    def real_time_monitoring(self):
        """Real-time sensor monitoring dashboard"""
        st.header("📊 Real-time Sensor Monitoring")
        
        # Get current sensor data
        sensor_data = self.sensor_ingestion.get_current_data()
        
        if sensor_data is not None:
            # Temperature and Humidity
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Temperature Sensors")
                temp_data = sensor_data[sensor_data['sensor_type'] == 'temperature']
                fig_temp = px.line(temp_data, x='timestamp', y='value', 
                                 color='location', title='Temperature Trends')
                st.plotly_chart(fig_temp, use_container_width=True)
            
            with col2:
                st.subheader("Humidity Sensors")
                humidity_data = sensor_data[sensor_data['sensor_type'] == 'humidity']
                fig_humidity = px.line(humidity_data, x='timestamp', y='value', 
                                     color='location', title='Humidity Trends')
                st.plotly_chart(fig_humidity, use_container_width=True)
            
            # HVAC and Lighting
            col3, col4 = st.columns(2)
            
            with col3:
                st.subheader("HVAC System Status")
                hvac_data = sensor_data[sensor_data['sensor_type'] == 'hvac']
                fig_hvac = px.bar(hvac_data, x='location', y='value', 
                                color='status', title='HVAC Performance')
                st.plotly_chart(fig_hvac, use_container_width=True)
            
            with col4:
                st.subheader("Lighting System")
                lighting_data = sensor_data[sensor_data['sensor_type'] == 'lighting']
                fig_lighting = px.pie(lighting_data, values='value', names='location', 
                                    title='Lighting Usage Distribution')
                st.plotly_chart(fig_lighting, use_container_width=True)
            
            # System health overview
            st.subheader("System Health Overview")
            health_data = self.sensor_ingestion.get_system_health()
            
            if health_data is not None:
                # Create radar chart using go.Figure instead of px.radar
                fig_health = go.Figure()
                
                fig_health.add_trace(go.Scatterpolar(
                    r=health_data['value'].tolist(),
                    theta=health_data['system'].tolist(),
                    fill='toself',
                    name='System Health'
                ))
                
                fig_health.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 100]
                        )),
                    showlegend=False,
                    title='System Health Overview'
                )
                
                st.plotly_chart(fig_health, use_container_width=True)
    
    def rag_insights(self):
        """RAG-powered insights and manual search"""
        st.header("🔍 RAG-powered Insights")
        
        # Search interface
        col1, col2 = st.columns([3, 1])
        
        with col1:
            query = st.text_input(
                "Search maintenance manuals, troubleshooting guides, or building specifications:",
                placeholder="e.g., How to troubleshoot HVAC system? What are the maintenance procedures for lighting system?"
            )
        
        with col2:
            search_type = st.selectbox(
                "Search Type",
                ["All", "Maintenance Manuals", "Troubleshooting", "Building Specs", "Best Practices"]
            )
        
        if st.button("🔍 Search", type="primary"):
            if query:
                with st.spinner("Searching through knowledge base..."):
                    results = self.rag_engine.search(query, search_type)
                    
                    if results:
                        st.success(f"Found {len(results)} relevant documents")
                        
                        for i, result in enumerate(results[:5]):  # Show top 5 results
                            with st.expander(f"Result {i+1}: {result['title']}"):
                                st.write(f"**Source:** {result['source']}")
                                st.write(f"**Relevance Score:** {result['score']:.3f}")
                                st.write("**Content:**")
                                st.write(result['content'])
                                
                                # AI-generated insights
                                if st.button(f"🤖 Get AI Insights for Result {i+1}"):
                                    insights = self.rag_engine.generate_insights(query, result['content'])
                                    st.info("**AI Insights:**")
                                    st.write(insights)
                    else:
                        st.warning("No relevant documents found. Try different keywords.")
        
        # Quick access to common queries
        st.subheader("Quick Access")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("HVAC Maintenance"):
                results = self.rag_engine.search("HVAC maintenance procedures", "Maintenance Manuals")
                self.display_quick_results(results, "HVAC Maintenance")
        
        with col2:
            if st.button("Energy Optimization"):
                results = self.rag_engine.search("energy optimization best practices", "Best Practices")
                self.display_quick_results(results, "Energy Optimization")
        
        with col3:
            if st.button("Emergency Procedures"):
                results = self.rag_engine.search("emergency procedures troubleshooting", "Troubleshooting")
                self.display_quick_results(results, "Emergency Procedures")
    
    def display_quick_results(self, results, title):
        """Display quick search results"""
        if results:
            st.success(f"📋 {title} Results")
            for result in results[:3]:
                st.write(f"• **{result['title']}** - {result['score']:.3f}")
    
    def predictive_maintenance_tab(self):
        """Predictive maintenance dashboard"""
        st.header("🔮 Predictive Maintenance")
        
        # Get predictions
        predictions = self.predictive_maintenance.get_predictions()
        
        if predictions is not None:
            # Equipment failure predictions
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Equipment Failure Predictions")
                fig_predictions = px.bar(predictions, x='equipment', y='failure_probability', 
                                       color='risk_level', title='Failure Risk Assessment')
                st.plotly_chart(fig_predictions, use_container_width=True)
            
            with col2:
                st.subheader("Maintenance Schedule")
                schedule = self.predictive_maintenance.get_maintenance_schedule()
                if schedule is not None:
                    st.dataframe(schedule, use_container_width=True)
            
            # Cost analysis
            st.subheader("Maintenance Cost Analysis")
            cost_data = self.predictive_maintenance.get_cost_analysis()
            
            if cost_data is not None:
                col3, col4 = st.columns(2)
                
                with col3:
                    fig_cost = px.pie(cost_data, values='cost', names='category', 
                                    title='Maintenance Cost Distribution')
                    st.plotly_chart(fig_cost, use_container_width=True)
                
                with col4:
                    # Create a simple bar chart for cost categories instead of line chart
                    fig_trend = px.bar(cost_data, x='category', y='cost', 
                                     title='Maintenance Cost by Category')
                    st.plotly_chart(fig_trend, use_container_width=True)
    
    def energy_optimization_tab(self):
        """Energy optimization dashboard"""
        st.header("⚡ Energy Optimization")
        
        # Get energy data
        energy_data = self.energy_optimization.get_energy_data()
        
        if energy_data is not None:
            # Energy consumption
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Energy Consumption by System")
                fig_consumption = px.bar(energy_data, x='system', y='consumption', 
                                       color='efficiency', title='Energy Consumption Analysis')
                st.plotly_chart(fig_consumption, use_container_width=True)
            
            with col2:
                st.subheader("Efficiency Trends")
                efficiency_data = self.energy_optimization.get_efficiency_trends()
                if efficiency_data is not None:
                    fig_efficiency = px.line(efficiency_data, x='date', y='efficiency', 
                                           color='system', title='System Efficiency Trends')
                    st.plotly_chart(fig_efficiency, use_container_width=True)
            
            # Optimization recommendations
            st.subheader("AI Optimization Recommendations")
            recommendations = self.energy_optimization.get_recommendations()
            
            if recommendations:
                for i, rec in enumerate(recommendations):
                    with st.expander(f"Recommendation {i+1}: {rec['title']}"):
                        st.write(f"**Impact:** {rec['impact']}")
                        st.write(f"**Implementation:** {rec['implementation']}")
                        st.write(f"**Expected Savings:** {rec['savings']}")
    
    def anomaly_detection_tab(self):
        """Anomaly detection dashboard"""
        st.header("🚨 Anomaly Detection")
        
        # Get anomalies
        anomalies = self.anomaly_detection.get_current_anomalies()
        
        if anomalies is not None and not anomalies.empty:
            # Anomaly overview
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Active Anomalies")
                fig_anomalies = px.bar(anomalies, x='severity', y='count', 
                                     color='system', title='Anomaly Distribution by Severity')
                st.plotly_chart(fig_anomalies, use_container_width=True)
            
            with col2:
                st.subheader("Anomaly Timeline")
                timeline_data = self.anomaly_detection.get_anomaly_timeline()
                if timeline_data is not None:
                    fig_timeline = px.scatter(timeline_data, x='timestamp', y='severity', 
                                            color='system', size='impact', 
                                            title='Anomaly Timeline')
                    st.plotly_chart(fig_timeline, use_container_width=True)
            
            # Alert management
            st.subheader("Alert Management")
            for _, anomaly in anomalies.iterrows():
                alert_class = "alert-high" if anomaly['severity'] == 'High' else \
                             "alert-medium" if anomaly['severity'] == 'Medium' else "alert-low"
                
                st.markdown(f"""
                <div class="metric-card {alert_class} alert-card">
                    <h4>🚨 {anomaly['system']} - {anomaly['description']}</h4>
                    <p><strong>Severity:</strong> {anomaly['severity']}</p>
                    <p><strong>Time:</strong> {anomaly['timestamp']}</p>
                    <p><strong>Impact:</strong> {anomaly['impact']}</p>
                    <p><strong>Anomaly Score:</strong> {anomaly['anomaly_score']:.3f}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("✅ No active anomalies detected. All systems are operating normally.")
        
        # Anomaly detection settings
        with st.expander("⚙️ Anomaly Detection Settings"):
            sensitivity = st.slider("Detection Sensitivity", 0.1, 1.0, 0.5)
            threshold = st.slider("Alert Threshold", 0.1, 1.0, 0.7)
            
            if st.button("Update Settings"):
                self.anomaly_detection.update_settings(sensitivity, threshold)
                st.success("Settings updated successfully!")

def main():
    """Main application entry point"""
    try:
        # Initialize the application
        app = SmartBuildingRAG()
        
        # Run the main dashboard
        app.main_dashboard()
        
        # Auto-refresh every 30 seconds
        time.sleep(30)
        st.rerun()
        
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.info("Please check your configuration and try again.")

if __name__ == "__main__":
    main()

