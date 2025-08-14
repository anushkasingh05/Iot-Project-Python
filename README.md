# IoT Sensor Data RAG for Smart Buildings

A comprehensive RAG (Retrieval-Augmented Generation) system that processes IoT sensor data, maintenance manuals, and building specifications to provide predictive maintenance insights and operational optimization for smart buildings.

## 🏗️ Features

### Core Functionality
- **Real-time IoT Sensor Data Processing**: Ingest and process sensor data from HVAC, lighting, security, and environmental systems
- **Maintenance Manual Integration**: RAG-powered access to equipment manuals and maintenance procedures
- **Predictive Maintenance**: ML-based failure prediction and maintenance scheduling
- **Operational Optimization**: AI-driven recommendations for energy efficiency and system optimization
- **Anomaly Detection**: Real-time monitoring and alerting for system anomalies

### Technical Components
- **Vector Database**: ChromaDB for efficient document retrieval
- **Embedding Models**: Sentence Transformers for semantic search
- **ML Models**: Scikit-learn for predictive maintenance
- **Real-time Processing**: Streamlit for interactive dashboard
- **Data Visualization**: Plotly for dynamic charts and graphs

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd iot-rag-smart-buildings
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Access the application**
   - Open your browser and go to `http://localhost:8501`

## 📊 System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   IoT Sensors   │───▶│  Data Ingestion │───▶│  Real-time      │
│                 │    │                 │    │  Processing     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Maintenance   │───▶│  Vector DB      │───▶│  RAG Engine     │
│   Manuals       │    │  (ChromaDB)     │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Building      │───▶│  ML Models      │───▶│  Streamlit      │
│   Specs         │    │  (Predictive)   │    │  Dashboard      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🔧 Configuration

### Environment Variables
Create a `.env` file with the following variables:
```
OPENAI_API_KEY=your_openai_api_key
CHROMA_DB_PATH=./chroma_db
SENSOR_DATA_PATH=./data/sensor_data
MANUALS_PATH=./data/manuals
```

### Data Structure
```
data/
├── sensor_data/
│   ├── hvac_sensors.csv
│   ├── lighting_sensors.csv
│   └── environmental_sensors.csv
├── manuals/
│   ├── hvac_manual.pdf
│   ├── lighting_manual.pdf
│   └── security_manual.pdf
└── building_specs/
    ├── floor_plans.pdf
    └── system_specifications.json
```

## 📈 Usage

### 1. Dashboard Overview
- **Real-time Sensor Monitoring**: View live sensor data from all building systems
- **System Health Status**: Monitor equipment health and performance metrics
- **Alert Management**: View and manage system alerts and notifications

### 2. Predictive Maintenance
- **Failure Prediction**: AI-powered predictions for equipment failures
- **Maintenance Scheduling**: Optimized maintenance schedules based on predictions
- **Cost Analysis**: Maintenance cost optimization recommendations

### 3. RAG-powered Insights
- **Manual Search**: Search through maintenance manuals and procedures
- **Troubleshooting**: Get AI-powered troubleshooting recommendations
- **Best Practices**: Access building operation best practices

### 4. Energy Optimization
- **Usage Analytics**: Detailed energy consumption analysis
- **Optimization Recommendations**: AI-driven energy saving suggestions
- **Performance Tracking**: Monitor optimization effectiveness

## 🧪 Evaluation Metrics

The system includes evaluation metrics for:
- **Retrieval Accuracy**: RAGAS-based evaluation of document retrieval
- **Prediction Accuracy**: ML model performance metrics
- **Response Latency**: System response time measurements
- **User Satisfaction**: Interactive feedback collection

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI for GPT models
- HuggingFace for sentence transformers
- ChromaDB for vector database
- Streamlit for the web interface
- Scikit-learn for machine learning capabilities

## 📞 Support

For support and questions:
- Create an issue in the GitHub repository
- Contact: [your-email@domain.com]

---

**Built with ❤️ for Smart Building Management**

