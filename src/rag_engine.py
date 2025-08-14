import chromadb
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import os
import json
from datetime import datetime
import openai
from dotenv import load_dotenv

load_dotenv()

class RAGEngine:
    """RAG engine for processing maintenance manuals and building specifications"""
    
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        self.collection_name = "smart_building_docs"
        self.collection = self._get_or_create_collection()
        self.openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Initialize with sample data if collection is empty
        if self.collection.count() == 0:
            self._initialize_sample_data()
    
    def _get_or_create_collection(self):
        """Get existing collection or create new one"""
        try:
            collection = self.chroma_client.get_collection(name=self.collection_name)
        except:
            collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"description": "Smart building maintenance and specification documents"}
            )
        return collection
    
    def _initialize_sample_data(self):
        """Initialize the vector database with sample documents"""
        sample_documents = self._create_sample_documents()
        
        # Prepare data for ChromaDB
        documents = []
        metadatas = []
        ids = []
        
        for i, doc in enumerate(sample_documents):
            documents.append(doc['content'])
            metadatas.append({
                'title': doc['title'],
                'source': doc['source'],
                'category': doc['category'],
                'timestamp': doc['timestamp']
            })
            ids.append(f"doc_{i}")
        
        # Add to collection
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
    
    def _create_sample_documents(self) -> List[Dict]:
        """Create sample documents for the knowledge base"""
        documents = [
            {
                'title': 'HVAC System Maintenance Manual',
                'content': '''
                HVAC System Maintenance Procedures:
                1. Regular Filter Replacement: Replace air filters every 3 months or as needed
                2. Coil Cleaning: Clean evaporator and condenser coils quarterly
                3. Duct Inspection: Inspect ductwork annually for leaks and damage
                4. Thermostat Calibration: Calibrate thermostats every 6 months
                5. Refrigerant Check: Monitor refrigerant levels and top up if necessary
                6. Motor Lubrication: Lubricate motors and bearings as per manufacturer specs
                7. Electrical Inspection: Check electrical connections and components monthly
                8. Performance Testing: Conduct efficiency tests quarterly
                
                Troubleshooting Common Issues:
                - Low airflow: Check filters and duct obstructions
                - Uneven cooling: Verify thermostat settings and duct balance
                - High energy consumption: Clean coils and check refrigerant levels
                - Unusual noises: Inspect motors and bearings
                ''',
                'source': 'HVAC_Manual_v2.1',
                'category': 'Maintenance Manuals',
                'timestamp': datetime.now().isoformat()
            },
            {
                'title': 'Lighting System Operation Guide',
                'content': '''
                Lighting System Best Practices:
                1. LED Maintenance: Clean LED fixtures every 6 months
                2. Sensor Calibration: Calibrate motion sensors monthly
                3. Dimming System Check: Test dimming functionality weekly
                4. Emergency Lighting: Test emergency lights monthly
                5. Energy Optimization: Use daylight harvesting and occupancy sensors
                
                Energy Saving Strategies:
                - Implement automatic dimming based on natural light
                - Use occupancy sensors in low-traffic areas
                - Schedule lighting based on building occupancy
                - Regular maintenance of sensors and controls
                - Monitor energy consumption patterns
                ''',
                'source': 'Lighting_Guide_v1.5',
                'category': 'Best Practices',
                'timestamp': datetime.now().isoformat()
            },
            {
                'title': 'Building Security System Manual',
                'content': '''
                Security System Maintenance:
                1. Camera Cleaning: Clean security cameras monthly
                2. Access Control Testing: Test card readers and biometric systems weekly
                3. Alarm System Check: Test fire and security alarms monthly
                4. Backup Power: Verify UPS and backup systems quarterly
                5. Software Updates: Update security software as needed
                
                Emergency Procedures:
                - Fire evacuation: Follow designated evacuation routes
                - Security breach: Contact security immediately
                - Power failure: Activate backup systems
                - System failure: Use manual override procedures
                ''',
                'source': 'Security_Manual_v3.0',
                'category': 'Emergency Procedures',
                'timestamp': datetime.now().isoformat()
            },
            {
                'title': 'Energy Optimization Guidelines',
                'content': '''
                Energy Efficiency Best Practices:
                1. HVAC Optimization: Set optimal temperature ranges (20-24°C)
                2. Lighting Control: Use smart lighting systems with sensors
                3. Building Automation: Implement BMS for centralized control
                4. Regular Audits: Conduct energy audits quarterly
                5. Equipment Maintenance: Maintain equipment at peak efficiency
                
                Cost Reduction Strategies:
                - Peak demand management
                - Load balancing across systems
                - Renewable energy integration
                - Smart metering and monitoring
                - Employee training on energy conservation
                ''',
                'source': 'Energy_Guide_v2.0',
                'category': 'Best Practices',
                'timestamp': datetime.now().isoformat()
            },
            {
                'title': 'Building Specifications',
                'content': '''
                Building System Specifications:
                - HVAC: 3 rooftop units, 2 air handlers, 15 VAV boxes
                - Lighting: LED fixtures with smart controls, 500+ fixtures
                - Security: 25 IP cameras, access control at all entrances
                - Electrical: 400A main service, backup generator
                - Plumbing: 50 fixtures, water conservation systems
                
                Maintenance Requirements:
                - Annual HVAC inspection and cleaning
                - Quarterly lighting system maintenance
                - Monthly security system testing
                - Weekly building walkthrough
                - Daily system monitoring
                ''',
                'source': 'Building_Specs_v1.0',
                'category': 'Building Specs',
                'timestamp': datetime.now().isoformat()
            },
            {
                'title': 'Predictive Maintenance Procedures',
                'content': '''
                Predictive Maintenance Implementation:
                1. Data Collection: Monitor key performance indicators
                2. Trend Analysis: Analyze historical data for patterns
                3. Failure Prediction: Use ML models to predict failures
                4. Maintenance Scheduling: Optimize maintenance schedules
                5. Cost Analysis: Track maintenance costs and savings
                
                Key Metrics to Monitor:
                - Equipment vibration levels
                - Temperature and pressure readings
                - Energy consumption patterns
                - Operating hours and cycles
                - Performance degradation trends
                ''',
                'source': 'Predictive_Maintenance_v1.2',
                'category': 'Maintenance Manuals',
                'timestamp': datetime.now().isoformat()
            }
        ]
        return documents
    
    def search(self, query: str, search_type: str = "All", top_k: int = 5) -> List[Dict]:
        """Search for relevant documents"""
        try:
            # Prepare query
            if search_type != "All":
                # Add category filter to query
                query = f"{query} {search_type}"
            
            # Search in ChromaDB
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                include=['metadatas', 'documents', 'distances']
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results['ids'][0])):
                result = {
                    'title': results['metadatas'][0][i]['title'],
                    'content': results['documents'][0][i],
                    'source': results['metadatas'][0][i]['source'],
                    'category': results['metadatas'][0][i]['category'],
                    'score': 1 - results['distances'][0][i],  # Convert distance to similarity score
                    'timestamp': results['metadatas'][0][i]['timestamp']
                }
                formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            print(f"Error in search: {e}")
            return []
    
    def generate_insights(self, query: str, context: str) -> str:
        """Generate AI insights based on query and context"""
        try:
            prompt = f"""
            Based on the following context from building maintenance documents, provide insights for the query: "{query}"
            
            Context:
            {context}
            
            Please provide:
            1. Key insights and recommendations
            2. Action items if applicable
            3. Best practices to follow
            4. Potential risks or considerations
            
            Format your response in a clear, actionable manner.
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert building maintenance consultant with deep knowledge of HVAC, lighting, security, and energy systems."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Unable to generate insights: {str(e)}"
    
    def add_document(self, title: str, content: str, source: str, category: str):
        """Add a new document to the knowledge base"""
        try:
            # Generate document ID
            doc_id = f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Add to collection
            self.collection.add(
                documents=[content],
                metadatas=[{
                    'title': title,
                    'source': source,
                    'category': category,
                    'timestamp': datetime.now().isoformat()
                }],
                ids=[doc_id]
            )
            
            return True
            
        except Exception as e:
            print(f"Error adding document: {e}")
            return False
    
    def get_document_categories(self) -> List[str]:
        """Get all available document categories"""
        try:
            # Get all documents and extract unique categories
            results = self.collection.get()
            categories = set()
            
            for metadata in results['metadatas']:
                categories.add(metadata['category'])
            
            return list(categories)
            
        except Exception as e:
            print(f"Error getting categories: {e}")
            return []
    
    def get_statistics(self) -> Dict:
        """Get knowledge base statistics"""
        try:
            total_docs = self.collection.count()
            categories = self.get_document_categories()
            
            stats = {
                'total_documents': total_docs,
                'categories': categories,
                'category_count': len(categories),
                'last_updated': datetime.now().isoformat()
            }
            
            return stats
            
        except Exception as e:
            print(f"Error getting statistics: {e}")
            return {}
    
    def semantic_search(self, query: str, filters: Dict = None) -> List[Dict]:
        """Advanced semantic search with filters"""
        try:
            # Build query with filters
            where_clause = {}
            if filters:
                for key, value in filters.items():
                    where_clause[key] = value
            
            # Perform search
            results = self.collection.query(
                query_texts=[query],
                n_results=10,
                where=where_clause if where_clause else None,
                include=['metadatas', 'documents', 'distances']
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results['ids'][0])):
                result = {
                    'title': results['metadatas'][0][i]['title'],
                    'content': results['documents'][0][i],
                    'source': results['metadatas'][0][i]['source'],
                    'category': results['metadatas'][0][i]['category'],
                    'score': 1 - results['distances'][0][i],
                    'timestamp': results['metadatas'][0][i]['timestamp']
                }
                formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            print(f"Error in semantic search: {e}")
            return []
    
    def update_document(self, doc_id: str, title: str, content: str, source: str, category: str):
        """Update an existing document"""
        try:
            # Delete existing document
            self.collection.delete(ids=[doc_id])
            
            # Add updated document
            self.collection.add(
                documents=[content],
                metadatas=[{
                    'title': title,
                    'source': source,
                    'category': category,
                    'timestamp': datetime.now().isoformat()
                }],
                ids=[doc_id]
            )
            
            return True
            
        except Exception as e:
            print(f"Error updating document: {e}")
            return False
    
    def delete_document(self, doc_id: str):
        """Delete a document from the knowledge base"""
        try:
            self.collection.delete(ids=[doc_id])
            return True
        except Exception as e:
            print(f"Error deleting document: {e}")
            return False
