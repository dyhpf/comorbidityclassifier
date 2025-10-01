# Clinical Comorbidity Classification Framework

A comprehensive LLM-based system for extracting comorbidity information from clinical texts, designed for neurology and sleep medicine applications.

## 📁 Files

- **`gccomorbidity.py`** - Single-level classifier with topic management
- **`multiagentgc.py`** - Multi-agent hierarchical classifier using LangGraph

## 🏥 Overview

This framework addresses the challenge of accurately identifying complex comorbidities from diverse medical documentation:

> *"Patients presenting to neurology clinics commonly have complex histories of comorbidities and partially documented health trajectories, making it essential to reliably extract comorbidity information from historical records."*

## 🎯 Key Features

### Single-Level Classifier (`gccomorbidity.py`)
- **Unified classification** across multiple comorbidity topics
- **Topic management system** for organizing classification tasks
- **Multiple inference backends**: Transformers, OpenAI, DeepInfra
- **Probability-based selection** for reliable predictions
- **Prompt optimization** with LLM feedback
- **CSV batch processing** for large datasets

### Multi-Agent Classifier (`multiagentgc.py`) 
- **Hierarchical architecture** using LangGraph ReAct agents
- **Specialized agents** for sleep disorder comorbidity detection:
  - Insomnia (chronic/short-term)
  - Sleep Apnea (mild/moderate/severe) 
  - Restless Legs Syndrome
  - Narcolepsy
  - Circadian Rhythm Disorders
- **Contradiction resolution** between parent-child classifications
- **Early judgment filtering** for efficiency
- **Context forwarding** between hierarchy levels

## 🚀 Quick Start

### Installation
```bash
pip install torch transformers langchain langgraph
