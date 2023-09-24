# Introduction to AI Workshop Series

Welcome to our comprehensive series on getting started with Artificial Intelligence (AI). This repository is dedicated to providing materials, notebooks, slides, and code samples for our three primary workshops:

## 1. Introduction to Machine Learning
- **Overview**:
  - What is Machine Learning?
  - Types of Machine Learning
  - Steps in a full machine learning projects
  - Building a simple ML model: Steps and Best Practices.
- **Labs**:
  - Getting started with popular ML libraries like Scikit-Learn.
  - Hands-on pipeline on regression and classification problems.

## 2. Introduction to Deep Learning
- **Overview**:
  - Deep Learning vs. Traditional Machine Learning.
  - Neural Networks and their magic: How do they work?
  - Popular architectures: ANN, CNN..
- **Labs**:
  - Building a Neural Network using TensorFlow/Keras.
  - Training models on image and text datasets.

## 3. Introduction to Data Visualization in Data Science
- **Overview**:
  - Importance of visualization in the Data Science pipeline.
  - Types of visualizations: From bar charts to complex visual analytics.
  - Tools and libraries: Matplotlib, Seaborn, and beyond.
- **Labs**:
  - Hands-on demo on creating insightful visualizations.
  - Interactive plots and dashboards.

### Prerequisites:
- Basic knowledge of Python programming.
- A curious mind ready to dive into the world of AI!

## Instructions for Setup

#### 1.  Cloning the Repository

Begin by cloning this repository using the following command:

```bash
$ git clone 'repo'
```
Ensure you replace 'repo' with the actual link to the GitHub repository.

#### 2. Requesting Resources and Running Jupyter Notebook

After cloning the repository, request the necessary resources using the command below:

```bash
$ srun --gpus-per-node=1 --time=03:00:00 --nodes=1 --mem=64G --constraint=a100 --reservation=A100 --resv-ports=1 --pty bash -l run_ai_env_jupyter.sh
```
Once the resources are allocated, the Jupyter notebook environment should be activated and ready for use.




