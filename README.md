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
  - Training models on images.

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

# Instructions for Setup

## Using Ibex

#### 1.   Connect to Ibex

First, establish a connection to Ibex using your KAUST username:

```bash
$ ssh 'kaust_username'@glogin.ibex.kaust.edu.sa
```
Replace 'kaust_username' with your actual KAUST username.

#### 2.  Cloning the Repository

Begin by cloning this repository using the following command:

```bash
$ git clone https://github.com/A-Halimi/Introduction-to-AI-workshop-series.git
```
Ensure you replace 'repo' with the actual link to the GitHub repository.

#### 3. Requesting Resources and Running Jupyter Notebook

After cloning the repository, request the necessary resources using the command below:

```bash
$ srun --gpus=1 --time=03:00:00 --resv-ports=1 --pty /bin/bash -l run_ai_env_jupyter.sh
```
Once the resources are allocated, the Jupyter notebook environment should be activated and ready for use.

## Using Classhub Binder

For those who prefer to work on ClassHub Binder:

- Make sure you have a GitHub account. If you don't, [create one here](https://github.com/).
- Sign into your GitHub account then add your Github ID to this [Link](https://assembly.kaust.edu.sa/form/c0944092-f221-4a23-b471-0c9dd6e4d879) 
- Navigate to [ClassHub Binder](https://classhub.kaust.edu.sa/course/ai-ws/) using the provided link.
- For questions [Etherpad](https://pad.carpentries.org/2024-09-26-kaust-vislab) 
- Follow the on-screen instructions to connect your GitHub account and access the workshop materials on ClassHub Binder.




