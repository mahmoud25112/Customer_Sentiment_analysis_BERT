# **Customer Sentiment Analysis Using BERT**

A real-time, entity-specific sentiment analysis system developed using advanced data mining and natural language processing (NLP) techniques. The system is designed to accurately identify entities mentioned in social media posts, particularly tweets, and classify the sentiment expressed towards each entity. By fine-tuning lightweight language models like **DistilBERT** or **TinyBERT** on a cleaned and preprocessed Twitter dataset, this solution aims to deliver high accuracy in **aspect-based sentiment classification**. Businesses can leverage this system to gain granular insights into public perception regarding their products, services, or brands.

## **Phase 1: Initial Model Training**

### **Logistic Regression Model and Findings**
1. **Data Preparation:**
   - **Exploratory Data Analysis (EDA):**
     - Inspected and relabeled sentiments: **1 (positive)** and **0 (negative)**.
     - Checked for outliers and duplicates.
   - **Data Cleaning:**
     - Converted text to lowercase.
     - Replaced URLs and usernames with placeholders (e.g., `<URL>` and `<USER>`).
     - Removed special characters and stop words.
     - Applied lemmatization for tokenization.
   - **Feature Extraction:**
     - Used **TF-IDF vectorization** with 3000 features, determined optimal after comparison with 7000 features.

### **BERT Model**

The initial model training is documented in the `Sentiment_analysis.ipynb` notebook.

### **Dataset and Label Augmentation**
- **Original Labels:** The dataset initially contained two sentiment labels: **positive** and **negative**.
- **Aggregating Neutral Class:** To simulate a real-world scenario, we collected neutral class records from different datasets and aggregated them into our dataset.
- **Class Imbalance Handling:** The loss function was modified to account for the imbalance in class distribution:
    - **Positive Class:** 800,000 samples
    - **Negative Class:** 800,000 samples
    - **Neutral Class:** 50,000 samples

### **Model Performance**
- **Neutral Class:** Achieved over **75% accuracy** using only 30,000 data points.
- **Positive and Negative Classes:** Both classes reached approximately **88% accuracy** using 800,000 samples each.

This phase demonstrated that the model can generalize well, especially with the weighted loss function improving performance for the minority (neutral) class.

---


## Phase 2: Fine-Tuning on Entity-Specific Sentiment

In Phase 2, the model was fine-tuned on an entity-specific sentiment analysis dataset, focusing only on positive and negative sentiments. The neutral class was removed to improve the model’s focus on entity-specific sentiment recognition, targeting entities such as products, services, or brands.


### Key Goals
- **Improve aspect-based sentiment analysis** to provide more accurate insights into sentiments about specific entities.
- **Achieve better generalization** by training on a balanced dataset, ensuring the model is robust across various entities and sentiments.

### Project Objectives
1. **Real-time Sentiment Analysis**: Enable real-time insights by identifying entities in social media posts and classifying the sentiment expressed toward them.
2. **Aspect-based Sentiment Classification**: Provide businesses with actionable data on public perception related to specific products, services, or brands. It can also help busineesses know what people are thinkng about their products before launching them and after launching them. The system could be used to collect all critical comments and see where improvments can be made.
3. **Model Generalization**: Ensure the model generalizes well across datasets by fine-tuning on balanced, entity-specific sentiment dataset.
4. **Build a dashboard**: The model will be called for inference over an ec2 instance to give sentiment on some products every month specifying which products have the highest and lowest sentiment in differnet domains.

### Model Performance

#### Metrics Achieved on Unseen test Set:
| Metric       | Precision | Recall  | F1-Score | Support |
|--------------|-----------|---------|----------|---------|
| **Negative** | 0.9461    | 0.8446  | 0.8925   | 457     |
| **Neutral**  | 0.9014    | 0.9281  | 0.9146   | 473     |
| **Positive** | 0.8752    | 0.9404  | 0.9067   | 470     |
| **Accuracy** |           |         | 0.9050   | 1400    |
| **Macro Avg**| 0.9076    | 0.9044  | 0.9046   | 1400    |
| **Weighted Avg** | 0.9072 | 0.9050 | 0.9047   | 1400    |

This phase demonstrated that the model can generalize well, given that it performed slightly lower on the  test dataset than it did on the training and validation datasets.

#### Confusion Matrix:

![image](https://github.com/user-attachments/assets/e6275342-b2bb-4582-b334-529d0f9a0283)



# AWS ETL Pipeline for Reddit Data Collection and Sentiment Analysis

This project is an AWS-based ETL pipeline designed to collect, process, and analyze Reddit data related to specific brands. The pipeline uses various AWS services, including Lambda, S3, EventBridge, ECS, and Batch, to provide an automated data extraction, transformation, and loading (ETL) process.

---

## Overview

The pipeline automates:
- **Data Extraction**: Fetches posts from Reddit based on specified search criteria.
- **Data Transformation**: Performs sentiment analysis and entity recognition on collected data.
- **Data Storage**: Stores raw and processed data in S3 buckets.
- **Scheduled Execution**: Executes the data collection and processing jobs weekly using EventBridge.

---

## Prerequisites

- **AWS Account** with necessary permissions
- **AWS CLI** configured locally
- **Docker** installed for containerization
- **Reddit API Credentials**: Client ID, Secret, and User Agent

---

## Steps for AWS Deployment

### Step 1: S3 Bucket Setup

1. Create two S3 buckets:
   - `raw-data-bucket`: For storing raw data collected from Reddit.
   - `processed-data-bucket`: For storing processed data after transformation.

### Step 2: Lambda Function for Data Collection

1. **Initialize the Lambda Environment**:

2. **Package and Deploy**:

3. **Set Environment Variables**:
Configure the Lambda function in the AWS console under **Configuration** > **Environment variables**.

4. **Update Execution Time**:
Set the Lambda timeout to 30 seconds.

### Step 3: Schedule the Lambda with EventBridge

1. Go to **EventBridge** > **Rules** and create a rule to trigger the Lambda weekly:
- **Pattern**: `cron(0 2 ? * SUN *)` to run every Sunday.

### Step 4: Local Testing and Docker Setup

1. **Set Up Environment**:

2. **Install Dependencies**:

3. **Dockerize**:
- Create a `Dockerfile` and build your image:
  ```
  docker build -t sentiment-app .
  ```

### Step 5: Create and Deploy a Batch Job

1. **Login to AWS ECR**:

2. **Push Docker Image to ECR**.
3. **Create Batch Execution Role**:
- Open IAM, create a new role for ECS tasks, and attach the `AmazonECSTaskExecutionRolePolicy` and S3 access policies as needed.

4. **Configure AWS Batch**:
- Set up a job definition using the uploaded Docker image.
- Configure environment variables, memory, and CPU settings.
- Create a compute environment with Fargate as the backend.

5. **Submit Batch Job**:
- Go to AWS Batch, set up a job queue, and submit the job.

### Step 6: Automate Batch Execution with EventBridge

1. **Schedule Batch Job Execution**:
- Configure EventBridge with the following cron expression to trigger every Thursday:
  ```
  cron(0 0 ? * 5 *)
  ```

### Summary

This ETL pipeline:
- Extracts Reddit posts weekly using AWS Lambda and stores them in S3.
- Transforms data using sentiment and entity recognition in AWS Batch.
- Loads transformed data back to S3 for further analysis or visualization.

--- 

## Next Steps

For data visualization, consider integrating with an analytics dashboard like **AWS QuickSight** or exporting to a **BI tool** for enhanced insights.

--- 

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
