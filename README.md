# **Customer Sentiment Analysis Using BERT**

A real-time, entity-specific sentiment analysis system developed using advanced data mining and natural language processing (NLP) techniques. The system is designed to accurately identify entities mentioned in social media posts, particularly tweets, and classify the sentiment expressed towards each entity. By fine-tuning lightweight language models like **DistilBERT** or **TinyBERT** on a cleaned and preprocessed Twitter dataset, this solution aims to deliver high accuracy in **aspect-based sentiment classification**. Businesses can leverage this system to gain granular insights into public perception regarding their products, services, or brands.

## **Phase 1: Initial Model Training**

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
2. **Aspect-based Sentiment Classification**: Provide businesses with actionable data on public perception related to specific products, services, or brands. It can also help busineesses know what people are thinkng about their roducts before launching them and after launching them. The system could be used to collect all critical comments and see where improvments can be made.
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


---

## Future Work

### Model Deployment
- Develop a dashboard to showcase the model, allowing users to inquire about different entities and check the sentiment on those specific entities.

### AWS Integration
- Deploy the model on an AWS EC2 instance to ensure scalability and accessibility.

### Continuous Improvement
- Expand the model by fine-tuning on more comprehensive datasets for entity recognition and sentiment classification.

---


## Next Steps: Reddit Sentiment Analysis Dashboard on AWS

## Project Overview

### Purpose of the Project
This project combines machine learning expertise with AWS cloud deployment skills to create a real-time sentiment analysis system for Reddit data. The application will:
- Analyze sentiments for entities from Reddit posts on a monthly basis.
- Display sentiment distribution for different entities (e.g., companies, products).
- Allow users to input text and receive immediate sentiment predictions.

### Final Product
The final application will be a web-based dashboard hosted on AWS, featuring:
- **Sentiment Visualization Dashboard**: Showcasing sentiment analysis of Reddit posts, highlighting entities with the highest and lowest sentiment scores.
- **Real-Time Sentiment Analysis**: Enabling users to input text to test the model and receive instant sentiment predictions.
- **Secure and Scalable Infrastructure**: Utilizing AWS services to ensure the application is secure and can scale with demand.

---

## Project Requirements

### Prerequisites
- **AWS Account**: With permissions to create VPCs, EC2 instances, Application Load Balancers, Route 53 configurations, IAM roles, and S3 buckets.
- **Data Collection Skills**: Ability to use Reddit's API for data collection or access to pre-collected Reddit data.
- **Machine Learning Skills**: Experience in model training, specifically with BERT or other transformer-based models for sentiment analysis.

### AWS Services Required
- **VPC (Virtual Private Cloud)**: For network isolation and security.
- **EC2 Instance**: To host the application and serve the ML model.
- **Application Load Balancer (ALB)**: To securely route and balance traffic.
- **S3 Buckets**: For storing data and assets.
- **Route 53**: For domain name management.
- **CloudWatch and VPC Flow Logs**: For monitoring and logging.
- **IAM**: For secure access control and role management.

---

## Step-by-Step Guide

### AWS Infrastructure Setup

1. **Set Up a VPC (Virtual Private Cloud)**
    - **Create a New VPC**:
        - Name: `RedditSentimentVPC`
        - CIDR block: `10.0.0.0/16`
        - Enable DNS hostnames.
    - **Create Subnets**:
        - Public Subnet: For the ALB (CIDR block: `10.0.1.0/24`).
        - Private Subnet: For EC2 instances (CIDR block: `10.0.2.0/24`).
    - **Internet Gateway**: Create and attach an Internet Gateway to the VPC.
    - **Route Tables**: 
        - Public Route Table: Associate with the public subnet and add a route to the Internet Gateway (`0.0.0.0/0`).
        - Private Route Table: Associate with the private subnet.

2. **Launch an EC2 Instance**
    - Select an AMI: Use `Ubuntu Server 20.04 LTS`.
    - Instance Type: `t3.medium`.
    - Security Group: Allow inbound traffic only from the ALB.
    - IAM Role: Attach an IAM role with necessary permissions (e.g., S3 read access).

3. **Configure Application Load Balancer (ALB)**
    - Place it in the public subnet.
    - Allow inbound traffic on ports `80` (HTTP) and `443` (HTTPS).
    - Create a target group and register the EC2 instance.

### Model Deployment and API Development

4. **Set Up the EC2 Instance**
    - Install Dependencies: Update packages and set up a Python environment.
    - Load the Fine-Tuned Model.
    - Develop the Flask API.

5. **Configure the Web Server**
    - Application Server: Use Gunicorn to serve the Flask app.

### Secure the Application and Domain Configuration

6. **Set Up HTTPS**
    - SSL Certificate: Request a certificate via AWS Certificate Manager.
    - Attach Certificate to ALB.

7. **Configure Domain with Route 53**
    - Hosted Zone: Create a hosted zone in Route 53.

### Building the Dashboard and Data Visualization

8. **Develop the Data Collection Pipeline**
    - **Reddit API Integration**.
    - Store data in an S3 bucket.

9. **Build the Dashboard**
    - Backend: Use Flask.
    - Frontend: Implement visualizations with libraries like Plotly or D3.js.

### Monitoring, Logging, and Testing

10. **Set Up Monitoring and Logging**
    - CloudWatch: Monitor EC2 instance performance metrics.

11. **Test and Validate the Application**
    - Functional Testing.
    - Security Testing.

