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

## **Phase 2: Fine-Tuning on Entity-Specific Sentiment**

In the second phase, the model will be fine-tuned on an **entity-specific sentiment analysis dataset**. This new dataset is more balanced and diverse, which will help the model generalize better in recognizing and classifying sentiments expressed towards specific entities (such as products, services, or brands) in text.

### **Key Goals:**
- Improve **aspect-based sentiment analysis** to provide more accurate insights into sentiments about specific entities.
- Achieve better generalization by training on a balanced dataset, ensuring the model is robust across various entities and sentiments.

---

## **Project Objectives**
- **Real-time Sentiment Analysis:** Enable real-time insights by identifying entities in social media posts and classifying the sentiment expressed toward them.
- **Aspect-based Sentiment Classification:** Provide businesses with actionable data on public perception related to specific products, services, or brands.
- **Model Generalization:** Ensure the model generalizes well across datasets by fine-tuning on balanced, entity-specific sentiment data.

---

## **Future Work**
- **Model Expansion:** Continuously fine-tune the model on more comprehensive datasets for entity recognition and sentiment classification.
- **Dashboard:** Design a dashboard that showcases our model and people can use it to ask about different entities and check the sentiment on that specific entity.




