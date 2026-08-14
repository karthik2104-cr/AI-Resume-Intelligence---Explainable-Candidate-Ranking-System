# Controlled matching pairs for semantic vs TF-IDF evaluation
# Each entry is a tuple: (resume_text, jd_text, label)
# label: 'high', 'medium', 'low'

matching_pairs = [
    ("Built machine learning models in Python using scikit-learn to predict customer churn.",
     "Develop predictive ML systems in Python with sklearn for churn prediction.",
     "high"),

    ("Experienced Python developer skilled in Django, REST APIs, and PostgreSQL.",
     "Looking for Python developer with experience in Django and PostgreSQL.",
     "high"),

    ("Frontend developer with React and TypeScript experience.",
     "Full-stack engineer required, experience with React is a plus.",
     "medium"),

    ("Digital marketer experienced in SEO, SEM, and content strategy.",
     "Machine learning engineer with experience in Python, PyTorch, and model deployment.",
     "low"),

    ("Data scientist using Python and XGBoost for large-scale modeling.",
     "Senior ML engineer needed, experience with PyTorch and deep learning preferred.",
     "medium"),

    ("Implemented fraud detection pipelines and anomaly detection models.",
     "Build systems to detect fraud and anomalies in transaction data.",
     "high"),

    # additional pairs to expand evaluation
    ("Built end-to-end ML pipelines and deployed models to AWS.",
     "Looking for ML engineer to deploy models and productionize ML on AWS.",
     "high"),

    ("Worked on natural language processing and transformer-based models.",
     "Experience building NLP systems using transformers and huggingface.",
     "high"),

    ("Developed mobile apps using Kotlin and Android Studio.",
     "Seeking Android developer familiar with Kotlin and mobile app lifecycle.",
     "high"),

    ("Researcher focused on reinforcement learning and multi-agent systems.",
     "Hiring research scientist in reinforcement learning and multi-agent RL.",
     "high"),

    ("Managed social media campaigns and grew engagement organically.",
     "Senior marketing manager with experience in paid acquisition and analytics.",
     "low"),

    ("Backend developer experienced in Java, Spring, and microservices.",
     "Frontend developer role focusing on React and UI engineering.",
     "low"),
]
