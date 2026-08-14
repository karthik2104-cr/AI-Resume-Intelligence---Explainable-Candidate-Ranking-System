"""Sample resume texts for parser tests."""

SAMPLE_RESUME = """
John Smith
john.smith@email.com | +1-555-123-4567

SUMMARY
Experienced Python developer with 5 years in backend systems and APIs.

SKILLS
Python, Django, PostgreSQL, REST APIs, Machine Learning

EXPERIENCE
Senior Python Developer at TechCorp
Jan 2020 - Present
- Built REST APIs with Django
- Led migration to PostgreSQL

Junior Developer at StartupXYZ
Jun 2017 - Dec 2019
- Developed Flask microservices

EDUCATION
B.Tech Computer Science
State University
2015 - 2019

PROJECTS
Resume Parser Tool
Built NLP pipeline using Python
Technologies: Python, spaCy, FastAPI

CERTIFICATIONS
AWS Certified Developer
Amazon Web Services
2022
""".strip()

MINIMAL_RESUME = """
Jane Doe
jane@example.com

SKILLS
Java, Spring Boot
""".strip()

UNSTRUCTURED_RESUME = """
This is a resume without clear section headings.
It mentions Python and SQL but has no labeled sections.
Contact: dev@example.com
""".strip()

MARKDOWN_HEADINGS_RESUME = """
Alex Chen
alex.chen@mail.com

## Summary
Data scientist focused on NLP.

## Skills
Python | SQL | scikit-learn

## Experience
Data Scientist at Analytics Inc
2019 - Present
""".strip()

INTERNSHIP_RESUME = """
Sam Lee
sam@example.com

EXPERIENCE
Software Engineering Intern at BigCo
Jun 2023 - Aug 2023
- Built internal tools with Python
""".strip()
