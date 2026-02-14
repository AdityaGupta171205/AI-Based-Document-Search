def generate_summary(func):
    result = func()
    return result["answer"]


def generate_study_notes(func):
    result = func()
    return result["answer"]


def generate_quiz(func):
    result = func()
    return result["answer"]


def extract_topics(func):
    result = func()
    return result["answer"]
