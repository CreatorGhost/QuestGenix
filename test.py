import json

data = {
    'qa_pairs': [
        {'question': 'What is Python?', 'answer': 'python is a programming language and it is also a interpreted based programming language and its compiler is basically written in c++'},
        {'question': 'What are the key features of Python?', 'answer': 'veesality'},
        {'question': 'What is the difference between Python 2 and Python 3?', 'answer': ''},
        {'question': 'How do you comment in Python?', 'answer': 'we can comment in python using # for single line comm6'},
        {'question': 'What are the different data types in Python?', 'answer': 'list tupple dict set'},
        {'question': 'How do you create a function in Python?', 'answer': ''},
        {'question': "What is the use of the 'if' statement in Python?", 'answer': 'its used for conditionally cheking based on the conditions provided '},
        {'question': 'How do you handle exceptions in Python?', 'answer': 'using try and catch '},
        {'question': 'What are modules in Python?', 'answer': ''},
        {'question': 'What is the difference between a list and a tuple in Python?', 'answer': ''}
    ]
}

json_string = json.dumps(data, indent=4)
print(json_string)