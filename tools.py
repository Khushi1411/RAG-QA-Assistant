import math
import re
import requests

def calculator_tool(query):
    try:
        # Remove unsafe characters and evaluate
        expression = re.sub(r'[^0-9+\-*/(). ]', '', query)
        result = eval(expression)
        return f"The result is: {result}"
    except Exception as e:
        return f"Calculation error: {e}"

def define_tool(query):
    try:
        word = query.split("define")[-1].strip()
        if not word:
            return "Please specify a word to define."
        
        # Simple dictionary API 
        url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}"
        response = requests.get(url)
        if response.status_code == 200:
            definition = response.json()[0]["meanings"][0]["definitions"][0]["definition"]
            return f"Definition of '{word}': {definition}"
        else:
            return f"No definition found for '{word}'."
    except Exception as e:
        return f"Definition error: {e}"



