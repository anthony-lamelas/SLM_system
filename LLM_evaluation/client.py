"""
OpenAI API client for GPT model evaluations.
Supports any OpenAI model including GPT-4, GPT-5, etc.
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def get_openai_client(task_type='grammar'):
    """
    Initialize and return OpenAI client using API key from environment.
    
    Args:
        task_type (str): Either 'grammar' or 'readability' to select the appropriate API key
    
    Returns:
        OpenAI: Configured OpenAI client instance
    
    Raises:
        ValueError: If the required API key is not found in environment
    """
    if task_type == 'grammar':
        api_key = os.getenv('GRAMMAR_OPENAI_API_KEY')
        key_name = 'GRAMMAR_OPENAI_API_KEY'
    elif task_type == 'readability':
        api_key = os.getenv('READABILITY_OPENAI_API_KEY')
        key_name = 'READABILITY_OPENAI_API_KEY'
    else:
        raise ValueError(f"task_type must be 'grammar' or 'readability', got '{task_type}'")
    
    if not api_key:
        raise ValueError(
            f"{key_name} not found in environment variables. "
            f"Please add it to your .env file."
        )
    
    return OpenAI(api_key=api_key)


def generate_completion(
    client,
    model_name,
    system_prompt,
    user_prompt
):
    """
    Generate a completion using the specified OpenAI model.
    
    Args:
        client (OpenAI): OpenAI client instance
        model_name (str): Model identifier (e.g., 'gpt-4', 'gpt-5', 'gpt-4-turbo')
        system_prompt (str): System prompt defining model behavior
        user_prompt (str): User prompt with the task
    
    Returns:
        str: Generated text response
    
    Raises:
        Exception: If API call fails
    """
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        raise Exception(f"OpenAI API call failed: {str(e)}")
