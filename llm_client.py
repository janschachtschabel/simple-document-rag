"""
Centralized OpenAI client and retry logic.
All modules should import from here instead of creating their own clients.
"""
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import openai as openai_module
from config import Config

# Single shared OpenAI client instance
client = OpenAI(api_key=Config.OPENAI_API_KEY)

# Retry decorator for OpenAI API calls (handles rate limits, timeouts, server errors)
openai_retry = retry(
    retry=retry_if_exception_type((
        openai_module.RateLimitError,
        openai_module.APITimeoutError,
        openai_module.APIConnectionError,
        openai_module.InternalServerError,
    )),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(4),
    reraise=True
)
