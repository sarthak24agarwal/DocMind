import logging
from typing import Iterator
from anthropic import Anthropic, AnthropicError
from app.config import settings

logger = logging.getLogger(__name__)

class AnthropicService:
    def __init__(self):
        self.api_key = settings.ANTHROPIC_API_KEY
        self.model = settings.CLAUDE_MODEL
        
        if self.api_key and self.api_key != "mock":
            self.client = Anthropic(api_key=self.api_key)
        else:
            self.client = None
            logger.warning("ANTHROPIC_API_KEY is not set or set to 'mock'. Using mock Claude responder.")

    def _generate_mock_stream(self, prompt: str):
        """Generates a mock token stream for offline testing."""
        import time
        # Provide a structured response with citations to allow testing the parser
        words = (
            "This is a mock RAG response generated for testing. "
            "Based on the documents provided in your workspace, the ingestion pipeline "
            "successfully chunks text and saves them as vectors [1]. "
            "Furthermore, all vector search operations are strictly scoped by workspace_id [2] "
            "to prevent multi-tenant data leakage. If you need details on chunk metadata, "
            "primary pages are indexed correctly [3]."
        ).split(" ")
        
        for word in words:
            yield word + " "
            time.sleep(0.02)  # Simulate API latency

    def stream_chat(self, system_prompt: str, chat_history: list[dict]) -> Iterator[str]:
        """
        Calls the Anthropic Messages API with streaming enabled.
        Yields text deltas as they arrive.
        """
        if self.client is None:
            for chunk in self._generate_mock_stream(chat_history[-1]["content"]):
                yield chunk
            return

        try:
            # chat_history format should be list of dict: [{"role": "user"|"assistant", "content": "text"}]
            stream = self.client.messages.create(
                model=self.model,
                max_tokens=2048,
                system=system_prompt,
                messages=chat_history,
                stream=True
            )
            
            for event in stream:
                # Anthropic API event structure details:
                # - content_block_delta has delta.text
                if event.type == "content_block_delta" and event.delta.type == "text_delta":
                    yield event.delta.text
                    
        except AnthropicError as e:
            logger.error(f"Anthropic API Error: {str(e)}")
            raise e
        except Exception as e:
            logger.error(f"Unexpected error calling Anthropic: {str(e)}")
            raise e

# Instantiate service singleton
anthropic_service = AnthropicService()
