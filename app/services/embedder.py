import time
import random
import logging
from openai import OpenAI, OpenAIError, RateLimitError, APIConnectionError, APITimeoutError
from app.config import settings

logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self):
        self.api_key = settings.OPENAI_API_KEY
        self.model = settings.EMBEDDING_MODEL
        self.dimension = settings.EMBEDDING_DIMENSION
        self.batch_size = settings.EMBEDDING_BATCH_SIZE
        
        # Initialize client unless it's configured as mock/development
        if self.api_key and self.api_key != "mock":
            self.client = OpenAI(api_key=self.api_key)
        else:
            self.client = None
            logger.warning("OPENAI_API_KEY is not set or set to 'mock'. Using mock embeddings generator.")

    def _generate_mock_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Generates random vectors for development and offline testing."""
        mock_embeddings = []
        for text in texts:
            # Generate deterministic mock values based on hash of text or simple random numbers
            # We use a seeded pseudo-random approach to ensure consistent dimensions
            seed = sum(ord(c) for c in text[:100])
            random.seed(seed)
            vector = [random.uniform(-1, 1) for _ in range(self.dimension)]
            # Normalize vector
            norm = sum(x**2 for x in vector)**0.5
            normalized_vector = [x / norm for x in vector] if norm > 0 else vector
            mock_embeddings.append(normalized_vector)
        return mock_embeddings

    def get_embeddings(self, texts: list[str], max_retries: int = 5, initial_delay: float = 1.0) -> list[list[float]]:
        """
        Generates embeddings for a list of texts in batches.
        Implements exponential backoff and jitter to handle rate limits and API failures.
        """
        if not texts:
            return []

        if self.client is None:
            return self._generate_mock_embeddings(texts)

        all_embeddings = []
        
        # Batch the texts
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = self._get_batch_embeddings_with_retry(
                batch, max_retries=max_retries, initial_delay=initial_delay
            )
            all_embeddings.extend(batch_embeddings)
            
        return all_embeddings

    def _get_batch_embeddings_with_retry(self, batch: list[str], max_retries: int, initial_delay: float) -> list[list[float]]:
        """
        Executes embedding API call for a single batch with retry logic.
        """
        delay = initial_delay
        
        for attempt in range(max_retries):
            try:
                response = self.client.embeddings.create(
                    input=batch,
                    model=self.model
                )
                # Extract and sort embeddings by original list order
                embeddings_data = sorted(response.data, key=lambda x: x.index)
                return [item.embedding for item in embeddings_data]
                
            except (RateLimitError, APITimeoutError, APIConnectionError) as e:
                if attempt == max_retries - 1:
                    logger.error(f"Failed to generate embeddings after {max_retries} attempts: {str(e)}")
                    raise e
                
                # Apply exponential backoff with random jitter (between 0.5x and 1.5x)
                jitter = random.uniform(0.5, 1.5)
                sleep_time = delay * jitter
                logger.warning(
                    f"Embedding API error (attempt {attempt + 1}/{max_retries}): {type(e).__name__}. "
                    f"Retrying in {sleep_time:.2f} seconds..."
                )
                time.sleep(sleep_time)
                delay *= 2  # Exponential growth
                
            except OpenAIError as e:
                # Other non-retryable OpenAI errors (e.g. authentication, invalid request schema)
                logger.error(f"Non-retryable OpenAI error encountered: {str(e)}")
                raise e
            except Exception as e:
                logger.error(f"Unexpected error during embedding generation: {str(e)}")
                raise e
                
        raise OpenAIError("Failed to retrieve embeddings from OpenAI.")

# Instantiate service singleton
embedding_service = EmbeddingService()
