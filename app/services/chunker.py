import re
import tiktoken
import logging
from app.services.parser import ParsingError

logger = logging.getLogger(__name__)

def split_into_sentences(text: str) -> list[str]:
    """
    Splits text into sentences using regex boundary detection.
    Prevents splitting common abbreviations (e.g. Mr., Dr., i.e., e.g.).
    """
    sentence_end = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s+')
    sentences = sentence_end.split(text)
    return [s.strip() for s in sentences if s.strip()]

class Chunker:
    def __init__(self, target_tokens: int = 500, overlap_tokens: int = 50):
        self.target_tokens = target_tokens
        self.overlap_tokens = overlap_tokens
        # Use cl100k_base encoding as standard for OpenAI / Voyage embeddings
        try:
            self.encoder = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            logger.warning(f"Failed to load cl100k_base encoding: {str(e)}. Falling back to default.")
            # Fallback encoder if internet is blocked / cache not available
            self.encoder = tiktoken.get_encoding("gpt2")

    def count_tokens(self, text: str) -> int:
        """Counts tokens in a string."""
        return len(self.encoder.encode(text))

    def chunk_document(self, blocks: list[dict]) -> list[dict]:
        """
        Groups parsed text blocks into chunks of ~500 tokens with a ~50 token overlap.
        Maintains structural sentence boundaries (never splits a sentence).
        """
        # Step 1: Flatten blocks into individual sentences with their page/position metadata
        sentences_with_meta = []
        for block in blocks:
            content = block["content"]
            page = block["page"]
            pos = block["position"]
            
            sentences = split_into_sentences(content)
            for s in sentences:
                token_cnt = self.count_tokens(s)
                # If a single sentence is larger than target_tokens, we have to split it by character/words
                # to prevent infinite loop or massive chunks.
                if token_cnt > self.target_tokens:
                    logger.warning(f"Extremely long sentence encountered ({token_cnt} tokens). Splitting by words.")
                    words = s.split()
                    sub_sentence = []
                    sub_tokens = 0
                    for word in words:
                        word_tokens = self.count_tokens(word + " ")
                        if sub_tokens + word_tokens > self.target_tokens and sub_sentence:
                            sentences_with_meta.append({
                                "text": " ".join(sub_sentence),
                                "page": page,
                                "position": pos,
                                "tokens": sub_tokens
                            })
                            sub_sentence = [word]
                            sub_tokens = word_tokens
                        else:
                            sub_sentence.append(word)
                            sub_tokens += word_tokens
                    if sub_sentence:
                        sentences_with_meta.append({
                            "text": " ".join(sub_sentence),
                            "page": page,
                            "position": pos,
                            "tokens": sub_tokens
                        })
                else:
                    sentences_with_meta.append({
                        "text": s,
                        "page": page,
                        "position": pos,
                        "tokens": token_cnt
                    })

        if not sentences_with_meta:
            raise ParsingError("No text segments could be extracted or chunked.")

        # Step 2: Build chunks using a sliding sentence window
        chunks = []
        current_sentences = []
        current_tokens = 0

        for idx, sentence in enumerate(sentences_with_meta):
            current_sentences.append(sentence)
            current_tokens += sentence["tokens"]

            # If we reach or exceed the target token count
            if current_tokens >= self.target_tokens:
                # Create a chunk from current accumulator
                chunk_data = self._create_chunk_object(current_sentences, len(chunks))
                chunks.append(chunk_data)

                # Collect the sentences for overlap
                # Walk backwards to get ~50 tokens overlap
                overlap_buffer = []
                overlap_tokens_count = 0
                for s in reversed(current_sentences):
                    if overlap_buffer and overlap_tokens_count + s["tokens"] > self.overlap_tokens:
                        break
                    if not overlap_buffer and s["tokens"] >= self.target_tokens:
                        # This single unit is already at/above the chunk target size
                        # (e.g. a word-split fragment of an oversized sentence).
                        # Carrying it forward as "overlap" would immediately blow the
                        # next chunk's budget on its own, so start the next chunk fresh.
                        break
                    overlap_buffer.insert(0, s)
                    overlap_tokens_count += s["tokens"]

                current_sentences = overlap_buffer
                current_tokens = overlap_tokens_count

        # Flush any remaining sentences
        # Ensure we don't write an empty or duplicate chunk (if the last chunk is identical to the previous chunk)
        if current_sentences:
            chunk_data = self._create_chunk_object(current_sentences, len(chunks))
            # Only append if it contains new content not fully represented in a small trailing overlap
            if not chunks or chunk_data["content"] != chunks[-1]["content"]:
                chunks.append(chunk_data)

        return chunks

    def _create_chunk_object(self, sentences: list[dict], index: int) -> dict:
        content = " ".join(s["text"] for s in sentences)
        pages = list(sorted(list(set(s["page"] for s in sentences))))
        positions = list(sorted(list(set(s["position"] for s in sentences))))
        total_tokens = sum(s["tokens"] for s in sentences)

        return {
            "chunk_index": index,
            "content": content,
            "metadata": {
                "pages": pages,
                "primary_page": pages[0] if pages else 1,
                "positions": positions,
                "token_count": total_tokens,
                "char_count": len(content)
            }
        }
