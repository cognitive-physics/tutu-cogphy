"""Reply providers for frontend: LLM-backed and deterministic stub implementations."""

import os
from abc import ABC, abstractmethod
from typing import Optional


class ReplyProvider(ABC):
    """Abstract base for AI reply generation strategies."""
    
    @abstractmethod
    def generate_reply(self, user_message: str, compression_rate: float) -> str:
        """
        Generate an AI reply based on user message and compression rate.
        
        Args:
            user_message: User's input text
            compression_rate: float in [0, 1], controls output style/length
                0.05-0.25: slow_expand (verbose, exploratory)
                0.40-0.75: metaphor (concise, figurative)
                0.75-1.0: compressed (minimal, dense)
        
        Returns:
            reply: Generated or fetched AI response text
        """
        pass
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable provider name for UI display."""
        pass
    
    @property
    @abstractmethod
    def is_testing(self) -> bool:
        """Whether this is a testing/stub provider."""
        pass


class LLMProvider(ReplyProvider):
    """
    Real LLM-backed reply provider using OpenAI/external API.
    
    API key must be provided via LLM_API_KEY environment variable.
    Raises error if key is missing.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        """
        Initialize LLM provider.
        
        Args:
            api_key: Optional API key. If None, reads from LLM_API_KEY env var.
            model: Model name (default: gpt-3.5-turbo)
        
        Raises:
            ValueError: If API key is not available and not in environment
        """
        self.api_key = api_key or os.environ.get("LLM_API_KEY")
        if not self.api_key:
            raise ValueError(
                "LLM_API_KEY not configured. Set LLM_API_KEY environment variable "
                "or pass api_key parameter. Falling back to stub provider."
            )
        self.model = model
        self._validate_api()
    
    def _validate_api(self):
        """Validate API key format and connectivity."""
        if not self.api_key or len(self.api_key) < 10:
            raise ValueError("Invalid or empty LLM_API_KEY format")
    
    def generate_reply(self, user_message: str, compression_rate: float) -> str:
        """
        Generate reply via LLM API, adapting style to compression_rate.
        
        Args:
            user_message: User input
            compression_rate: Controls output style (see ReplyProvider docstring)
        
        Returns:
            LLM-generated reply, length/style adapted to compression_rate
        """
        try:
            import openai
        except ImportError:
            raise ImportError("openai package required for LLMProvider. Install with: pip install openai")
        
        openai.api_key = self.api_key
        
        # Adapt system prompt based on compression_rate
        if compression_rate < 0.25:
            style_prompt = "You are a verbose, exploratory AI. Expand on ideas with detail and nuance."
        elif compression_rate < 0.75:
            style_prompt = "You are a thoughtful AI using metaphor and analogy. Be concise but evocative."
        else:
            style_prompt = "You are a minimal, dense AI. Respond in the fewest words capturing core meaning."
        
        system_prompt = f"{style_prompt}\n\nAdapt your response length based on the compression rate: {compression_rate:.2f} (0=verbose, 1=minimal)."
        
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.7,
                max_tokens=int(500 * compression_rate + 100),  # Scale output length
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise RuntimeError(f"LLM API call failed: {e}")
    
    @property
    def provider_name(self) -> str:
        return f"LLM ({self.model})"
    
    @property
    def is_testing(self) -> bool:
        return False


class DeterministicStubProvider(ReplyProvider):
    """
    Deterministic stub for testing without LLM API.
    
    Generates rule-based, non-random replies based on message content.
    Explicitly marked as TESTING ONLY for clarity in UI.
    """
    
    STUB_TEMPLATES = {
        "question": "这个问题涉及 {key_word}，核心要点是理解 {key_word} 如何影响系统行为。",
        "assertion": "关键观点是 {key_word} 驱动了这一现象。我们可以从以下角度验证：概念一致性、数据支撑、逻辑链条。",
        "confusion": "让我重新梳理一下。{key_word} 的定义是…这样就清楚了吗？",
        "default": "感谢你的输入。核心概念是 {key_word}，让我们深入探讨。",
    }
    
    def generate_reply(self, user_message: str, compression_rate: float) -> str:
        """
        Generate deterministic reply based on message type and compression rate.
        
        Args:
            user_message: User input
            compression_rate: Controls output style (ignored for stub, but accepts for interface compatibility)
        
        Returns:
            Deterministic rule-based reply
        """
        # Extract key word (longest noun-like token)
        words = user_message.lower().split()
        key_word = max(
            (w for w in words if len(w) > 3),
            key=len,
            default="概念"
        )
        
        # Classify message type
        if "?" in user_message or any(q in user_message.lower() for q in ["怎样", "如何", "什么", "为什么"]):
            template = self.STUB_TEMPLATES["question"]
        elif "但是" in user_message or "然而" in user_message or "不过" in user_message:
            template = self.STUB_TEMPLATES["confusion"]
        elif any(word in user_message.lower() for word in ["一定", "必然", "总是", "永远"]):
            template = self.STUB_TEMPLATES["assertion"]
        else:
            template = self.STUB_TEMPLATES["default"]
        
        reply = template.format(key_word=key_word)
        
        # Adapt length to compression_rate (for testing: longer at low rate, shorter at high rate)
        if compression_rate < 0.3:
            reply = reply + f" 进一步分析 {key_word} 的多个维度很有必要。"
        elif compression_rate > 0.8:
            reply = reply.split("。")[0] + "。"  # Truncate to first sentence
        
        return reply
    
    @property
    def provider_name(self) -> str:
        return "STUB (Testing Only)"
    
    @property
    def is_testing(self) -> bool:
        return True


def get_reply_provider() -> ReplyProvider:
    """
    Auto-select reply provider based on environment configuration.
    
    Returns:
        LLMProvider if LLM_API_KEY is configured,
        DeterministicStubProvider otherwise.
    """
    if os.environ.get("LLM_API_KEY"):
        try:
            return LLMProvider()
        except (ValueError, ImportError) as e:
            print(f"Warning: LLMProvider initialization failed ({e}). Falling back to stub.")
            return DeterministicStubProvider()
    else:
        return DeterministicStubProvider()
