import re
import ollama

MODEL_NAME = "qwen3:4b"  # Fast local model for synthesis

class AnswerSynthesizer:
    def __init__(self, model_name=MODEL_NAME, num_ctx=8192):
        self.model_name = model_name
        self.num_ctx = num_ctx

    def synthesize(self, query: str, context: str, chat_history: list = None) -> dict:
        """
        Synthesizes an answer based on query, context, and conversation history using a Chain-of-Thought approach.
        Returns a dictionary with 'thought' and 'answer'.
        """
        system_prompt = """
        IDENTITY:
        You are HERMES, a cognitive Hybrid Graph-Vector RAG assistant. Your goal is to provide highly accurate, objective, and detailed answers in Turkish, based strictly on the provided context.
        
        RULES:
        1. LANGUAGE: You MUST write your analysis and answer in Turkish.
        2. CONTEXT ADHERENCE: Answer the user query using ONLY the provided DOKÜMAN PARÇALARI, Kavram İlişkileri, and İLGİLİ KAVRAM TANIMLARI. Do not assume or extrapolate beyond the provided data. If the answer is not in the context, clearly state: "Verilen kaynaklarda bu bilgi bulunmamaktadır."
        3. CHAIN OF THOUGHT: Before writing the actual answer, you MUST write down your reasoning and plan inside <thought> and </thought> tags. Do not skip this thinking process.
        
        STRUCTURE OF RESPONSE:
        <thought>
        - Analyze the user query.
        - Plan which facts from the context to use.
        - Structure the final response.
        </thought>
        [Detailed Turkish Answer here, listing key concepts and facts clearly]
        """

        user_content = f"""
        KAYNAK BAĞLAMI (CONTEXT):
        {context}
        
        KULLANICI SORUSU (QUERY):
        {query}
        """

        # Build messages including chat history
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        if chat_history:
            for turn in chat_history:
                messages.append({
                    "role": turn["role"],
                    "content": turn["content"]
                })
                
        # Append latest turn
        messages.append({
            "role": "user",
            "content": user_content
        })

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=messages,
                options={"temperature": 0.2, "num_predict": 2048, "num_ctx": self.num_ctx}
            )
            
            raw_response = response["message"]["content"]
            return self.parse_thought_and_answer(raw_response)
        except Exception as e:
            return {
                "thought": f"Model çağrısı sırasında hata oluştu: {str(e)}",
                "answer": "Üzgünüm, cevap sentezlenirken teknik bir hata oluştu."
            }

    def parse_thought_and_answer(self, text: str) -> dict:
        """
        Extracts content inside <thought>...</thought> and the main response.
        Handles missing tags gracefully.
        """
        # Look for <thought>...</thought> tags case-insensitively
        thought_match = re.search(r'<thought>(.*?)</thought>', text, re.DOTALL | re.IGNORECASE)
        
        if thought_match:
            thought = thought_match.group(1).strip()
            # The answer is everything outside the <thought>...</thought> block
            answer = re.sub(r'<thought>.*?</thought>', '', text, flags=re.DOTALL | re.IGNORECASE).strip()
        else:
            # Fallback: check if only opening tag exists
            thought_start_match = re.search(r'<thought>(.*)', text, re.DOTALL | re.IGNORECASE)
            if thought_start_match:
                # If there's an opening tag but no closing tag
                parts = re.split(r'<thought>', text, flags=re.IGNORECASE)
                thought = parts[1].strip() if len(parts) > 1 else ""
                answer = parts[0].strip()
            else:
                # No tags found
                thought = "Düşünme aşaması model tarafından etiketlenmedi."
                answer = text.strip()
                
        # Clean up any leftover tags or excessive whitespace
        answer = re.sub(r'^\s*</thought>\s*', '', answer, flags=re.IGNORECASE)
        
        return {
            "thought": thought,
            "answer": answer
        }
