"""
DermaCortex AI - Chatbot Service
Dermatology-focused AI chatbot using OpenAI
"""

import os
from typing import List, Optional, Tuple, Dict
from openai import AsyncOpenAI

from app.config import settings
from app.core.deps import validate_dermatology_query


# Default responses for non-dermatology queries
DEFAULT_NON_DERMATOLOGY_RESPONSE = (
    "I am a dermatology-focused assistant and cannot answer non-skin-related queries. "
    "Please ask me questions about skin conditions, skincare, hair, nails, or dermatological treatments."
)

# System prompt for DermaCortex AI
SYSTEM_PROMPT = """You are DermaCortex AI, an intelligent dermatology assistant.

Your role is to help users understand skin conditions, skincare routines, and treatment options.

Rules:
- Provide helpful skincare guidance
- Explain conditions in simple language
- Suggest common dermatology treatments
- Recommend consulting a dermatologist for serious conditions
- Never claim to replace a doctor
- Avoid giving dangerous medical advice
- If unsure, say the user should seek professional care"""

# Legacy prompt for backward compatibility
DERMATOLOGY_SYSTEM_PROMPT = """You are DermaCortex, an advanced AI-powered dermatology assistant. Your role is to provide helpful, accurate information about skin conditions, skincare, hair health, nail health, and dermatological treatments.

Guidelines:
1. Only answer questions related to dermatology, skin health, hair, nails, and skincare
2. Provide evidence-based information
3. Always recommend consulting a dermatologist for proper diagnosis and treatment
4. Be clear about limitations and when to seek professional help
5. Use a friendly, professional tone
6. Provide actionable recommendations when possible
7. If asked about non-dermatology topics, politely redirect to your expertise area

Remember: You are not a substitute for professional medical advice. Always encourage users to consult with qualified healthcare providers for proper diagnosis and treatment."""


class ChatbotService:
    """Chatbot service class for dermatology-focused AI"""
    
    def __init__(self):
        self.client = None
        if settings.GROQ_API_KEY:
            self.client = AsyncOpenAI(
                api_key=settings.GROQ_API_KEY,
                base_url="https://api.groq.com/openai/v1"
            )
    
    async def chat(
        self,
        user_message: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Tuple[str, Optional[List[str]]]:
        """
        Process a chat message and return a response
        Returns: (response, suggestions)
        """
        # Validate if the query is dermatology-related
        if not validate_dermatology_query(user_message):
            suggestions = [
                "What causes acne?",
                "How to treat eczema?",
                "Best skincare routine for dry skin?",
                "What is psoriasis?",
                "How to identify skin cancer?"
            ]
            return DEFAULT_NON_DERMATOLOGY_RESPONSE, suggestions
        
        # If OpenAI client is not configured, use fallback responses
        if not self.client:
            return self._fallback_response(user_message)
        
        try:
            # Build messages for OpenAI
            messages = [
                {"role": "system", "content": DERMATOLOGY_SYSTEM_PROMPT}
            ]
            
            # Add conversation history if provided
            if conversation_history:
                messages.extend(conversation_history)
            
            # Add current message
            messages.append({"role": "user", "content": user_message})
            
            # Call Groq API
            response = await self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=messages,
                max_tokens=600,
                temperature=0.4
            )
            
            # Extract response
            bot_response = response.choices[0].message.content
            
            # Generate suggestions
            suggestions = self._generate_suggestions(user_message, bot_response)
            
            return bot_response, suggestions
            
        except Exception as e:
            print(f"Error in chatbot service: {e}")
            return self._fallback_response(user_message)
    
    def _fallback_response(self, user_message: str) -> Tuple[str, List[str]]:
        """Provide fallback responses when OpenAI is not available"""
        
        user_message_lower = user_message.lower()
        
        # Simple keyword-based responses
        responses = {
            "acne": (
                "Acne is a common skin condition caused by clogged pores from oil and dead skin cells. "
                "Treatment options include:\n"
                "- Over-the-counter products with benzoyl peroxide or salicylic acid\n"
                "- Prescription medications like retinoids or antibiotics\n"
                "- Lifestyle changes like avoiding touching your face\n"
                "- Consult a dermatologist for severe cases."
            ),
            "eczema": (
                "Eczema (atopic dermatitis) is a condition causing red, itchy, inflamed skin. "
                "Management includes:\n"
                "- Regular moisturizing with fragrance-free creams\n"
                "- Identifying and avoiding triggers\n"
                "- Using mild, gentle skincare products\n"
                "- Prescription topical corticosteroids for flare-ups\n"
                "- Consult a dermatologist for proper diagnosis and treatment plan."
            ),
            "psoriasis": (
                "Psoriasis is an autoimmune condition causing rapid skin cell buildup, "
                "resulting in thick, scaly patches. Treatment options include:\n"
                "- Topical treatments (steroids, vitamin D analogs)\n"
                "- Light therapy (phototherapy)\n"
                "- Systemic medications\n"
                "- Biologics for moderate to severe cases\n"
                "- Always consult a dermatologist for proper diagnosis."
            ),
            "dry skin": (
                "For dry skin, consider these tips:\n"
                "- Use a gentle, hydrating cleanser\n"
                "- Apply moisturizer immediately after bathing\n"
                "- Use products with hyaluronic acid or ceramides\n"
                "- Avoid hot showers and harsh soaps\n"
                "- Use a humidifier in dry climates\n"
                "- Consider products for sensitive skin."
            ),
            "sun protection": (
                "Sun protection is crucial for healthy skin:\n"
                "- Use broad-spectrum SPF 30+ sunscreen daily\n"
                "- Reapply every 2 hours when outdoors\n"
                "- Seek shade during peak sun hours (10am-4pm)\n"
                "- Wear protective clothing and hats\n"
                "- Don't forget often-missed areas like ears and neck\n"
                "- Regular skin checks can detect early signs of damage."
            ),
            "aging": (
                "To maintain youthful skin:\n"
                "- Use retinoids/retinol for collagen production\n"
                "- Apply vitamin C serum daily\n"
                "- Stay hydrated and maintain healthy diet\n"
                "- Use sunscreen religiously\n"
                "- Consider peptides and growth factors\n"
                "- Get adequate sleep (7-8 hours)\n"
                "- Avoid smoking and excessive alcohol."
            ),
            "hair loss": (
                "Hair loss can have many causes:\n"
                "- Androgenetic alopecia (hereditary)\n"
                "- Stress-related telogen effluvium\n"
                "- Nutritional deficiencies\n"
                "- Medical conditions\n"
                "- Treatment options include minoxidil, finasteride, PRP therapy\n"
                "- Consult a dermatologist for proper evaluation."
            ),
            "mole": (
                "Moles should be monitored using the ABCDE rule:\n"
                "- A: Asymmetry\n"
                "- B: Border irregularity\n"
                "- C: Color variation\n"
                "- D: Diameter >6mm\n"
                "- E: Evolving changes\n"
                "- Get regular skin checks and report any changes to a dermatologist."
            ),
            "rosacea": (
                "Rosacea causes facial redness and visible blood vessels. Management includes:\n"
                "- Identifying and avoiding triggers (spicy foods, alcohol, etc.)\n"
                "- Using gentle skincare products\n"
                "- Prescription topical/oral medications\n"
                "- Laser treatments for persistent redness\n"
                "- Consult a dermatologist for diagnosis."
            ),
            "scar": (
                "For scar treatment:\n"
                "- Silicone gel sheets or gels\n"
                "- Pressure therapy\n"
                "- Laser treatments\n"
                "- Chemical peels\n"
                "- Microneedling\n"
                "- Consult a dermatologist for personalized treatment."
            )
        }
        
        # Check for keyword matches
        for keyword, response in responses.items():
            if keyword in user_message_lower:
                suggestions = [
                    "What causes acne?",
                    "How to treat dry skin?",
                    "Best sunscreen for sensitive skin?",
                    "How to prevent aging?",
                    "When to see a dermatologist?"
                ]
                return response, suggestions
        
        # Default response
        default_response = (
            "Thank you for your dermatology question. For personalized advice, "
            "please consult a board-certified dermatologist who can properly evaluate your condition. "
            "In the meantime, maintain good skincare practices and protect your skin from sun damage."
        )
        
        suggestions = [
            "What causes acne?",
            "How to treat eczema?",
            "Best skincare for dry skin",
            "How to identify skin cancer?",
            "Sun protection tips"
        ]
        
        return default_response, suggestions
    
    def _generate_suggestions(self, user_message: str, bot_response: str) -> List[str]:
        """Generate follow-up suggestions based on the conversation"""
        
        suggestions = [
            "What are the best treatments for this condition?",
            "What ingredients should I look for?",
            "How can I prevent this from recurring?",
            "When should I see a dermatologist?",
            "Are there any side effects to be aware of?"
        ]
        
        return suggestions
    
    async def chat_with_context(
        self,
        message: str,
        context: Optional[str] = None
    ) -> str:
        """
        Process a chat message with optional context (diagnosis result)
        This is the simplified method for POST /chatbot/ endpoint
        
        Args:
            message: User's question
            context: Optional diagnosis result from skin analysis
            
        Returns:
            AI generated response string
        """
        # Validate message is not empty
        if not message or not message.strip():
            return "Please provide a message."
        
        # If OpenAI client is not configured, return fallback
        if not self.client:
            return self._simple_fallback_response(message)
        
        try:
            # Build messages array
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT}
            ]
            
            # Add context as separate system message if provided
            if context:
                messages.append({
                    "role": "system",
                    "content": f"User skin analysis result: {context}"
                })
            
            # Add user message
            messages.append({"role": "user", "content": message})
            
            # Call Groq API
            response = await self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=messages,
                temperature=0.4,
                max_tokens=600
            )
            
            # Extract reply
            reply = response.choices[0].message.content
            
            return reply
            
        except Exception as e:
            print(f"Error in chatbot service: {e}")
            return "Sorry, I couldn't process that request right now. Please try again."
    
    def _simple_fallback_response(self, user_message: str) -> str:
        """Simple fallback response when OpenAI is not available"""
        return (
            "I apologize, but I'm having trouble processing your request right now. "
            "Please try again later or consult a dermatologist for professional advice."
        )


# Create singleton instance
chatbot_service = ChatbotService()

