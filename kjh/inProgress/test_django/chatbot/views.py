
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import logging
import os
from .rag_module import create_pipeline, RAGConfig

logger = logging.getLogger(__name__)

# Global pipeline instance to avoid reloading on every request
pipeline = None

def get_pipeline():
    global pipeline
    if pipeline is None:
        try:
            # RAGConfig initialization
            # Ensure keys are in env, which dotenv loaded in settings.py
            config = RAGConfig(
                temperature=0.1,
                enable_rerank=True,
                enable_bm25=True,
            )
            pipeline = create_pipeline(config=config)
            logger.info("RAG Pipeline initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            return None
    return pipeline

def index(request):
    # Initialize pipeline on index load to warm it up (optional, but good for UX)
    # get_pipeline() 
    return render(request, 'chatbot/index.html')

@csrf_exempt
def chat_api(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            message = data.get('message')
            if not message:
                return JsonResponse({'error': 'No message provided'}, status=400)

            rag = get_pipeline()
            if not rag:
                # Try one more time?
                return JsonResponse({'error': 'AI System is not ready. Check logs or environment variables.'}, status=503)

            response = rag.generate_answer(message)
            return JsonResponse({'response': response})
        except json.JSONDecodeError:
             return JsonResponse({'error': 'Invalid JSON'}, status=400)
        except Exception as e:
            logger.error(f"Error in chat_api: {e}")
            return JsonResponse({'error': str(e)}, status=500)
    return JsonResponse({'error': 'Method not allowed'}, status=405)
