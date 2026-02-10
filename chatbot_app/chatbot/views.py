from django.shortcuts import render
from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
import json
import logging
import os
import sys

# Add modules folder to path
MODULES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'modules')
if MODULES_DIR not in sys.path:
    sys.path.insert(0, MODULES_DIR)

from rag_module import create_pipeline, RAGConfig

# OCR module import
try:
    from ocr_module import extract_text_from_bytes
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

logger = logging.getLogger(__name__)

# Global pipeline instance to avoid reloading on every request
pipeline = None


def get_pipeline():
    global pipeline
    if pipeline is None:
        try:
            # RAGConfig initialization
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
    return render(request, 'chatbot/index.html')


@csrf_exempt
def chat_api(request):
    """Chat API endpoint supporting both JSON and multipart/form-data (file uploads).
    
    Returns:
        - normalized_query: The standardized/normalized query
        - references: List of retrieved and reranked legal documents
        - response: The LLM-generated answer
    """
    if request.method != 'POST':
        return JsonResponse({'error': 'Method not allowed'}, status=405)

    try:
        # Check content type for file uploads
        content_type = request.content_type or ''
        extra_context = None
        file_uploaded_this_request = False  # ✅ Track if file was uploaded in THIS request
        
        if 'multipart/form-data' in content_type:
            # Handle file upload request
            message = request.POST.get('message', '').strip()
            uploaded_files = request.FILES.getlist('files')
            
            if not message:
                return JsonResponse({'error': 'No message provided'}, status=400)
            
            # Process OCR for uploaded files
            if uploaded_files and OCR_AVAILABLE:
                file_uploaded_this_request = True  # ✅ File uploaded in this request
                ocr_texts = []
                logger.info(f"🔍 Processing {len(uploaded_files)} uploaded file(s)")
                for i, uploaded_file in enumerate(uploaded_files, start=1):
                    try:
                        file_bytes = uploaded_file.read()
                        filename = uploaded_file.name
                        logger.info(f"📄 OCR processing file {i}: {filename} ({len(file_bytes)} bytes)")
                        
                        ocr_result = extract_text_from_bytes(
                            file_bytes,
                            filename,
                            gpu=False,
                            dpi=200,
                            prefer_easyocr=False,  # ✅ Tesseract만 사용
                        )
                        
                        extracted_text = ocr_result.text.strip()
                        logger.info(f"✅ OCR result for {filename}: mode={ocr_result.mode}, chars={len(extracted_text)}")
                        
                        if extracted_text:
                            ocr_texts.append(
                                f"[첨부문서 {i}: {filename} | OCR: {ocr_result.mode} | {len(extracted_text)}자]\n{extracted_text}"
                            )
                        else:
                            logger.warning(f"⚠️ OCR returned empty text for {filename}")
                    except Exception as e:
                        logger.warning(f"❌ OCR failed for {uploaded_file.name}: {e}")
                
                if ocr_texts:
                    joined = "\n\n".join(ocr_texts)
                    max_chars = 12000
                    extra_context = joined[:max_chars] if len(joined) > max_chars else joined
                    
                    # ✅ 세션에 OCR 컨텍스트 저장 (follow-up 질문용)
                    request.session['ocr_context'] = extra_context
                    logger.info(f"📋 Total OCR context: {len(extra_context)} chars (saved to session)")
                else:
                    logger.warning("⚠️ No OCR text extracted from any files")
            elif uploaded_files and not OCR_AVAILABLE:
                logger.warning("OCR module not available, ignoring uploaded files")
            
            # ✅ 파일이 없어도 세션에서 OCR 컨텍스트 가져오기 (follow-up 질문용)
            if not extra_context:
                extra_context = request.session.get('ocr_context')
                if extra_context:
                    logger.info(f"📋 Using OCR context from session ({len(extra_context)} chars)")
        
        else:
            # Handle JSON request (backward compatible)
            data = json.loads(request.body)
            message = data.get('message', '').strip()
            
            # ✅ JSON 요청도 세션에서 OCR 컨텍스트 가져오기
            extra_context = data.get('extra_context') or request.session.get('ocr_context')
            
            if not message:
                return JsonResponse({'error': 'No message provided'}, status=400)

        rag = get_pipeline()
        if not rag:
            return JsonResponse(
                {'error': 'AI System is not ready. Check logs or environment variables.'},
                status=503
            )

        # Log if extra_context is being passed
        if file_uploaded_this_request:
            logger.info(f"📎 File uploaded this request - passing OCR context to pipeline ({len(extra_context)} chars)")
        elif extra_context:
            logger.info(f"📋 Using session OCR context for follow-up ({len(extra_context)} chars)")
        else:
            logger.info("💬 No extra context (no file uploaded and no session data)")

        # Use answer_with_trace to get normalized_query, references, and answer
        # Pass use_contract_mode=True only when file is uploaded in THIS request
        result = rag.answer_with_trace(
            message, 
            extra_context=extra_context,
            use_contract_mode=file_uploaded_this_request  # ✅ Only use contract prompt when file uploaded
        )
        
        return JsonResponse({
            'normalized_query': result.get('normalized_query', message),
            'references': result.get('references', []),
            'response': result.get('answer', ''),
            'has_ocr_context': bool(extra_context),  # ✅ 프론트엔드에 OCR 상태 알림
        })

    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)
    except Exception as e:
        logger.error(f"Error in chat_api: {e}")
        return JsonResponse({'error': str(e)}, status=500)


@csrf_exempt
def clear_ocr_context(request):
    """세션에 저장된 OCR 컨텍스트 삭제"""
    if request.method == 'POST':
        if 'ocr_context' in request.session:
            del request.session['ocr_context']
            logger.info("🗑️ OCR context cleared from session")
            return JsonResponse({'message': 'OCR context cleared'})
        else:
            return JsonResponse({'message': 'No OCR context to clear'})
    return JsonResponse({'error': 'Method not allowed'}, status=405)