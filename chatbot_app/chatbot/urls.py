
from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('api/chat/', views.chat_api, name='chat_api'),
    path('api/clear-ocr/', views.clear_ocr_context, name='clear_ocr'),  # ✅ 이거 추가
]