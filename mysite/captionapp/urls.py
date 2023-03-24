from django.urls import path
from .views import upload_image, feedback


urlpatterns = [
    path('', upload_image, name='upload_image'),
    path('feedback/', feedback, name='feedback'),
]
