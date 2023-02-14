from django.urls import path
from .views import upload_image, images


urlpatterns = [
    path('', upload_image, name='upload_image'),
    path('image/', images, name='images')
]
