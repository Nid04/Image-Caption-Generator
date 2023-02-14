from django.shortcuts import render, redirect
from .forms import ImageUploadForm
from .models import Image

def upload_image(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('images')
    else:
        form = ImageUploadForm()
    return render(request, 'upload/upload.html', {'form': form})

def images(request):
    image = Image.objects.latest('uploaded_at')
    return render(request, 'upload/images.html', {'image': image})