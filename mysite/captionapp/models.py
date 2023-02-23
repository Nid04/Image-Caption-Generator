from django.db import models

class Image(models.Model):
    image = models.ImageField(upload_to='images/')
    feedback = models.CharField(max_length=255, null=True, blank=True)

    def __str__(self):
        return self.feedback
