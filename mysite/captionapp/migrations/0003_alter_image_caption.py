# Generated by Django 4.1.7 on 2023-02-14 19:51

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("captionapp", "0002_alter_image_caption"),
    ]

    operations = [
        migrations.AlterField(
            model_name="image",
            name="caption",
            field=models.CharField(blank=True, max_length=255, null=True),
        ),
    ]
