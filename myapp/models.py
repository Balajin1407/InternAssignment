from django.db import models
from django.contrib.auth.models import User
from django.utils.text import slugify
import random
import string

# Chat Session Model
class ChatSession(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    title = models.CharField(max_length=200, default="New Chat")
    slug = models.SlugField(unique=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    has_document = models.BooleanField(default=False)

    class Meta:
        ordering = ['-updated_at']

    def __str__(self):
        return f'{self.user.username}: {self.title}'

    def save(self, *args, **kwargs):
        if not self.slug or self.slug.startswith("chat-"):
            base_slug = slugify(self.title) if self.title != "New Chat" else "chat"
            slug = base_slug
            while ChatSession.objects.filter(slug=slug).exclude(pk=self.pk).exists():
                slug = f"{base_slug}-{''.join(random.choices(string.ascii_lowercase + string.digits, k=4))}"
            self.slug = slug
        super().save(*args, **kwargs)


# Chat Message Model
class Chat(models.Model):
    session = models.ForeignKey(ChatSession, on_delete=models.CASCADE, related_name='messages')
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    message = models.TextField()
    response = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    is_rag_response = models.BooleanField(default=False)

    class Meta:
        ordering = ['created_at']

    def __str__(self):
        return f'{self.user.username}: {self.message[:50]}'


# Document Model
class Document(models.Model):
    session = models.ForeignKey(ChatSession, on_delete=models.CASCADE, related_name='documents')
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    file = models.FileField(upload_to='documents/%Y/%m/%d/')
    filename = models.CharField(max_length=255)
    file_type = models.CharField(max_length=10)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    chunk_count = models.IntegerField(default=0)
    processed = models.BooleanField(default=False)

    class Meta:
        ordering = ['-uploaded_at']

    def __str__(self):
        return f'{self.user.username}: {self.filename}'

