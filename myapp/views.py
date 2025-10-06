from django.http import JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.contrib.auth.models import User
from django.contrib import auth
from .models import ChatSession, Chat, Document
from dotenv import load_dotenv
from django.utils import timezone
from openai import OpenAI
from django.core.files.storage import default_storage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document as LangchainDocument
import os
from PyPDF2 import PdfReader
from docx import Document as DocxDocument

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)
embeddings = OpenAIEmbeddings()


def register(request):
    if request.method == 'POST':
        username = request.POST['username']
        email = request.POST['email']
        password1 = request.POST['password1']
        password2 = request.POST['password2']
        if password1 == password2:
            try:
                user = User.objects.create_user(username, email, password1)
                user.save()
                auth.login(request, user)
                return redirect('chatbot')
            except:
                error_message = 'Error creating user. Username may already exist.'
                return render(request, 'register.html', {'error_message': error_message})
        else:
            error_message = 'Passwords do not match'
            return render(request, 'register.html', {'error_message': error_message})
    return render(request, 'register.html')


def login(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = auth.authenticate(request, username=username, password=password)
        if user is not None:
            auth.login(request, user)
            return redirect('chatbot')
        else:
            error_message = 'Invalid Username & Password'
            return render(request, 'login.html', {'error_message': error_message})
    else:
        return render(request, 'login.html')


def logout(request):
    auth.logout(request)
    return redirect('login')


def new_chat(request):
    if request.user.is_authenticated:
        new_session = ChatSession.objects.create(user=request.user)
        return redirect('chat_session', slug=new_session.slug)
    return redirect('login')


def process_file(file, filename, session):
    ext = os.path.splitext(filename)[1].lower()
    
    try:
    
        if ext == ".txt":
            text = file.read().decode("utf-8", errors="ignore")
            
        elif ext == ".pdf":
            reader = PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
                
        elif ext == ".docx":
            doc = DocxDocument(file)
            text = "\n".join([para.text for para in doc.paragraphs])
        else:
            return f"Unsupported file type: {ext}"
        
        # Text into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_text(text)
        
        # Save file
        file.seek(0)
        saved_path = default_storage.save(f'documents/{session.slug}/{filename}', file)
        
        # Create Doc record
        doc_record = Document.objects.create(session=session,user=session.user,file=saved_path,filename=filename,file_type=ext.replace('.', ''), chunk_count=len(chunks),processed=False)
        
        # FAISS vector store
        # Convert chunks to Langchain documents
        documents = [LangchainDocument(page_content=chunk) for chunk in chunks]
        
        # Create FAISS vector store with embeddings
        vector_store = FAISS.from_documents(documents, embeddings)
        
        # Save FAISS index to disk
        index_path = f"media/faiss_indexes/{session.slug}"
        os.makedirs("media/faiss_indexes", exist_ok=True)
        vector_store.save_local(index_path)
        
        # Step 7: Update session
        session.has_document = True
        session.save()
        
        doc_record.processed = True
        doc_record.save()
        
        return f"File '{filename}' processed! {len(chunks)} chunks created. Ask questions about it!"
    
    except Exception as e:
        return f"Error: {str(e)}"


def rag_answer(query, session):
    try:
        # Loading FAISS index
        index_path = f"media/faiss_indexes/{session.slug}"
        if not os.path.exists(index_path):
            return None
        
        vector_store = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        
        # Search for similar chunks
        results = vector_store.similarity_search(query, k=3)
        
        if not results:
            return None
        
        # Combine chunks as context
        context = "\n\n".join([doc.page_content for doc in results])
        
        # Ask AI with context
        prompt = f"""Based on this context, answer the question: Context: {context} Question: {query} Answer:"""
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        print(f"RAG Error: {e}")
        return None


def ask_ai(message, session):
    previous_chats = Chat.objects.filter(session=session).order_by('created_at')[:10]
    
    conversation = [{"role": "system", "content": "You are a helpful assistant."}]
    
    for chat in previous_chats:
        conversation.append({"role": "user", "content": chat.message})
        conversation.append({"role": "assistant", "content": chat.response})
    
    conversation.append({"role": "user", "content": message})
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=conversation,
            max_tokens=500,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"


def generate_chat_title(first_message):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Create a short title (3-6 words) for this conversation:"
                },
                {"role": "user", "content": first_message}
            ],
            max_tokens=20,
            temperature=0.3,
        )
        title = response.choices[0].message.content.strip().strip('"').strip("'")
        return title[:50]
    except:
        return first_message[:30] + "..." if len(first_message) > 30 else first_message


def chatbot(request, slug=None):
    if not request.user.is_authenticated:
        return redirect('login')

    user_sessions = ChatSession.objects.filter(user=request.user).order_by('-updated_at')

    if slug:
        current_session = get_object_or_404(ChatSession, slug=slug, user=request.user)
        chats = Chat.objects.filter(session=current_session, user=request.user).order_by('created_at')
    else:
        current_session = None
        chats = []

    if request.method == 'POST':
        message = request.POST.get('message')
        uploaded_file = request.FILES.get('file')
        
        if uploaded_file:
            if not current_session:
                current_session = ChatSession.objects.create(
                    user=request.user,
                    title=f"Document: {uploaded_file.name[:30]}"
                )
            
            status = process_file(uploaded_file, uploaded_file.name, current_session)
            return JsonResponse({
                'message': 'File uploaded',
                'response': status,
                'session_slug': current_session.slug
            })
        
        if not message:
            return JsonResponse({'error': 'Message cannot be empty'}, status=400)
        
        if not current_session:
            current_session = ChatSession.objects.create(user=request.user, title='New Chat')
        
        if current_session.has_document:
            response = rag_answer(message, current_session)
            is_rag = True if response else False
        else:
            response = None
            is_rag = False
        
        if response is None:
            response = ask_ai(message, current_session)
            is_rag = False
        
        Chat.objects.create(session=current_session,user=request.user,message=message,response=response,is_rag_response=is_rag,created_at=timezone.now())
        
        if current_session.title == 'New Chat':
            current_session.title = generate_chat_title(message)
        
        current_session.updated_at = timezone.now()
        current_session.save()
        
        return JsonResponse({'message': message,'response': response,'session_slug': current_session.slug})
    
    return render(request, 'chatbot.html', {'chats': chats,'user_sessions': user_sessions,'current_session': current_session})