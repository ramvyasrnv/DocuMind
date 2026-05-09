# importing libraries 

import os
import chromadb
import PyPDF2
from dotenv import load_dotenv
from openai import OpenAI

# STEP 2 — Load API Key
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# STEP 3 — Read PDF
def read_pdf(file_path):
    print(f"Opening PDF: {file_path}")
    
    all_text = ""
    
    with open(file_path, "rb") as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        total_pages = len(pdf_reader.pages)
        print(f"Total pages found: {total_pages}")
        
        for page_number in range(total_pages):
            page = pdf_reader.pages[page_number]
            page_text = page.extract_text()
            
            if page_text:
                all_text += page_text
                print(f"Page {page_number + 1} extracted")
    
    print(f"\nPDF reading complete!")
    print(f"Total characters extracted: {len(all_text)}")
    
    return all_text



pdf_path = "data/Classic350.pdf"

extracted_text = read_pdf(pdf_path)

# STEP 4 — Chunking
def split_into_chunks(text, chunk_size=504, overlap=100):
    """
    Splits big text into smaller overlapping chunks.
    
    chunk_size = how many characters per chunk
    overlap    = how many characters to repeat 
                 between chunks
    """
    
    print(f"\nStarting chunking...")
    print(f"Chunk size: {chunk_size} characters")
    print(f"Overlap: {overlap} characters")
    
    chunks = []       
    start = 0         
    
    while start < len(text):
        
        end = start + chunk_size
        
        chunk = text[start:end]
        
        chunks.append(chunk)
    
        start = end - overlap
    
    print(f"Total chunks created: {len(chunks)}")
    print(f"\n--- FIRST CHUNK PREVIEW ---")
    print(chunks[0])
    print(f"\n--- SECOND CHUNK PREVIEW ---")
    print(chunks[1])
    
    return chunks

# STEP 5 — Embeddings
def get_embedding(text):
    """
    Send one piece of text to OpenAI.
    Get back 1536 numbers (the embedding/vector).
    """
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    
    embedding = response.data[0].embedding
    
    return embedding


# STEP 6 — Setup ChromaDB

def setup_chromadb():
    """
    Create a ChromaDB database on your computer.
    Create a collection (like a table) to store our chunks.
    """
    print("\nSetting up ChromaDB...")
    

    chroma_client = chromadb.PersistentClient(path="my_database")
    
    collection = chroma_client.get_or_create_collection(
        name="royal_enfield_manual"
    )
    
    print("ChromaDB ready!")
    return collection


# STEP 7 — Store Chunks in ChromaDB

def store_chunks_in_chromadb(chunks, collection):
    """
    Take all chunks.
    Convert each one to embedding using OpenAI.
    Store chunk text + embedding in ChromaDB.
    """
    print(f"\nStoring {len(chunks)} chunks in ChromaDB...")
    print("This may take a minute - we are calling OpenAI for each chunk...")
    
    for i, chunk in enumerate(chunks):
        
        embedding = get_embedding(chunk)
        

        collection.add(
            ids=[f"chunk_{i}"],
            documents=[chunk],
            embeddings=[embedding]
        )
        
        if i % 10 == 0:
            print(f"Stored {i}/{len(chunks)} chunks...")
    
    print(f"\nAll {len(chunks)} chunks stored in ChromaDB!")

# STEP 8 — Similarity Search

def find_relevant_chunks(question, collection):
    """
    Take the user's question.
    Convert it to embedding.
    Search ChromaDB for the 3 most similar chunks.
    Return those chunks.
    """
    print(f"\n🔍 Searching for relevant chunks...")
    print(f"Question: {question}")
    

    question_embedding = get_embedding(question)
    

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=3
    )

    relevant_chunks = results['documents'][0]
    distances = results['distances'][0]
    
    print(f"\n✅ Found {len(relevant_chunks)} relevant chunks!")
    print(f"Similarity scores: {distances}")
    print(f"\n--- Most Relevant Chunk ---")
    print(relevant_chunks[0][:200])  # preview first chunk
    
    return relevant_chunks


# STEP 9 — Send to GPT

def get_answer(question, relevant_chunks):
    """
    Take the question + relevant chunks.
    Build a prompt.
    Send to GPT.
    Get answer back.
    """
    print(f"\n🤖 Sending to GPT...")
    

    context = "\n\n".join(relevant_chunks)
    

    prompt = f"""You are a helpful assistant for Royal Enfield Classic 350 motorcycle.
    
Use ONLY the following information from the owner's manual to answer the question.
If the answer is not in the provided information, say "I could not find this in the manual."

INFORMATION FROM MANUAL:
{context}

USER QUESTION:
{question}

ANSWER:"""
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system", 
                "content": "You are a helpful Royal Enfield motorcycle assistant. Answer only based on the provided manual content."
            },
            {
                "role": "user", 
                "content": prompt
            }
        ],
        temperature=0.1  
    )
    
    answer = response.choices[0].message.content
    
    return answer


# STEP 10 — Main Program Flow

# Step 1: Read PDF
pdf_path = "data/Classic350.pdf"
extracted_text = read_pdf(pdf_path)

# Step 2: Split into chunks
chunks = split_into_chunks(extracted_text)

# Step 3: Setup ChromaDB
collection = setup_chromadb()


store_chunks_in_chromadb(chunks, collection)

print("\n✅ DATABASE IS READY!")
print(f"Total chunks stored: {collection.count()}")


print("\n" + "="*50)
print("🏍️ ROYAL ENFIELD MANUAL ASSISTANT READY!")
print("="*50)
print("Ask me anything about your Royal Enfield Classic 350")
print("Type 'quit' to exit\n")

# STEP 11 — Chat Loop
while True:
    
    question = input("Your Question: ")

    if question.lower() == "quit":
        print("Goodbye! 🏍️")
        break

    if question.strip() == "":
        print("Please type a question!")
        continue

    relevant_chunks = find_relevant_chunks(question, collection)

    answer = get_answer(question, relevant_chunks)

    print("\n" + "="*50)
    print("ANSWER:")
    print("="*50)
    print(answer)
    print("="*50 + "\n")