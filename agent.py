import ollama
from youtube_transcript_api import YouTubeTranscriptApi
from googleapiclient.discovery import build
import os
import chromadb
import uuid
client = chromadb.PersistentClient(path="./youtube_rag_db")
collection = client.get_or_create_collection(name="youtube_data")


def search_youtube(query:str,max_searches:int=3) ->list[str] :
    youtube = build("youtube","v3",developerKey=os.getenv("YOUTUBE_API_KEY"))
    request = youtube.search().list(
        q=query,
        maxResults=max_searches,
        part = "snippet",
        type ="video"
    )

    response = request.execute()
    videos = []
    for item in response.get("items",[]):
        video_id  = item["id"]["videoId"]
        videos.append(video_id)
    
    print("youtube is fetched")
    return videos
def get_transcript_chunks(video_id:str)->str:
    
    transcript = YouTubeTranscriptApi().fetch(video_id=video_id)
    text = str(" ".join(snippet.text for snippet in transcript.snippets))
    print("got transcript")
    chunk_size=1000
    overlap=200
    if len(text)<= chunk_size:
        chunks=[text]
    else:
        start = 0
        chunks = []
        while start < len(text):
            end = start+chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start += chunk_size - overlap
    
    print("chunks created")
    for chunk in chunks :
        embedding = get_embedding(chunk)
        collection.add(
            ids=[str(uuid.uuid4())],
            documents=[chunk],
            embeddings=[embedding]
        )
    print("transcript stored in database")

def get_embedding(text):
    response = ollama.embeddings(model="nomic-embed-text",prompt=text)
    return response["embedding"]
    
def retreive_chunks(question,n_results=5):
    print(question)
    question_embedding = get_embedding(question)
    response = collection.query(query_embeddings=[question_embedding],n_results=n_results)
    print("chunks are retreieved")
    return response["documents"][0]

def ask_rag(question):
    print("question sent to ask_rag funciton: ",question)
    chunks = retreive_chunks(question)
    context = '\n\n'.join(chunks)
    prompt = f"""
    answer the question based on the context if the answer is not in the context say you dont know the answer
    context:
    {context}

    question:
    {question}"""
    ai_response = ollama.chat(model="qwen3:8b",messages=[{"role":"user","content":prompt}])
    print("rag answered the question")
    return ai_response["message"]["content"]
available_functions = {
    "search_youtube": search_youtube,
    "get_transcript_chunks": get_transcript_chunks,
    "ask_rag": ask_rag
}
messages = []
tools=[
    {"type":"function",
     "function":{
        "name":"search_youtube",
        "description":"find the youtube videos detials based on query",
        "parameters":{
            "type":"object",
            "properties":{
                "query":{
                    "type":"string"
                },
                "max_searches":{
                    "type":"integer"
                }
            },
            "required":["query","max_searches"]
        }}} , {"type":"function","function":{
        "name":"get_transcript_chunks",
        "description":"get transcript of the youtube video using video id and create chunks and store those chunks in database",
        "parameters":{
            "type":"object",
            "properties":{
                "video_id":{
                    "type": "string"
            }},
            "required":["video_id"]
        }
        }},
        {"type":"function",
         "function":{
             "name":"ask_rag",
             "description":"answers a question if the context is stored in vector database and not in the llm's training data",
             "parameters":{
                 "type":"object",
                 "properties":{
                     "question":{
                         "type":"string"
                     }
                 },"required":["question"]
             }
         }}]
while True:
    user_input = input("You:")
    if user_input == "bye":
        break
    messages.append({"role":"system","context":"""you are a youtube research agent . you have multiple tools and you should call all those tools whenever required . you should call multiple tools at once .
                     you can get transcripts of youtube videos using get_transcript_chunks function by sending video id as arguments and this function will create chunks for that transcript and store those chunks in database, 
                     you can get video id by using search_youtube funciton by sending search query as arguments,
                     you can find answers to questions which involves data that is not in your training data using ask_rag funciton which retreives relevent chunks from the vector database,
                     whenever you get a transcipt you should break it into chunks using get_chunks and store in the database using storea_transcipt,so that the user can ask questions which you should answer using ask_rag funciton.
                     don't answer questions based on the training data answer them using ask_rag or say you dont know the answer """
    })
    messages.append({'role':"user","content":user_input})
    
    response = ollama.chat(model="qwen3:8b",messages=messages,tools=tools)
    assistant_message = response["message"]
    if assistant_message.get("tool_calls"):
        messages.append(assistant_message)
        for tool_call in assistant_message["tool_calls"]:
            name = tool_call["function"]["name"]
            arguments = tool_call["function"]["arguments"]
            function = available_functions[name]
            print("calling",function)
            result = function(**arguments)
            messages.append({"role":"tool","content":str(result)})
        final_response = ollama.chat(
            model="qwen3:8b",
            messages=messages
        )
        final_answer = final_response["message"]["content"]
        messages.append({"role":"assistant","content":final_answer})
        print("qwen:",final_answer)
    else:
        answer = assistant_message["content"]
        messages.append({'role':"assistant","content":answer})
        print("qwen:",answer)
