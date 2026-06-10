from ollama import chat
from youtube_transcript_api import YouTubeTranscriptApi
from googleapiclient.discovery import build
import os


def search_youtube(query:str,max_searches:int=3) ->list[str] :
    """
    this function will search the query in youtube and find the top most videos and returns video id .

    query : the topic that needed to be searched in youtube
    max_searches: maximum number of videos needed to be returned . default is 3.
    """
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
def get_transcript(video_id:str)->str:
    
    transcript = YouTubeTranscriptApi().fetch(video_id=video_id)
    text = " ".join(snippet.text for snippet in transcript.snippets)
    print("got transcript")
    return text
available_functions = {
    "search_youtube": search_youtube,
    "get_transcript": get_transcript
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
        "name":"get_transcript",
        "description":"get transcript of the youtube video using video id",
        "parameters":{
            "type":"object",
            "properties":{
                "video_id":{
                    "type": "string"
            }},
            "required":["video_id"]
        }
        }}]
while True:
    user_input = input("You:")
    if user_input == "bye":
        break
    messages.append({'role':"user","content":user_input})
    
    response = chat(model="qwen3:8b",messages=messages,tools=tools)
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
        final_response = chat(
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
