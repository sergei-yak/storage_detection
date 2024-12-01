import os
import requests
import json
import glob
from PIL import Image
from langchain import OpenAI, hub
from openai import OpenAI as openaiconnect
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType, AgentExecutor, create_react_agent, create_structured_chat_agent
from langchain_core.tools import Tool
from langchain.memory import ConversationBufferMemory
from langchain_community.agent_toolkits import JsonToolkit, create_json_agent
from langchain_community.tools.json.tool import JsonSpec

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

import time
from requests.exceptions import ConnectTimeout, RequestException
#from openai.error import APIConnectionError

import os
from dotenv import load_dotenv #to store keys privately

# Load environment variables from the .env file
load_dotenv()

# Access the variables
your_openai_api_key = os.getenv('your_openai_api_key')
TOKEN = os.getenv('TOKEN')
chat_id = os.getenv('chat_id')

# Define the path and directories
MODEL_PATH = "yolov8s-world.pt"
INPUT_DIR = 'auction_images'
OUTPUT_DIR = 'auction_images/predictions'
web_loc = 'https://www.storagetreasures.com/auctions/tx/dallas/'
confidance = 0.5

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_predictions(json_file='predicted_items.json'):
    """Load the predicted items from a JSON file."""
    if os.path.exists(json_file):
        with open(json_file, 'r') as f:
            return json.load(f)
    return {}

##########################################################################
# Retrieve image paths for a specific item or filename
def query_item_image(query):
    """Return text responses and image paths for a specific item or filename."""
    predicted_items = load_predictions()
    response_pairs = []  # To store text and image path pairs

    # Match the query against both filenames and predicted class names
    for item, data in predicted_items.items():
        # Check if the query matches the prediction filename or class name
        if query.lower() in [cls.lower() for cls in data['predicted_data'].keys()] and max(data['predicted_data'][query.lower()])>= confidance:
            text_response = f"Found '{query}' in the auction with probability {round(max(data['predicted_data'][query.lower()])*100,2)}%. \nLink: {'https://www.storagetreasures.com/auctions/tx/dallas/'+data['auction_id']}"
            image_path = os.path.join(OUTPUT_DIR, data['prediction_filename'])
            
            # Normalize path, replaced '\\' with '/' to avoid formatting issues
            normalized_path = os.path.normpath(image_path).replace('\\', '/')
            
            # If the image path exists, add the response pair
            if os.path.exists(normalized_path):
                response_pairs.append((text_response, normalized_path))

    if response_pairs:
        print(response_pairs)
        return response_pairs
    else:
        return [(f"No '{query}' found in the current images.", None)]
    
# Send text and images to Telegram one by one -- we dont need it anymore???
def send_text_and_image_to_telegram(query):
    """Send text responses and associated images to the Telegram chat."""
    response_pairs = query_item_image(query)

    # Iterate through the response pairs and send each one individually
    for text_msg, image_path in response_pairs:
        send_to_telegram(text_msg, [image_path] if image_path else [])

def send_to_telegram(text_msg, image_paths):
    """Send a text message and associated images to the Telegram chat."""
    # Send the text message
    url_msg = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text={text_msg}"
    requests.get(url_msg, timeout=10).json()

    # Send the corresponding image (if any)
    for image_path in image_paths:
        if image_path:  # Only attempt to send if the image path is valid
            with open(image_path, "rb") as image:
                url_photo = f"https://api.telegram.org/bot{TOKEN}/sendPhoto"
                params = {"chat_id": chat_id}
                files = {"photo": image}
                requests.post(url_photo, params=params, files=files)

#########################################################################


# Retrieve image paths for a specific item or filename
def get_image_for_item(query):
    """Return image paths for a specific item or filename."""
    predicted_items = load_predictions()
    found_image_paths = []

    # Match the query against both filenames and predicted class names
    for item, data in predicted_items.items():
        # Check if the query matches the prediction filename or class name
        if query.lower() == data['original_filename'].lower() or query.lower() in [cls.lower() for cls in data['predicted_data'].keys()]:
            image_path = os.path.join(OUTPUT_DIR, data['prediction_filename'])
            
            # Normalize and replace '\\' with '/' to avoid formatting issues
            normalized_path = os.path.normpath(image_path).replace('\\', '/')
            if os.path.exists(normalized_path):
                found_image_paths.append(normalized_path)

    return found_image_paths

# Function to handle the response from the agent and send to Telegram
def handle_agent_response(agent_response, query):
    """Process the agent's response and send relevant details to Telegram."""
    # Pass agent_response['output'] to answer_query_about_items
    #query_response = answer_query_about_items(query, agent_response) # testing
    # Retrieve the list of (text, image path) pairs for the query
    response_pairs = query_item_image(query)

    # Iterate through the response pairs and send text with corresponding images
    for text_msg, image_path in response_pairs:
        # Send text and corresponding image one by one
        send_to_telegram(text_msg, [image_path] if image_path else [])
    # Handle the case when no items are found and send agent_response['output']
    if not response_pairs:
        send_to_telegram(agent_response[1], [])  # Send the agent's response as a message if no images found

##################################################################
def dashboard_plot(json_data):
    # Extract classes, scores, and occurrences
    class_scores = {}
    class_occurrences = {}

    for image_data in json_data.values():
        for obj, scores in image_data["predicted_data"].items():
            if obj not in class_scores:
                class_scores[obj] = []
                class_occurrences[obj] = 0
            class_scores[obj].extend(scores)
            class_occurrences[obj] += len(scores)

    # Prepare data for scores and occurrences bar plots
    classes = list(class_scores.keys())
    scores = [round(np.mean(scores), 2) for scores in class_scores.values()]  # Average confidence score for each class
    occurrences = list(class_occurrences.values())  # Total occurrences of each class

    # Create DataFrames for sorted plotting
    df_scores = pd.DataFrame({"classes": classes, "scores": scores}).sort_values("scores", ascending=False)
    df_occurrences = pd.DataFrame({"classes": classes, "occurrences": occurrences}).sort_values("occurrences", ascending=False)

    # Create subplots with 4 rows and 1 column
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=(
            "Confidence Scores (average per class)",
            "Class Occurrences",
            "Heatmap of Confidence Scores Across Images and Objects",
            "Box Plot of Confidence Scores for Each Object Type"
        )
    )

    # Bar plot for scores in the first row
    fig.add_trace(
        go.Bar(
            x=df_scores["classes"],
            y=df_scores["scores"],
            text=df_scores["scores"],
            textposition="outside",
            name="Scores"
        ),
        row=1, col=1
    )

    # Bar plot for occurrences in the second row
    fig.add_trace(
        go.Bar(
            x=df_occurrences["classes"],
            y=df_occurrences["occurrences"],
            text=df_occurrences["occurrences"],
            textposition="outside",
            name="Occurrences"
        ),
        row=2, col=1
    )

    # Prepare data for the heatmap (confidence scores across images and objects)
    object_types = sorted(class_scores.keys())
    image_names = list(json_data.keys())
    confidence_matrix = []

    for image_name in image_names:
        row = []
        for obj in object_types:
            scores = json_data[image_name]["predicted_data"].get(obj, [])
            avg_score = np.mean(scores) if scores else 0  # Use average score, or 0 if object not present
            row.append(avg_score)
        confidence_matrix.append(row)

    # Heatmap for confidence scores in the third row
    fig.add_trace(
        go.Heatmap(
            z=confidence_matrix,
            x=object_types,
            y=image_names,
            colorscale='Viridis',
            colorbar=dict(title="Confidence Score"),
            name="Confidence Heatmap"
        ),
        row=3, col=1
    )

    # Prepare data for the box plot of confidence scores for each object type
    box_plot_data = {obj: scores for obj, scores in class_scores.items()}

    # Box plot for confidence scores in the fourth row
    for obj, scores in box_plot_data.items():
        fig.add_trace(
            go.Box(
                y=scores,
                name=obj,
                boxmean=True,
                boxpoints='all',  # Show all points for better insights
            ),
            row=4, col=1
        )

    # Update layout for the entire figure
    fig.update_layout(
        title="Object Detection Analysis",
        showlegend=False,
        height=1800  # Adjust height for better spacing
    )
    fig.update_xaxes(title_text="Objects", row=1, col=1)
    fig.update_yaxes(title_text="Confidence Level", row=1, col=1)
    fig.update_xaxes(title_text="Objects", row=2, col=1)
    fig.update_yaxes(title_text="Occurrences", row=2, col=1)
    fig.update_xaxes(title_text="Object Type", row=3, col=1)
    fig.update_yaxes(title_text="Image Name", row=3, col=1)
    fig.update_xaxes(title_text="Object Type", row=4, col=1)
    fig.update_yaxes(title_text="Confidence Score", row=4, col=1)

    # Save plot as an image
    fig.write_image("dashboard.png", format="png", width=1600, height=2000, scale=2)

    return "dashboard.png"
##################################################################

def langchain_response(json_file, query):
    if query == "show dashboard":
        with open('predicted_items.json', 'r') as file:
            json_data = json.load(file)
        dashboard_plot(json_data)
        return send_to_telegram('Below is the dashboard based on objects data:', ["dashboard.png"])
    elif len(query.split()) > 1:
        # Step 1: Open the file and load the JSON data
        with open(json_file, 'r') as file:
            json_data = json.load(file)

        # Step 2: Convert the JSON object to a string
        json_string = json.dumps(json_data, indent=4)

        client = openaiconnect(api_key=your_openai_api_key)

        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "assistant",
                    "content": f"{query}, use this data to answer - {json_string}",
                }
            ],
            model="gpt-3.5-turbo",#"gpt-3.5-turbo", #"gpt-4-0613"
        )
        return send_to_telegram(chat_completion.choices[0].message.content, [])
    else:
        agent_response = agent_executor.invoke({"input": str(query).lower()})
        return handle_agent_response(agent_response, query)

def langchain_response_alt(json_file, query): ### dont need it anymore
    if len(query.split()) > 1:
        # Step 1: Open the file and load the JSON data
        with open(json_file, 'r') as file:
            json_data = json.load(file)
        # Set up the LLM model, we will be using the latest OpenAI model, GPT-4o for the best reasoning and understanding.
        llm = ChatOpenAI(model="gpt-4o", temperature=0.9, api_key=your_openai_api_key)
        # Set up the JsonToolkit and JsonSpec for the agent using the latest toolkit: https://python.langchain.com/v0.2/docs/integrations/toolkits/json/
        json_spec = JsonSpec(dict_=json_data, max_value_length=4000)
        json_toolkit = JsonToolkit(spec=json_spec)
        # Create the JSON agent executor and pass the LLM and JsonToolkit
        json_agent_executor = create_json_agent(
            llm=llm, toolkit=json_toolkit, verbose=True
        )
        #json_agent_executor.run(f"{query}")
        return send_to_telegram(json_agent_executor.run(f"{query}"), [])
    else:
        agent_response = agent_executor.invoke({"input": str(query).lower()})
        return handle_agent_response(agent_response, query)


# LangChain tools setup
tools = [
    Tool(
        name="GetImageForItem",
        func=query_item_image,
        description="Use this tool to find detected items in auction images." #"Use this tool to get the image path for a specific item found in auction images.",
    )
]

# Create a simple LangChain agent using OpenAI model
llm = ChatOpenAI(model="gpt-4", openai_api_key=your_openai_api_key)
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

###-->
prompt = hub.pull("hwchase17/react")
#prompt = hub.pull("hwchase17/structured-chat-agent")
#memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
#agent = create_structured_chat_agent(llm=llm, prompt=prompt, tools=tools)
agent = create_react_agent(
    llm=llm,
    prompt=prompt,
    tools=tools,
    stop_sequence=True
)
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True
)


###<--

# send this message at the begining
msg = f"Hi, this is a channel to find different items in storage units. \nCurrently I am looking for storage units at {web_loc.split('/')[-2].capitalize()}"
url_tel = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text={msg}"
requests.get(url_tel).json()


# Example usage - for testing purposes
#user_query = "bicyles"
#agent_response = agent_executor.invoke({"input": user_query})
#response = agent.run(user_query)
#print(agent_response['output'])
# Handle the agent's response to send formatted text and images to Telegram
#handle_agent_response(agent_response['output'], user_query)

def balance_command():
    url = f"https://api.telegram.org/bot{TOKEN}/getUpdates?offset=-1"
    t_data = requests.get(url).json()

    if t_data['result']:
        message_text = t_data['result'][0]['channel_post']['text'].lower()
        message_id = t_data['result'][0]['channel_post']['message_id']

        return message_text, message_id
    else:
        return None, None

last_message_id = None


# run telegram bot constantly
while True:
    try:
        message_text, message_id = balance_command()
        if message_id != last_message_id:
            #agent_response = agent_executor.invoke({"input": str(message_text).lower()})
            #handle_agent_response(agent_response['output'], str(message_text).lower())
            langchain_response("predicted_items.json", message_text) #if meassage_text is more than one word, it will use OpenAI API, else it will use agent_executor

            # send this message when finished object detection
            words = message_text.split()
            if len(words) <= 1:
                msg = f"I've finished checking storage units for {str(message_text).upper()}.\nIf you want to look for another specific item in the storage units, just send the name of the item to this chat.\nIf you want to see visual summary statistics for detected items, just type 'show dashboard'.\nIf you want to see the list of items found or any other questions related to items in storage units, feel free to type anything."
                url_tel = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text={msg}"
                requests.get(url_tel).json()
            last_message_id = message_id

    # Handle OpenAI API connection errors
    #except APIConnectionError as e:
    #    print(f"OpenAI API connection error: {e}. Retrying in 10 seconds...")
    #    time.sleep(10)

    # Handle connection timeouts when sending messages
    except ConnectTimeout:
        print("Telegram connection timed out. Retrying in 5 seconds...")
        time.sleep(5)

    # Handle other request-related errors
    except RequestException as e:
        print(f"Request error: {e}. Retrying in 5 seconds...")
        time.sleep(5)

    # Catch any other unexpected exceptions
    except Exception as e:
        print(f"An unexpected error occurred: {e}. Retrying in 5 seconds...")
        time.sleep(5)
