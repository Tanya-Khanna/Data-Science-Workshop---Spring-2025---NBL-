## Import necessary modules for handling prompts, using a language model, creating a web app, and getting YouTube transcripts.
from langchain_core.prompts import ChatPromptTemplate  # Allows us to create and manage structured prompts for large language models (LLMs).
# LangChain is a toolkit that helps build apps using LLMs by offering tools to manage prompts, keep conversation history, and connect to different models and data sources.

from langchain_community.llms import Ollama  # Adds support for Ollama, a tool to run large language models locally or on private servers.
# Ollama works with LangChain and lets us run LLMs on our own systems, which is useful for keeping data private and secure.

import streamlit as st  # Streamlit helps us make interactive web apps quickly, great for building user-friendly ML and data interfaces.

from youtube_transcript_api import YouTubeTranscriptApi  # Lets us pull transcripts directly from YouTube videos, which is useful for tasks like summarizing or analyzing video content.

from urllib.parse import urlparse, parse_qs  # Provides tools to break down URLs and get specific parts, like video IDs from YouTube links.


# Page setting
st.set_page_config(layout="wide")  # Sets the page layout in Streamlit to "wide" for a more spacious and flexible user interface.

# Initialize Ollama with the 3B model and optimized parameters
llm = Ollama(
    model="llama3.2",  # Specifies the 3B version of the Llama model for efficient and high-quality language generation.
    temperature=0.1,  # Sets a low temperature for more predictable and focused responses, reducing randomness in outputs.
    num_ctx=2048,     # Sets the context window to 2048 tokens, which fits well with the 3B model’s capacity to process larger text inputs.
    top_p=0.8,        # Uses top-p sampling to focus on the highest-probability tokens, promoting more concise answers.
    num_predict=512,  # Limits the generated response to 512 tokens, preventing overly long outputs.
)


# Optimized prompt template for the instruct model
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a focused AI assistant analyzing video content. Instructions:
    - Provide clear, concise answers
    - Focus on the most important information
    - If unsure, admit it
    - Use bullet points for clarity when appropriate"""),  # System message gives the model clear instructions on answering style and focus.

    ("user", """Here is the video transcript:
    {content}
    
    Question: {question}
    
    Provide a clear and structured answer based on the transcript.""")  # User message template where {content} is the video transcript, and {question} is the specific question for the model to answer.
])

# This prompt setup provides the model with a structured format to follow, encouraging brief, relevant responses and clarity.


# Create chain
chain = prompt | llm  # Combines the prompt template and the language model into a chain for streamlined processing of inputs and outputs.

st.title("🎬🤖 YT Buddy: Your AI Video Companion")  # Sets the title for the Streamlit app, displaying it at the top of the page.

# Initialize session state
if "content" not in st.session_state:
    st.session_state.content = ""  # Initializes the session state variable "content" to an empty string if it doesn’t already exist.
    # This ensures that the content variable persists across interactions within the session, allowing the user to store and reuse data in the app.


def clean_transcript(text):
    """Clean and format transcript text for model input"""
    
    # Basic cleaning to remove unnecessary whitespace
    text = ' '.join(text.split())  # Splits the text into words and joins them back with a single space between each,
                                   # effectively removing any extra whitespace or newlines for a cleaner input.

    # Remove non-verbal indicators like '[Music]' or '[Applause]'
    text = text.replace('[Music]', '')  # Removes '[Music]' from the text, commonly used in transcripts to indicate background music,
                                        # ensuring that only spoken content is processed by the model.
    text = text.replace('[Applause]', '')  # Removes '[Applause]' from the text, which often marks audience applause,
                                           # keeping the text focused on actual speech for more accurate analysis.

    # Limit length of the text to fit within the Llama 3B model’s context window
    # The Llama 3B model has a limited context capacity, so restricting the length prevents overload and improves performance.
    return text[:4000]  # Trims the text to the first 4000 characters, ensuring it doesn't exceed the model's processing capacity.
                        # This limit balances retaining as much of the transcript as possible while fitting within model constraints.


def get_transcript_content(url):
    content = ""  # Initialize an empty string to store the cleaned transcript content
    
    try:
        # Parse the URL to extract the query parameters
        parsed_url = urlparse(url)  # Break down the URL into components (scheme, netloc, path, query, etc.)
        query_params = parse_qs(parsed_url.query)  # Extract query parameters as a dictionary
        video_id = query_params.get('v', [None])[0]  # Get the 'v' parameter (video ID), or None if it's missing
        
        # Check if the video ID is present
        if not video_id:
            st.error("Invalid YouTube URL")  # Display an error message in Streamlit if the URL lacks a valid video ID
            return content  # Return the empty content string early since there's no valid video ID
            
        # Fetch transcript using the video ID
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)  # Retrieve the transcript as a list of dictionaries, each containing text and timing info
        content = " ".join([t["text"] for t in transcript_list])  # Combine all text segments into a single string

        # Clean and format the transcript content
        content = clean_transcript(content)  # Apply the clean_transcript function to remove unwanted elements and trim the text
            
    except Exception as e:
        # Catch any errors during the process and display an error message in Streamlit
        st.error(f"Error getting transcript: {str(e)}")  # Show the error message with details for easier debugging
    
    return content  # Return the cleaned and formatted transcript content

# Suggested questions optimized for the 3B model to encourage concise responses
SUGGESTED_QUESTIONS = [
    "What is the main topic covered in this video?",
    "List the key points in bullet points.",
    "What are the top takeaways from this content?",
    "Can you summarize the video briefly?",
    "Are there any important examples or facts mentioned?",
]

def main_page():
    # Display page title and description
    st.markdown("""
    ### 📺 YouTube Video Analysis
    Using Llama 3.2B Instruct model for concise video content analysis.
    
    **Tips:**
    - Ask short, specific questions
    - Use bullet points option for clearer summaries
    - Keep questions focused on main content
    """)  # Provides an introductory section for users with instructions and tips.

    # Input field for YouTube URL
    url = st.text_input(
        "Enter YouTube URL",
        value="https://www.youtube.com/watch?v=LWMzyfvuehA&t=1369s"  # Provides a default example URL for user convenience.
    )

    # Two-column layout for buttons
    col1, col2 = st.columns([1, 1])  # Creates two equal-width columns for the button and checkbox.
    with col1:
        clicked = st.button("Load Video", type="primary")  # Button to trigger transcript loading.
    with col2:
        use_bullets = st.checkbox("Use bullet points in response", value=True)  # Checkbox to toggle bullet points in responses.

    # Load transcript if button is clicked
    if clicked:
        with st.spinner("Loading transcript..."):  # Shows a loading spinner while fetching the transcript.
            content = get_transcript_content(url)  # Calls function to fetch and clean the transcript.
            if content:
                st.session_state.content = content  # Saves transcript in session state for later access.
                st.success("Loaded!")  # Success message on successful transcript loading.
            else:
                st.error("Could not load transcript")  # Error message if transcript loading fails.

    # Display transcript and allow questions if content is available
    if st.session_state.content:
        col1, col2 = st.columns([4, 6])  # Create a layout with wider space for user interactions.

        with col1:
            with st.expander("Video Transcript", expanded=True):  # Shows transcript in an expandable section.
                st.write(st.session_state.content)  # Displays the loaded transcript content.
            
            st.markdown("### Quick Questions")
            for q in SUGGESTED_QUESTIONS:
                if st.button(q):  # Display suggested questions as buttons.
                    if use_bullets:
                        q += " Please format as bullet points."  # Appends bullet point instruction if checkbox is selected.
                    st.session_state.current_question = q  # Sets the clicked question as the current question.

        with col2:
            question = st.text_area(
                "Ask about the video:",
                value=getattr(st.session_state, 'current_question', "What is this video about?"),  # Default question prompt.
                height=100  # Sets the height of the text area for user input.
            )
            
            if question:
                with st.spinner("Analyzing..."):  # Shows spinner while processing the question.
                    try:
                        # Modify question to include bullet points if requested by the user
                        if use_bullets and "bullet points" not in question.lower():
                            question += " Please format as bullet points."
                            
                        response = chain.invoke({
                            "content": st.session_state.content,
                            "question": question
                        })  # Sends content and question to the chain for AI analysis.
                        
                        with st.container(border=True):
                            st.markdown("### Analysis:")
                            st.write(response)  # Displays the model's response to the question.
                            
                    except Exception as e:
                        st.error(f"Error: {str(e)}")  # Error handling if chain invocation fails.
                        st.info("Make sure Ollama is running with the correct model")  # Additional tip for troubleshooting.


if __name__ == "__main__":
    main_page()  # Entry point for the Streamlit app. Runs the main_page function if the script is executed directly.
    # This ensures that the main_page function is only called when this file is run as the main program,
    # and not when imported as a module in another script.
