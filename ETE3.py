import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from PIL import Image, ImageFilter
import random
from datetime import datetime, timedelta
import calendar
from collections import Counter
import io
import requests

# Set page config
st.set_page_config(
    page_title="Cultural Events Analytics",
    layout="wide",
    page_icon="🎭"
)

# Custom color scheme
COLORS = {
    "deep_purple": "#6A0DAD",
    "vibrant_pink": "#FF0080",
    "electric_blue": "#0077FF",
    "golden_yellow": "#FFD700",
    "neon_green": "#39FF14",
    "fiery_orange": "#FF4500",
    "dark_charcoal": "#121212",
    "graffiti_gray": "#444444"
}

# Apply custom CSS
def set_custom_style():
    custom_css = f"""
    <style>
        /* Main background */
        .stApp {{
            background-color: {COLORS['dark_charcoal']};
            color: white;
        }}

        /* Sidebar */
        .css-1d391kg {{
            background-color: {COLORS['dark_charcoal']} !important;
            border-right: 1px solid {COLORS['vibrant_pink']};
        }}

        /* Calendar Styles */
        .calendar-container {{
            font-family: Arial, sans-serif;
            max-width: 900px;
            margin: 0 auto 30px;
        }}

        .calendar-month {{
            font-size: 24px;
            font-weight: bold;
            text-align: center;
            margin-bottom: 15px;
            color: {COLORS['vibrant_pink']};
        }}

        .calendar-weekdays {{
            display: grid;
            grid-template-columns: repeat(7, 1fr);
            text-align: center;
            font-weight: bold;
            margin-bottom: 10px;
            color: {COLORS['golden_yellow']};
        }}

        .calendar-days {{
            display: grid;
            grid-template-columns: repeat(7, 1fr);
            gap: 5px;
        }}

        .calendar-day {{
            padding: 10px;
            text-align: center;
            border-radius: 4px;
            background-color: {COLORS['graffiti_gray']};
            min-height: 80px;
            color: white;
        }}

        .calendar-day.empty {{
            background-color: transparent;
            border: none;
        }}

        .calendar-day.has-events {{
            background-color: {COLORS['deep_purple']};
            color: white;
            font-weight: bold;
            border: 1px solid {COLORS['vibrant_pink']};
            cursor: pointer;
        }}

        .calendar-day.has-events:hover {{
            background-color: {COLORS['vibrant_pink']};
        }}

        .event-count {{
            font-size: 12px;
            margin-top: 5px;
            color: {COLORS['neon_green']};
        }}

        .participant-details {{
            background-color: {COLORS['graffiti_gray']};
            border-radius: 8px;
            padding: 15px;
            margin-top: 20px;
            border-left: 4px solid {COLORS['electric_blue']};
        }}

        .image-gallery {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}

        .image-container {{
            position: relative;
            border-radius: 8px;
            overflow: hidden;
            border: 2px solid {COLORS['graffiti_gray']};
        }}

        .image-container:hover {{
            border-color: {COLORS['vibrant_pink']};
        }}

        .word-frequency {{
            background-color: {COLORS['graffiti_gray']};
            border-radius: 8px;
            padding: 15px;
            margin-top: 10px;
        }}
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

# Function to display banner image
def add_banner():
    try:
        banner = Image.open("Inbloom.png")
        st.image(banner, use_container_width=True)
    except FileNotFoundError:
        # Create a gradient banner if image not found
        gradient = Image.new('RGB', (1200, 200))
        pixels = gradient.load()
        for i in range(gradient.size[0]):
            for j in range(gradient.size[1]):
                # Gradient from deep purple to vibrant pink
                r = int(106 + (255-106) * i/gradient.size[0])
                g = int(13 + (0-13) * i/gradient.size[0])
                b = int(173 + (128-173) * i/gradient.size[0])
                pixels[i,j] = (r, g, b)

        st.image(gradient, use_container_width=True,
                caption="Add Inbloom.png to this folder for custom banner")

# Sample data for generation
colleges = ["St. Xavier's", "Loyola", "Christ University", "Presidency", "Jain University",
            "Mount Carmel", "Kristu Jayanti", "Bangalore University", "NMIMS", "Symbiosis"]

states = ["Karnataka", "Maharashtra", "Tamil Nadu", "Kerala", "Delhi",
          "West Bengal", "Telangana", "Andhra Pradesh", "Gujarat", "Rajasthan"]

events = ["Music Competition", "Dance Performance", "Drama", "Poetry Slam",
          "Debate Competition", "Art Exhibition", "Photography Contest",
          "Fashion Show", "Stand-up Comedy", "Short Film Screening"]

feedback_phrases = [
    "Amazing event, well organized", "Could improve time management",
    "Loved the performances", "Judges were very professional",
    "Venue was too crowded", "Sound system needs improvement",
    "Great participation from all colleges", "Food arrangements were excellent",
    "Looking forward to next year", "Some events started late",
    "Very competitive atmosphere", "Excellent coordination by team",
    "More seating required", "Sound quality was poor",
    "Loved the cultural diversity", "Need better signage for venues"
]

# Generate random dates for the 5-day event
def generate_dates():
    start_date = datetime(2025, 3, 1)
    return [start_date + timedelta(days=i) for i in range(5)]

# Generate participant data
def generate_dataset():
    data = []
    dates = generate_dates()

    for i in range(1, 251):
        participant = {
            "Participant_ID": f"P{i:03d}",
            "Name": f"Participant {i}",
            "College": random.choice(colleges),
            "State": random.choice(states),
            "Event": random.choice(events),
            "Day": random.choice(dates),
            "Rating": random.randint(1, 5),
            "Feedback": random.choice(feedback_phrases),
            "Contact": f"{random.randint(7000000000, 9999999999)}",
            "Registration_Fee": random.choice([0, 100, 150, 200])
        }
        data.append(participant)

    df = pd.DataFrame(data)
    df['Day'] = pd.to_datetime(df['Day']).dt.date
    return df

# Generate word cloud
def generate_wordcloud(text):
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color=COLORS['dark_charcoal'],
        colormap='viridis'
    ).generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    return plt

# Get top words from feedback
def get_top_words(feedback_text, n=10):
    words = feedback_text.lower().split()
    words = [word for word in words if len(word) > 3]  # Ignore short words
    word_counts = Counter(words)
    return word_counts.most_common(n)

# Image processing functions
def process_image(image, operation):
    img = image.copy()
    if operation == "Grayscale":
        return img.convert('L')
    elif operation == "Sepia":
        sepia_filter = np.array([[0.393, 0.769, 0.189],
                                [0.349, 0.686, 0.168],
                                [0.272, 0.534, 0.131]])
        sepia_img = np.dot(np.array(img), sepia_filter.T)
        sepia_img = np.clip(sepia_img, 0, 255).astype(np.uint8)
        return Image.fromarray(sepia_img)
    elif operation == "Blur":
        return img.filter(ImageFilter.BLUR)
    elif operation == "Contour":
        return img.filter(ImageFilter.CONTOUR)
    else:
        return img

# Display calendar with participant details on click
def display_calendar_with_participants(year, month, df):
    # Get calendar data
    cal = calendar.monthcalendar(year, month)
    month_name = calendar.month_name[month]

    # Create a dictionary of participants by day
    participants_by_day = {}
    for day in df['Day'].unique():
        participants_by_day[day] = df[df['Day'] == day]

    # Display month header
    st.markdown(f"<div class='calendar-month'>MONTH : {month_name.upper()} {year}</div>",
               unsafe_allow_html=True)

    # Display weekday headers
    weekdays = ["SUNDAY", "MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY"]
    cols = st.columns(7)
    for i, col in enumerate(cols):
        col.markdown(f"<div class='calendar-weekdays'>{weekdays[i]}</div>",
                    unsafe_allow_html=True)

    # Track selected day
    selected_day = st.session_state.get('selected_day', None)

    # Filters for college and event
    selected_college = st.sidebar.selectbox("Filter by College", ["All"] + colleges)
    selected_event = st.sidebar.selectbox("Filter by Event", ["All"] + events)

    # Filter data based on selected college and event
    if selected_college != "All":
        df = df[df['College'] == selected_college]
    if selected_event != "All":
        df = df[df['Event'] == selected_event]

    # Display calendar days
    for week in cal:
        cols = st.columns(7)
        for i, day in enumerate(week):
            with cols[i]:
                if day == 0:
                    st.markdown('<div class="calendar-day empty"></div>',
                               unsafe_allow_html=True)
                else:
                    current_date = datetime(year, month, day).date()
                    day_participants = participants_by_day.get(current_date, pd.DataFrame())
                    count = len(day_participants)

                    if count > 0:
                        if st.button(f"{day} \n \n {count} participants",
                                    key=f"day_{day}",
                                    help=f"Click to view {count} participants"):
                            selected_day = current_date
                            st.session_state['selected_day'] = selected_day
                    else:
                        st.markdown(f'<div class="calendar-day">{day}</div>',
                                   unsafe_allow_html=True)

    # Display participant details for selected day
    if selected_day:
        day_participants = participants_by_day.get(selected_day, pd.DataFrame())
        if not day_participants.empty:
            st.markdown(f"<h3>Participants on {selected_day.strftime('%B %d, %Y')}</h3>",
                       unsafe_allow_html=True)
            st.dataframe(day_participants[['Participant_ID', 'Name', 'College', 'Event']])
        else:
            st.warning(f"No participants found for {selected_day.strftime('%B %d, %Y')}")

# Main App
def main():
    # Apply custom styles
    set_custom_style()

    # Add the banner image at the very top
    add_banner()

    # Generate or load dataset
    if 'df' not in st.session_state:
        st.session_state.df = generate_dataset()

    df = st.session_state.df

    # Sidebar navigation
    with st.sidebar:
        st.header("Navigation")
        app_mode = st.radio(
            "Select Section",
            ["Calendar View", "Dashboard", "Feedback Analysis", "Image Gallery"],
            index=0
        )

        # Add filters to sidebar when in Dashboard mode
        if app_mode == "Dashboard":
            st.header("Filters")
            selected_events = st.multiselect("Select Events", events, default=events[:3])
            selected_states = st.multiselect("Select States", states, default=states[:3])
            selected_colleges = st.multiselect("Select Colleges", colleges, default=colleges[:3])

        # Add event selector when in Feedback Analysis mode
        elif app_mode == "Feedback Analysis":
            selected_event = st.selectbox("Select Event for Feedback Analysis", events)

    # Main content area
    if app_mode == "Calendar View":
        st.header("Event Calendar")
        display_calendar_with_participants(2025, 3, df)

    elif app_mode == "Dashboard":
        st.header("Participation Analytics Dashboard")
        st.write("Interactive visualizations of participation trends.")

        # Filter data
        filtered_df = df[
            (df['Event'].isin(selected_events if selected_events else events)) &
            (df['State'].isin(selected_states if selected_states else states)) &
            (df['College'].isin(selected_colleges if selected_colleges else colleges))
        ]

        # Visualizations - only show if enough data
        if len(selected_events or []) >= 2:
            st.subheader("Event-wise Participation")
            event_counts = filtered_df['Event'].value_counts()
            if len(event_counts) > 0:
                fig, ax = plt.subplots()
                event_counts.plot(kind='bar', ax=ax, color=COLORS['vibrant_pink'])
                plt.xticks(rotation=45)
                st.pyplot(fig)

        if len(selected_states or []) >= 2:
            st.subheader("State-wise Participation")
            state_counts = filtered_df['State'].value_counts()
            if len(state_counts) > 0:
                fig, ax = plt.subplots()
                sns.barplot(x=state_counts.index, y=state_counts.values, ax=ax, palette=[
                    COLORS['golden_yellow'], COLORS['neon_green'], COLORS['fiery_orange']]
                )
                plt.xticks(rotation=45)
                st.pyplot(fig)

        if len(selected_colleges or []) >= 2:
            st.subheader("College-wise Participation")
            college_counts = filtered_df['College'].value_counts()
            if len(college_counts) > 0:
                fig, ax = plt.subplots()
                college_counts.plot(kind='pie', ax=ax, autopct='%1.1f%%', colors=[
                    COLORS['deep_purple'], COLORS['vibrant_pink'], COLORS['electric_blue']]
                )
                st.pyplot(fig)

        # Always show day-wise participation
        st.subheader("Day-wise Participation")
        day_counts = filtered_df['Day'].value_counts().sort_index()
        fig, ax = plt.subplots()
        day_counts.plot(kind='line', marker='o', ax=ax, color=COLORS['electric_blue'])
        plt.xticks(rotation=45)
        st.pyplot(fig)

    elif app_mode == "Feedback Analysis":
        st.header("Participant Feedback Analysis")

        # Get feedback for selected event
        event_feedback = ' '.join(df[df['Event'] == selected_event]['Feedback'])

        if event_feedback:
            st.subheader(f"Word Cloud for {selected_event}")
            wordcloud_fig = generate_wordcloud(event_feedback)
            st.pyplot(wordcloud_fig)

            st.subheader("Feedback Sentiment")
            avg_rating = df[df['Event'] == selected_event]['Rating'].mean()
            st.metric("Average Rating", f"{avg_rating:.1f}/5")

            st.subheader("Top 10 Words in Feedback")
            top_words = get_top_words(event_feedback)
            if top_words:
                words_df = pd.DataFrame(top_words, columns=['Word', 'Count'])
                st.dataframe(words_df)

            st.subheader("Sample Feedback")
            feedback_samples = df[df['Event'] == selected_event]['Feedback'].sample(5).values
            for feedback in feedback_samples:
                st.markdown(f"""
                <div class="highlight-box">
                    {feedback}
                </div>
                """, unsafe_allow_html=True)

    elif app_mode == "Image Gallery":
        st.header("Event Image Gallery")

        # Image upload and URL input
        col1, col2 = st.columns(2)
        with col1:
            uploaded_files = st.file_uploader("Upload images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        with col2:
            image_urls = st.text_area("Or enter image URLs (one per line)")

        # Initialize session state for image processing
        if 'images' not in st.session_state:
            st.session_state.images = []
        if 'processed_images' not in st.session_state:
            st.session_state.processed_images = {}

        # Load images from either upload or URLs
        if uploaded_files:
            for uploaded_file in uploaded_files:
                image = Image.open(uploaded_file)
                st.session_state.images.append(image)
        if image_urls:
            urls = image_urls.strip().split("\n")
            for url in urls:
                try:
                    image = Image.open(io.BytesIO(requests.get(url).content))
                    st.session_state.images.append(image)
                except:
                    st.error(f"Could not load image from URL: {url}")

        # Display and process images
        if st.session_state.images:
            st.subheader("Image Gallery")
            for i, image in enumerate(st.session_state.images):
                col1, col2 = st.columns(2)
                with col1:
                    st.image(image, caption=f"Image {i+1}", use_container_width=True)
                with col2:
                    st.subheader("Processing Options")
                    operation = st.selectbox(
                        f"Select processing operation for Image {i+1}",
                        ["None", "Grayscale", "Sepia", "Blur", "Contour"],
                        key=f"operation_{i}"
                    )
                    if operation != "None":
                        processed_image = process_image(image, operation)
                        st.session_state.processed_images[i] = processed_image
                        st.image(processed_image, caption=f"Processed Image {i+1}", use_container_width=True)
                        # Download button for processed image
                        buf = io.BytesIO()
                        processed_image.save(buf, format="PNG")
                        byte_im = buf.getvalue()
                        st.download_button(
                            label=f"Download Processed Image {i+1}",
                            data=byte_im,
                            file_name=f"processed_image_{i+1}.png",
                            mime="image/png",
                            key=f"download_{i}"
                        )
                    else:
                        st.info("Select a processing operation")

if __name__ == "__main__":
    main()
